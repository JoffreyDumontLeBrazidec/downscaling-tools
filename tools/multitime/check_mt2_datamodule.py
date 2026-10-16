#!/usr/bin/env python3
"""Build Vera's datamodule from the mt2 config, pull one training batch, check the forecast blocks.

Piece 1 of the mt2_e2_20k preparation: does Vera Gahlen's branch (fork claude/project-thread-n1zaq3,
HindcastBlockDataset) read the E2 hindcast stores through the lane fork of anemoi-datasets
(/home/ecm5702/dev/anemoi-datasets, the one with fake_hindcasts) without a port?

It composes the config with Hydra (the branch's search path plugin supplies the default groups),
builds AnemoiDatasetsDataModule exactly as AnemoiTrainer does, and then:
  1. prints which anemoi.datasets module is imported (it must be the lane fork);
  2. logs every attribute and call the branch's readers make on the opened datasets (a proxy
     around open_dataset), i.e. the anemoi-datasets API they exercise;
  3. prints rows, forecast blocks and anchors of every reader of the three splits;
  4. pulls ONE training batch through the real DataLoader (one worker) and prints its shapes
     (expect time length 2 for in_lres, in_hres and out_hres);
  5. for a few anchors per split, maps both slots of every dataset back to raw store rows and to
     the store's fake_hindcasts mapping (reference date, step), and checks that the two slots are
     the same forecast 12 h apart, that the three datasets read the same rows, and that the values
     equal the raw zarr rows;
  6. prints the data indices (model input / output names) and loads the residual statistics.

Usage (Vera x86 venv, never its activate.sh):
  PYTHONPATH=<core>/models/src:<core>/training/src:<core>/graphs/src:/home/ecm5702/dev/anemoi-datasets/src \
  OUTPUT=/tmp/... python check_mt2_datamodule.py --config-dir <S>/configs --config-name mt2_e2_20k
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


CALLS: collections.Counter = collections.Counter()


class Proxy:
    """Wraps an opened anemoi dataset and counts what the reader touches."""

    def __init__(self, ds, tag):
        object.__setattr__(self, "_ds", ds)
        object.__setattr__(self, "_tag", tag)

    def __getattr__(self, name):
        CALLS[f"attr {name}"] += 1
        return getattr(object.__getattribute__(self, "_ds"), name)

    def __getitem__(self, key):
        kind = type(key).__name__
        if isinstance(key, tuple):
            kind = "tuple(" + ",".join(type(k).__name__ for k in key) + ")"
        CALLS[f"__getitem__ {kind}"] += 1
        return object.__getattribute__(self, "_ds")[key]

    def __len__(self):
        CALLS["__len__"] += 1
        return len(object.__getattribute__(self, "_ds"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-dir", required=True)
    p.add_argument("--config-name", default="mt2_e2_20k")
    p.add_argument("--anchors-per-split", type=int, default=4)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    import anemoi.datasets
    import anemoi.training
    import anemoi.models
    import torch

    log(f"anemoi.datasets {anemoi.datasets.__file__} version {getattr(anemoi.datasets, '__version__', '?')}")
    log(f"anemoi.training {anemoi.training.__file__}")
    log(f"anemoi.models {anemoi.models.__file__}; torch {torch.__version__}")

    import anemoi.training.data.data_reader as dr

    orig_open = dr.open_dataset

    def logged_open(*a, **k):
        cfg = a[0] if a else None
        keys = sorted(cfg) if isinstance(cfg, dict) else type(cfg).__name__
        CALLS[f"open_dataset(config keys {keys}, kwargs {sorted(k)})"] += 1
        return Proxy(orig_open(*a, **k), str(cfg.get("dataset") if isinstance(cfg, dict) else cfg))

    dr.open_dataset = logged_open

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import DictConfig, OmegaConf

    from anemoi.training.data.datamodule import AnemoiDatasetsDataModule
    from anemoi.training.schemas.base_schema import UnvalidatedBaseSchema, convert_to_omegaconf

    with initialize_config_dir(config_dir=os.path.abspath(args.config_dir), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=["dataloader.num_workers.training=1"])
    config = convert_to_omegaconf(UnvalidatedBaseSchema(**DictConfig(OmegaConf.to_object(cfg))))
    log("config composed and resolved (as AnemoiTrainer does with config_validation false)")

    task = instantiate(config.task)
    log(f"task {type(task).__name__}: input offsets {task._input_offsets}, output offsets {task._output_offsets}")
    dm = AnemoiDatasetsDataModule(config, task)

    summary = {"splits": {}}
    for label, attr in (("training", "ds_train"), ("validation", "ds_valid"), ("test", "ds_test")):
        ds = getattr(dm, attr)
        info = {"anchors": int(len(ds.anchors)), "relative_date_indices": {k: str(v) for k, v in ds.relative_date_indices.items()}}
        for name, reader in ds.data_readers.items():
            info[name] = {"reader": type(reader).__name__, "rows": int(len(reader.data.dates)),
                          "blocks": int(reader.num_sequences), "steps_per_block": int(reader.sequence_length()),
                          "frequency": str(reader.frequency), "n_variables": len(reader.variables),
                          "grid": int(reader.grid_size), "missing_rows": len(reader.missing)}
        summary["splits"][label] = info
        log(f"{label}: {json.dumps(info)}")

    # One training batch through the real DataLoader (one worker).
    dl = dm.train_dataloader()
    t = time.time()
    batch = next(iter(dl))
    shapes = {k: list(v.shape) for k, v in batch.items()}
    summary["training_batch_shapes"] = shapes
    log(f"one training batch in {time.time() - t:.1f} s: shapes {shapes} (batch, time, ensemble, grid, variables)")
    for k, v in batch.items():
        log(f"  {k}: dtype {v.dtype}, finite fraction {torch.isfinite(v).float().mean().item():.6f}, "
            f"slot means {[round(float(v[:, i].float().nanmean()), 4) for i in range(v.shape[1])]}")

    # Slot checks against the raw stores and their fake_hindcasts mapping.
    import zarr

    raw_cache = {}

    def raw_info(path):
        if path not in raw_cache:
            z = zarr.open(path, mode="r")
            attrs = dict(z.attrs)
            dates = np.asarray(z["dates"][:]).astype("datetime64[s]")
            raw_cache[path] = (z, attrs["fake_hindcasts"], {int(d.astype("int64")): i for i, d in enumerate(dates)},
                               list(attrs["variables"]), dates)
        return raw_cache[path]

    def ref_step(value):
        if isinstance(value, dict):
            ref = value.get("reference_date") or value.get("refdate") or value.get("base_date") or value.get("date")
            return str(ref), int(value["step"])
        return str(value[0]), int(value[-1])

    rng = np.random.default_rng(756)
    checks = []
    for label, attr in (("training", "ds_train"), ("validation", "ds_valid"), ("test", "ds_test")):
        ds = getattr(dm, attr)
        picks = rng.choice(len(ds.anchors), size=min(args.anchors_per_split, len(ds.anchors)), replace=False)
        for idx in picks:
            seq, pos = (int(v) for v in ds.anchors[idx])
            sample = ds.get_sample(int(idx))
            rows_by_ds = {}
            for name, reader in ds.data_readers.items():
                rel = ds.relative_date_indices[name]
                if isinstance(rel, slice):
                    rel = list(range(rel.start or 0, rel.stop, rel.step or 1))
                rows = reader._rows(seq, [pos + int(r) for r in np.asarray(rel).reshape(-1)])
                store = config.dataloader[label].datasets[name].dataset_config.dataset
                z, mapping, raw_index, raw_vars, raw_dates = raw_info(store)
                view_dates = np.asarray(reader.data.dates).astype("datetime64[s]")
                raw_rows = [raw_index[int(view_dates[r].astype("int64"))] for r in rows]
                rs = [ref_step(mapping[str(np.datetime_as_string(raw_dates[r], unit="s"))]) for r in raw_rows]
                # values: slot k of the sample equals the raw zarr row, variables mapped by name
                sel = list(reader.variables)
                cols = [raw_vars.index(v) for v in sel]
                gaps = []
                for k, r in enumerate(raw_rows):
                    raw = np.asarray(z["data"][r, :, 0, :])[cols].T  # (grid, vars)
                    got = sample[name][k, 0].numpy()
                    gaps.append(float(np.nanmax(np.abs(raw - got))))
                rows_by_ds[name] = raw_rows
                checks.append({"split": label, "anchor": [seq, pos], "dataset": name, "raw_rows": raw_rows,
                               "refdate_step": rs, "max_abs_vs_raw": gaps,
                               "same_forecast": len({r for r, _ in rs}) == 1,
                               "step_gap_h": rs[1][1] - rs[0][1] if len(rs) > 1 else None})
            same_rows = len({tuple(v) for v in rows_by_ds.values()}) == 1
            checks[-1]["three_datasets_same_rows"] = same_rows
    ok = all(c["same_forecast"] and c["step_gap_h"] == 12 and max(c["max_abs_vs_raw"]) == 0.0 for c in checks)
    ok &= all(c.get("three_datasets_same_rows", True) for c in checks)
    for c in checks:
        log(f"slot check {c}")
    summary["slot_checks_ok"] = bool(ok)
    log(f"SLOT CHECKS {'OK' if ok else 'FAILED'} ({len(checks)} dataset-anchor pairs)")

    di = dm.data_indices
    for name, ix in di.items():
        log(f"data indices {name}: model.input {len(ix.model.input)} {list(ix.model.input.name_to_index)}")
        log(f"data indices {name}: model.output {len(ix.model.output)} {list(ix.model.output.name_to_index)}")
        log(f"data indices {name}: prognostic {len(ix.prognostic) if hasattr(ix, 'prognostic') else '?'}, "
            f"diagnostic {[k for k in ix.name_to_index if ix.name_to_index[k] in set(ix.data.output.diagnostic.tolist())]}")
    rs = dm.residual_statistics
    log(f"residual statistics datasets {list(rs or {})}; out_hres stdev[:3] {rs['out_hres']['stdev'][:3]} "
        f"tp/cp {[float(rs['out_hres'][k][-1]) for k in ('mean', 'stdev')]}")

    summary["anemoi_datasets_api_calls"] = dict(CALLS)
    log("anemoi-datasets API exercised by the branch's readers (counts):")
    for k, v in sorted(CALLS.items()):
        log(f"  {k}: {v}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1, default=str)
    log("DONE " + ("OK" if ok else "WITH FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
