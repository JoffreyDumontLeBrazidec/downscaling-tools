#!/usr/bin/env python3
"""One statistics pass for B2L: the residual of the truth to a frozen deterministic base.

B2L (`epics/misc/20261014_regression_base_check_deterministic_model.md` section 10) trains a diffusion
model on ``y - base(x)``, where ``base(x) = interp(x) + B1L(x)`` is the base's full prediction. The lane's
residual normaliser (``system.input.residual_statistics``, mean/stdev per variable of ``y - interp(x)``) is
wrong for that target, so this script walks the training pool once, runs the base on every pair and writes
the per-variable statistics of ``y - base(x)`` in the same ``.npy`` dict format the datamodule reads
(``DownscalingAnemoiDatasetsDataModule._load_residual_statistics``: ``{"mean": {var: v}, "stdev": {...},
"minimum": {...}, "maximum": {...}}``). Point ``system.input.residual_statistics`` of the B2L arm at the
output and set ``system.input.base_checkpoint`` to the same base checkpoint.

The base forward is ``AnemoiD2ModelEncProcDec.regression_base_prediction`` of anemoi-core, the one function
the training step and the sampler also use, with the same sigma and noise seed, so the statistics describe
exactly the target the arm will see.

Usage (one GPU; the pool of 5,290 pairs at 12-24 h is about 0.5 s per pair on a GH200):

    python tools/b2l/base_residual_statistics.py \\
        --config  <rendered training config of the arm, yaml (hydra --cfg job render)> \\
        --base-checkpoint <inference-ema10000-anemoi-by_step-...ckpt of the base> \\
        --out <dir>/b2l_residual_stats.npy [--batch-size 1] [--max-batches N] [--device cuda] \\
        [--set dataloader.num_workers.training=0 ...]

``--max-batches 2`` is the smoke. A JSON sidecar (``<out>.json``) records provenance, the pair and point
counts and, when the config names the lane's old statistics, old against new stdev per variable (the "sigma
ladder moves toward the band" check of section 10 item 3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


# ---------------------------------------------------------------------------------------------- accumulator


class ResidualStatsAccumulator:
    """Per-variable count, sum, sum of squares, min and max over every (sample, grid point); NaN-aware.

    Sums are float64. ``update`` takes a residual tensor or array of shape (..., n_vars) and reduces every
    leading dimension.
    """

    def __init__(self, names: list[str]):
        self.names = list(names)
        n = len(self.names)
        self.count = np.zeros(n, dtype=np.int64)
        self.sum = np.zeros(n, dtype=np.float64)
        self.sumsq = np.zeros(n, dtype=np.float64)
        self.minimum = np.full(n, np.inf, dtype=np.float64)
        self.maximum = np.full(n, -np.inf, dtype=np.float64)
        self.n_samples = 0

    def update(self, residual) -> None:
        if hasattr(residual, "detach"):  # torch tensor
            residual = residual.detach().to("cpu", dtype=__import__("torch").float64).numpy()
        r = np.asarray(residual, dtype=np.float64)
        if r.shape[-1] != len(self.names):
            raise ValueError(f"residual has {r.shape[-1]} variables, expected {len(self.names)}")
        flat = r.reshape(-1, r.shape[-1])
        finite = np.isfinite(flat)
        self.count += finite.sum(axis=0)
        self.sum += np.where(finite, flat, 0.0).sum(axis=0)
        self.sumsq += np.where(finite, flat * flat, 0.0).sum(axis=0)
        self.minimum = np.minimum(self.minimum, np.where(finite, flat, np.inf).min(axis=0))
        self.maximum = np.maximum(self.maximum, np.where(finite, flat, -np.inf).max(axis=0))
        self.n_samples += int(np.prod(r.shape[:-1])) if r.ndim > 1 else 1

    def result(self) -> dict[str, dict[str, float]]:
        count = np.maximum(self.count, 1)
        mean = self.sum / count
        var = self.sumsq / count - mean * mean
        stdev = np.sqrt(np.maximum(var, 0.0))
        out = {"mean": {}, "stdev": {}, "minimum": {}, "maximum": {}}
        for i, name in enumerate(self.names):
            out["mean"][name] = float(mean[i])
            out["stdev"][name] = float(stdev[i])
            out["minimum"][name] = float(self.minimum[i]) if math.isfinite(self.minimum[i]) else float("nan")
            out["maximum"][name] = float(self.maximum[i]) if math.isfinite(self.maximum[i]) else float("nan")
        return out


def write_stats(path: Path, stats: dict, sidecar: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stats, allow_pickle=True)
    if sidecar is not None:
        Path(str(path) + ".json").write_text(json.dumps(sidecar, indent=1, default=str))


def output_variable_names(interface) -> list[str]:
    """Output variable names in model-output channel order (the order of ``y[..., data.output.full]``)."""
    target_ds = interface.model._decoder_datasets[0]
    nti = dict(interface.model.data_indices[target_ds].model.output.name_to_index)
    names = [None] * (max(nti.values()) + 1)
    for name, idx in nti.items():
        names[idx] = name
    if any(n is None for n in names):
        raise ValueError(f"output name_to_index of {target_ds} has gaps: {nti}")
    return names


def compare_with_old(old_path: str | None, new_stats: dict) -> dict | None:
    if not old_path:
        return None
    try:
        old = np.load(str(old_path), allow_pickle=True).item()
    except Exception as exc:  # noqa: BLE001 - the comparison is a courtesy
        return {"error": f"could not read {old_path}: {exc}"}
    rows = {}
    for name, new_std in new_stats["stdev"].items():
        old_std = old.get("stdev", {}).get(name)
        rows[name] = {
            "old_stdev": None if old_std is None else float(old_std),
            "new_stdev": float(new_std),
            "ratio_new_over_old": None if not old_std else float(new_std) / float(old_std),
        }
    return {"old_file": str(old_path), "per_variable": rows}


# ---------------------------------------------------------------------------------------------- the pass


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="rendered training config (yaml) of the arm or its trunk")
    ap.add_argument("--base-checkpoint", required=True, help="the base's inference-*.ckpt (or the Lightning ckpt next to it)")
    ap.add_argument("--out", required=True, help="output .npy (a .json sidecar is written next to it)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=0, help="stop after N batches (smoke); 0 = whole pool")
    ap.add_argument("--base-sigma", type=float, default=500000.0)
    ap.add_argument("--base-noise-seed", default="0", help="integer, or 'none' for fresh noise per call")
    ap.add_argument("--split", default="training", choices=["training", "validation"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="OmegaConf dotlist override of the config")
    return ap.parse_args(argv)


def _resolve_inference_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.name.startswith("inference-"):
        return p
    companion = p.with_name("inference-" + p.name)
    if companion.exists():
        return companion
    raise SystemExit(f"no inference checkpoint for {p} (need inference-{p.name})")


def main(argv=None) -> int:
    args = _parse_args(argv)
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(REPO_ROOT))
    from anemoi.models.models.diffusiondownscaler_encoder_processor_decoder import AnemoiD2ModelEncProcDec
    from manual_inference.checkpoints import get_datamodule  # the predict route's datamodule builder

    t0 = time.time()
    seed = None if str(args.base_noise_seed).lower() in ("none", "null", "") else int(args.base_noise_seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    config = OmegaConf.load(args.config)
    if args.set:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    config.dataloader.batch_size[args.split] = int(args.batch_size)
    # the config's own num_workers stay: anemoi's DataLoader passes prefetch_factor and persistent_workers=True,
    # which both need workers (first smoke on Jupiter, jobs 2398089 and 2398213, 18 Oct 2026)

    ckpt = _resolve_inference_path(args.base_checkpoint)
    print(f"[b2l-stats] loading base {ckpt}", file=sys.stderr, flush=True)
    base = torch.load(str(ckpt), map_location=device, weights_only=False).to(device).eval()
    for mod in base.modules():
        if hasattr(mod, "_cached_halo_info"):
            mod._cached_halo_info = None
        if hasattr(mod, "_cached_partition"):
            mod._cached_partition = None
    names = output_variable_names(base)
    target_ds = base.model._decoder_datasets[0]

    datamodule = get_datamodule(config, base.graph_data)
    loader = datamodule.train_dataloader() if args.split == "training" else datamodule.val_dataloader()
    dataset = loader.dataset
    # MultiDataset is an iterable dataset: no len(); and it slices each sample by the reader shard sizes that the
    # DDP strategy sets in training (set_comm_group_info). One process, one reader, one model group: the whole
    # grid of every dataset, as get_balanced_partition_sizes(grid, 1) gives it (Jupiter smoke 2398301).
    shard_sizes = {name: [int(base.graph_data[name]["data"].num_nodes)] for name in dataset.datasets}
    dataset.set_comm_group_info(0, 0, 0, 1, 0, 1, shard_sizes)
    n_pool = len(dataset.valid_date_indices) if hasattr(dataset, "valid_date_indices") else "?"
    print(f"[b2l-stats] {args.split} pool: {n_pool} samples, batch {args.batch_size}, device {device}, "
          f"sigma {args.base_sigma:g}, noise seed {seed}, grid {shard_sizes}", file=sys.stderr, flush=True)

    acc = ResidualStatsAccumulator(names)
    n_batches = 0
    for batch in loader:
        x_in_lres = batch["in_lres"].to(device)
        x_in_hres = batch["in_hres"].to(device)
        y = batch[target_ds].to(device)
        lead_hours = batch.get("lead_hours")
        if lead_hours is not None:
            lead_hours = lead_hours.to(device)
        with torch.no_grad():
            # exactly the training step's conditioning: upsample, add the ensemble axis, normalise
            x_up = base.model.residual["in_lres"](x_in_lres, grid_shard_sizes=None, model_comm_group=None)[:, :, None, :, :]
            x_up = base.pre_processors["in_lres"](x_up, in_place=False)
            x_hres = base.pre_processors["in_hres"](x_in_hres, in_place=False)
            base_full = AnemoiD2ModelEncProcDec.regression_base_prediction(
                base, x_up, x_hres, sigma=args.base_sigma, noise_seed=seed, lead_hours=lead_hours
            )
            full_data = base.model.data_indices[target_ds].data.output.full
            residual = y[..., full_data] - base_full
        acc.update(residual)
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"[b2l-stats] {n_batches} batches, {acc.n_samples} samples, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
        if args.max_batches and n_batches >= args.max_batches:
            break

    stats = acc.result()
    old_path = None
    try:
        old_path = config.system.input.get("residual_statistics", None)
    except Exception:  # noqa: BLE001
        old_path = None
    sidecar = {
        "what": "per-variable statistics of y - base(x) over the training pool (B2L, B1 note section 10)",
        "base_checkpoint": str(ckpt),
        "base_class": type(base.model).__name__,
        "base_sigma": args.base_sigma,
        "base_noise_seed": seed,
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "batches": n_batches,
        "samples": acc.n_samples,
        "points_per_variable": {n: int(c) for n, c in zip(names, acc.count)},
        "elapsed_s": round(time.time() - t0, 1),
        "stats": stats,
        "against_old": compare_with_old(str(old_path) if old_path else None, stats),
    }
    write_stats(Path(args.out), stats, sidecar)
    print(f"[b2l-stats] wrote {args.out} ({n_batches} batches, {acc.n_samples} samples, {time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    for name in names:
        row = (sidecar["against_old"] or {}).get("per_variable", {}).get(name, {}) if sidecar["against_old"] else {}
        old = row.get("old_stdev")
        print(f"  {name:>6s}  mean {stats['mean'][name]:+.4g}  stdev {stats['stdev'][name]:.4g}"
              + (f"  (old {old:.4g}, ratio {row['ratio_new_over_old']:.3f})" if old else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
