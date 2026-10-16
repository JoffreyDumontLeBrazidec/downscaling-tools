#!/usr/bin/env python3
"""Materialise the donor 12dcefea step 400,000 with the do0 recipe's 68 -> 70 output extension.

Stage (a) of the mt2_e2_20k warm start. The do0 run (Atos sandbox 20261007-pl25-ac, config
do0_e2_50k_ac.yaml) starts from the donor weights only and appends tp and cp as direct outputs
at step 0: its job body passes system.input.warm_start=<donor>, training.transfer_learning=True
and training.load_weights_only=True, and the config sets transfer_learning_extend_inputs and
transfer_learning_extend_outputs true. AnemoiTrainer.model then calls
anemoi.training.utils.checkpoint.transfer_learning_loading with both extensions: the encoder's
input embedding gets two zero columns (the noised tp, cp channels, placed by the input column map)
and the decoder's output layer two zero rows (tp, cp outputs).

This script does exactly that, in the do0 sandbox's own venv and code (activate.sh of the sandbox,
nothing modified there): it composes the do0 config with the job body's overrides, builds
AnemoiTrainer(config).model on the CPU, and saves
  <out>.ckpt            {"state_dict", "hyper_parameters": {"data_indices"}, "provenance"}
  <out>.interface.pt    the pickled AnemoiModelInterface (for the forward check)
and checks every donor tensor against the materialised one (equal, or extended with the donor
block exact and the new entries zero).

Usage (login node, detached; the do0 venv):
  source /home/ecm5702/scratch/sandbox/20261007-pl25-ac/activate.sh
  OUTPUT=/tmp/... python materialise_do0_donor.py --out <S>/checkpoints/warm/stage_a_do0_donor70
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

DO0 = Path("/ec/res4/scratch/ecm5702/sandbox/20261007-pl25-ac")
DONOR = "/home/ecm5702/perm/checkpoints/o320_o1280/12dcefeafa92457aab286d4931ffe63f/anemoi-by_epoch-epoch_172-step_400000.ckpt"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output stem (no extension)")
    p.add_argument("--config-name", default="do0_e2_50k_ac")
    args = p.parse_args()
    out = Path(args.out)
    for suffix in (".ckpt", ".interface.pt"):
        if out.with_name(out.name + suffix).exists():
            log(f"refusing: {out.name + suffix} exists (move it aside first)")
            return 2

    os.environ.setdefault("ANEMOI_BASE_SEED", "756")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("SLURM_GPUS_PER_NODE", "4")
    os.environ.setdefault("ANEMOI_GRAPHS_FORCE_CPU", "1")
    import torch
    import anemoi.training
    import anemoi.models

    log(f"anemoi.training {anemoi.training.__file__}; anemoi.models {anemoi.models.__file__}; torch {torch.__version__}")
    from hydra import compose, initialize_config_dir

    from anemoi.training.train.train import AnemoiTrainer

    overrides = [
        "config_validation=False",
        f"system.input.warm_start={DONOR}",
        "training.transfer_learning=True",
        "training.load_weights_only=True",
        "training.run_id=null",
        "training.fork_run_id=null",
        "diagnostics.log.mlflow.enabled=False",
        "training.max_steps=50000",
        "training.max_epochs=null",
    ]
    with initialize_config_dir(config_dir=str(DO0 / "configs"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    flags = {k: cfg.training.get(k) for k in ("transfer_learning", "load_weights_only", "transfer_learning_extend_inputs",
                                               "transfer_learning_extend_outputs")}
    log(f"composed {args.config_name} with the job body's donor overrides: {flags}")
    assert all(flags.values()), flags

    trainer = AnemoiTrainer(cfg)
    model = trainer.model  # transfer_learning_loading(..., extend_input_columns=True, extend_output_rows=True)
    log(f"built and loaded {type(model).__name__}")

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    donor = torch.load(DONOR, map_location="cpu", weights_only=False, mmap=True)
    dsd = donor["state_dict"]
    rows = {"equal": 0, "extended": [], "differs": [], "not_in_donor": [], "not_in_model": []}
    for k, v in sd.items():
        if k not in dsd:
            rows["not_in_donor"].append(k)
            continue
        c = dsd[k]
        if tuple(c.shape) == tuple(v.shape):
            if torch.equal(c.to(v.dtype), v):
                rows["equal"] += 1
            else:
                rows["differs"].append((k, float((c.double() - v.double()).abs().max())))
            continue
        rows["extended"].append((k, tuple(c.shape), tuple(v.shape)))
    rows["not_in_model"] = [k for k in dsd if k not in sd]
    # The grown tensors in detail: where the donor block sits and what the new entries hold.
    detail = []
    gmasks = getattr(model, "_grown_entry_masks", {}) or {}
    for k, cs, ms in rows["extended"]:
        v, c = sd[k], dsd[k]
        mask = gmasks.get(k)
        if mask is not None:
            old = v[~mask]
            new = v[mask]
            ok = old.numel() == c.numel() and torch.equal(old, c.reshape(-1).to(v.dtype))  # row-major order is kept
            detail.append({"key": k, "donor": cs, "model": ms, "donor_values_kept": bool(ok),
                           "new_entries": int(mask.sum()), "new_entries_max_abs": float(new.abs().max()) if new.numel() else 0.0})
        else:
            detail.append({"key": k, "donor": cs, "model": ms, "mask": "none recorded"})
    net_roots = ("encoder", "decoder", "processor", "noise_cond_mlp", "noise_embedder")
    n_net = sum(1 for k in sd if k.startswith("model.model.") and k.split(".")[2] in net_roots)
    log(f"materialised vs donor: {rows['equal']} tensors equal, {len(rows['extended'])} extended, "
        f"{len(rows['differs'])} differ, {len(rows['not_in_donor'])} not in the donor, "
        f"{len(rows['not_in_model'])} donor keys not in the model; {n_net} network tensors")
    for d in detail:
        log(f"  grown: {d}")
    for k, m in rows["differs"][:40]:
        log(f"  differs (same shape): {k} max abs {m:.3e}")
    log(f"  not in donor: {rows['not_in_donor'][:40]}")
    log(f"  donor keys not in model: {rows['not_in_model'][:40]}")

    di = model.data_indices
    for name, ix in di.items():
        log(f"data indices {name}: model.input {len(ix.model.input)} {list(ix.model.input.name_to_index)}")
        log(f"data indices {name}: model.output {len(ix.model.output)} {list(ix.model.output.name_to_index)}")

    prov = {"donor": DONOR, "recipe": str(DO0 / "configs" / (args.config_name + ".yaml")), "overrides": overrides,
            "code": anemoi.training.__file__, "built": datetime.datetime.utcnow().isoformat() + "Z",
            "check": {k: (v if k == "equal" else len(v)) for k, v in rows.items()}, "grown": detail}
    torch.save({"state_dict": sd, "hyper_parameters": {"data_indices": di}, "provenance": prov},
               out.with_name(out.name + ".ckpt"))
    torch.save(model.model, out.with_name(out.name + ".interface.pt"))
    Path(out.with_name(out.name + ".check.json")).write_text(json.dumps(prov, indent=1, default=str))
    log(f"wrote {out}.ckpt, {out}.interface.pt, {out}.check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
