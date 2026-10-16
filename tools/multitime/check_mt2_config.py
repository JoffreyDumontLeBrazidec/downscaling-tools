#!/usr/bin/env python3
"""Check the rendered (Hydra-composed, resolved) mt2_e2_20k config against the agreed recipe.

Reads the rendered YAML written by
  anemoi-training train --config-path=<S>/configs --config-name=mt2_e2_20k --cfg job --resolve
and writes CHECK.md: one line per required value with the rendered value and PASS or FAIL, plus
the deviations that cannot be expressed on Vera Gahlen's branch. With timm importable it also
evaluates the learning-rate schedule the config builds, at a few steps.

Usage: python check_mt2_config.py --rendered <S>/configs/mt2_e2_20k.rendered.yaml --out <S>/configs/CHECK.md \
          [--jobs <S>/jobs/smoke_mt2.sbatch <S>/jobs/mt2_e2_20k.sbatch <S>/jobs/mt2_job.sh]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import yaml

E2V4 = "/ec/res4/scratch/ecm5702/o320_o1280_europe_e2_v4/data"
STORES = {"in_lres": f"{E2V4}/in_lres_o320_halo1p5.zarr", "out_hres": f"{E2V4}/out_hres_o1280.zarr",
          "in_hres": "/ec/res4/scratch/ecm5702/o320_o1280_europe_e2/data/in_hres_o1280_forcings.zarr"}
STEPS = [12, 24, 36, 48, 60, 72]
FAKE = {"training": {"end": 2021, "steps": STEPS}, "validation": {"start": 2022, "end": 2022, "steps": STEPS},
        "test": {"start": 2023, "end": 2023, "steps": STEPS}}
NL35 = {"default": 1, "q": 0.7, "t": 0.35, "u": 0.525, "v": 0.525, "w": 0.7, "z": 0.175, "sp": 1.5, "10u": 2.5,
        "10v": 2.5, "2d": 2, "tp": 2, "cp": 2, "msl": 2, "2t": 2, "tcw": 1, "skt": 0.5}


def get(cfg, path):
    d = cfg
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return "<missing>"
        d = d[k]
    return d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rendered", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", nargs="*", default=[])
    args = p.parse_args()
    text = Path(args.rendered).read_text()
    cfg = yaml.safe_load(text)
    rows = []

    def check(item, path_or_value, expected, ok=None, note=""):
        got = get(cfg, path_or_value) if isinstance(path_or_value, str) and not path_or_value.startswith("=") \
            else path_or_value[1:] if isinstance(path_or_value, str) else path_or_value
        passed = (got == expected) if ok is None else ok
        rows.append((item, path_or_value if isinstance(path_or_value, str) else "(derived)", expected, got,
                     "PASS" if passed else "FAIL", note))

    for split in ("training", "validation", "test"):
        for ds, store in STORES.items():
            base = f"dataloader.{split}.datasets.{ds}"
            check(f"{split} {ds} store", f"{base}.dataset_config.dataset", store)
            check(f"{split} {ds} fake_hindcasts", f"{base}.dataset_config.fake_hindcasts", FAKE[split])
            check(f"{split} {ds} hindcast_blocks", f"{base}.hindcast_blocks", {"steps_per_sequence": 6, "step": "12h"})
    check("input offsets", "task.input_offsets", ["0h", "12h"])
    check("output offsets", "task.output_offsets", ["0h", "12h"])
    check("prediction mode", "training.transport.prediction_mode", "residual")
    check("objective (EDM, loss weight edm_loss_weight hard-wired; no uncertainty weighting in the branch)",
          "training.transport.objective", "edm_diffusion")
    check("training noise log_normal(1.0, 2.5)", "model.model.transport.training_condition",
          {"distribution": "log_normal", "mean": 1.0, "std": 2.5})
    check("NL35 general_variable weights (surface and tp/cp unchanged)",
          "training.scalers.datasets.out_hres.general_variable.weights", NL35)
    check("loss scalers", "training.training_loss.datasets.out_hres.scalers",
          ["pressure_level", "general_variable", "nan_mask_weights", "node_weights"])
    check("pressure-level scaler slope", "training.scalers.datasets.out_hres.pressure_level.slope", 0.0001659751037)
    check("node weights attribute (ring mask, as do0)", "training.scalers.datasets.out_hres.node_weights.nodes_attribute_name",
          "area_weight_ring2")
    check("tp, cp as diagnostics", "data.datasets.out_hres.diagnostic", ["tp", "cp"])
    check("AdamW", "training.optimization.optimizer._target_", "torch.optim.AdamW")
    check("betas", "training.optimization.optimizer.betas", [0.9, 0.95])
    check("eps", "training.optimization.optimizer.eps", 1e-8)
    check("weight decay", "training.optimization.optimizer.weight_decay", 0.0)
    hw = get(cfg, "system.hardware")
    lr = get(cfg, "training.optimization.lr")
    eff = hw["num_nodes"] * hw["num_gpus_per_node"] * lr / hw["num_gpus_per_model"]
    check("effective peak lr = num_nodes*gpus_per_node*lr/gpus_per_model", f"={eff!r}", 1e-4,
          ok=math.isclose(eff, 1e-4, rel_tol=1e-9), note=f"lr {lr} x {hw['num_nodes'] * hw['num_gpus_per_node']}")
    check("scheduler", "training.optimization.lr_scheduler._target_", "timm.scheduler.CosineLRScheduler")
    check("warm-up 1000 steps", "training.optimization.lr_scheduler.warmup_t", 1000)
    check("lr_min = the effective rate (constant after warm-up)", "training.optimization.lr_scheduler.lr_min", 1e-4)
    check("scheduler steps per optimiser step", "training.optimization.pl_lr_scheduler.interval", "step")
    try:
        import torch
        from timm.scheduler import CosineLRScheduler

        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=eff)
        sc = get(cfg, "training.optimization.lr_scheduler")
        s = CosineLRScheduler(opt, t_initial=sc["t_initial"], lr_min=sc["lr_min"], warmup_t=sc["warmup_t"],
                              t_in_epochs=sc["t_in_epochs"])
        pts = {}
        for t in (0, 1, 500, 999, 1000, 1001, 5000, 10000, 19999, 20000):
            s.step_update(t)
            pts[t] = opt.param_groups[0]["lr"]
        flat = all(math.isclose(pts[t], 1e-4, rel_tol=1e-9) for t in (1000, 1001, 5000, 10000, 19999, 20000))
        ramp = math.isclose(pts[500], 0.5e-4, rel_tol=1e-6) and pts[0] == 0.0
        check("timm schedule evaluated (step: lr)", f"={pts}", "0 -> 1e-4 linear over 1000 steps, then 1e-4",
              ok=flat and ramp)
    except Exception as exc:  # noqa: BLE001
        rows.append(("timm schedule evaluated", "(derived)", "", f"not evaluated: {exc}", "SKIP", ""))
    wa = get(cfg, "training.weight_averaging")
    hl = math.log(0.5) / math.log(wa["decay"]) if isinstance(wa, dict) and "decay" in wa else None
    check("EMA (half-life from decay)", f"={hl}", 2000, ok=hl is not None and abs(hl - 2000) < 1e-6,
          note="EMAWeightAveraging, decay " + (str(wa.get("decay")) if isinstance(wa, dict) else "?"))
    rows.append(("EMA second half-life 10000", "(not expressible)", 10000, "absent", "FAIL",
                 "the branch builds ONE weight-averaging callback (diagnostics/callbacks/weight_averaging.py); two "
                 "Lightning WeightAveraging callbacks would both swap the saved state_dict"))
    check("gradient clip", "training.gradient_clip", {"val": 1.0, "algorithm": "norm"})
    check("precision", "training.precision", "bf16-mixed")
    check("nodes", "system.hardware.num_nodes", 2)
    check("GPUs per node", "system.hardware.num_gpus_per_node", 4)
    check("GPUs per model", "system.hardware.num_gpus_per_model", 1)
    check("batch per GPU (global 8)", "dataloader.batch_size.training", 1)
    check("limit_batches training", "dataloader.limit_batches.training", 1000)
    check("limit_batches validation", "dataloader.limit_batches.validation", 144)
    check("checkpoint every 1000 steps", "diagnostics.checkpoint.every_n_train_steps.save_frequency", 1000)
    check("keep all step checkpoints", "diagnostics.checkpoint.every_n_train_steps.num_models_saved", -1)
    check("max_steps", "training.max_steps", 20000)
    check("MLflow run name", "diagnostics.log.mlflow.run_name", "mt2_e2_20k")
    check("MLflow on, offline", "=" + str((get(cfg, "diagnostics.log.mlflow.enabled"), get(cfg, "diagnostics.log.mlflow.offline"))),
          "(True, True)")
    for k in ("in_lres", "in_hres", "out_hres", "hidden"):
        check(f"node trainable parameters {k}", f"model.node_trainable_parameters.{k}", 0)
    for k in ("data2hidden", "hidden2hidden", "hidden2data"):
        check(f"edge trainable parameters {k}", f"model.edge_trainable_parameters.{k}", 0)
    check("keep_batch_sharded", "model.keep_batch_sharded", True)
    check("num_channels", "model.num_channels", 512)
    check("processor layers", "model.processor.num_layers", 16)
    for part in ("model.processor", "model.encoders.0.mapper", "model.decoders.0.mapper"):
        check(f"{part} heads", f"{part}.num_heads", 4)
    check("residual reference", "model.residual_reference", {"out_hres": "in_lres"})
    rs = get(cfg, "data.datasets.out_hres.residual_statistics")
    check("residual statistics (.npz, exists)", "data.datasets.out_hres.residual_statistics", rs,
          ok=isinstance(rs, str) and rs.endswith(".npz") and os.path.exists(rs))
    tr = get(cfg, "data.datasets.in_lres.spatial_processor.file_path")
    check("projection matrix (E2 box, exists)", "data.datasets.in_lres.spatial_processor.file_path", tr,
          ok=isinstance(tr, str) and "europe_e2" in tr and os.path.exists(tr),
          note="rows sum to 1 within 1.1e-16, so row_normalize True changes nothing")
    check("CrossGridProjector", "data.datasets.in_lres.spatial_processor._target_",
          "anemoi.models.preprocessing.cross_grid_projector.CrossGridProjector")
    g = get(cfg, "system.input.graph")
    check("graph file (exists, loaded as is)", "system.input.graph", g, ok=isinstance(g, str) and os.path.exists(g))
    check("graph group does not rebuild", "graph.overwrite", False)
    check("sampler", "model.model.transport.inference_defaults.sampler",
          {"sampler": "heun_window", "window": 2, "stride": 1, "S_churn": 0.0})
    check("sampling schedule", "model.model.transport.inference_defaults.sampling_schedule",
          {"schedule_type": "karras", "sigma_max": 100000.0, "sigma_min": 0.0, "rho": 7.0, "num_steps": 30})
    check("config validation off (every key above is asserted instead)", "config_validation", False)

    if args.jobs:
        # The sbatch files are thin wrappers around the body; the checks apply to the job as a whole.
        t = "\n".join(Path(j).read_text() for j in args.jobs if Path(j).exists())
        names = " + ".join(Path(j).name for j in args.jobs)
        for item, pat in (("seed 756 (ANEMOI_BASE_SEED)", r"ANEMOI_BASE_SEED=756"),
                          ("TRITON_CACHE_DIR on node-local /tmp", r"TRITON_CACHE_DIR=/tmp/"),
                          ("--config-path on the command line", r"--config-path="),
                          ("core dumps off", r"ulimit -c 0"),
                          ("job layout: 2 nodes x 4 GPUs", r"--nodes=2[\s\S]*--gpus-per-node=4")):
            found = re.search(pat, t) is not None
            rows.append((f"job files ({names}): {item}", "(job files)", pat, "found" if found else "absent",
                         "PASS" if found else "FAIL", ""))
    n_fail = sum(1 for r in rows if r[4] == "FAIL")
    lines = ["# mt2_e2_20k: rendered config check", "",
             f"Rendered config: `{args.rendered}`. Generated by `tools/multitime/check_mt2_config.py`. "
             f"{len(rows)} lines, {n_fail} FAIL.", "",
             "| item | key | expected | rendered | result | note |", "|---|---|---|---|---|---|"]
    for item, key, exp, got, res, note in rows:
        lines.append(f"| {item} | `{key}` | `{exp}` | `{got}` | {res} | {note} |")
    lines += ["", "## Deviations that are not config keys", "",
              "- Projection matrix: the brief named `interpol_o320_to_o1280_grid-box-average.mat.npz`, which is the "
              "global 6,599,680 x 421,120 matrix and cannot multiply the E2 box stores (55,871 -> 814,810 points). "
              "The config uses the do0 recipe's E2-box matrix (linear, rows sum to 1 within 1.1e-16).",
              "- Graph: the do0 graph files (three per-dataset files) were converted into the one fused graph the "
              "branch loads (`tools/multitime/make_vera_graph.py`), node sets in_lres, in_hres, out_hres, hidden.",
              "- EMA: one EMA (half-life 2000 steps) instead of two (2000 and 10000), and no EMA warm-up "
              "(our recipe has `warmup: true, warmup_ratio: 0.2`); the branch's callback has neither.",
              "- Seed 756 is the environment variable ANEMOI_BASE_SEED set by the job files, not a config key."]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(rows)} lines, {n_fail} FAIL")
    for r in rows:
        if r[4] != "PASS":
            print("  ", r[4], r[0], "| expected", r[2], "| got", r[3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
