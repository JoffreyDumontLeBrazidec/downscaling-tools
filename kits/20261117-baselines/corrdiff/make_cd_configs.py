#!/usr/bin/env python3
"""Hydra configs of the CorrDiff baseline, written into the pinned PhysicsNeMo example (examples/weather/corrdiff/conf,
tag v2.2.2). Published defaults, the GEFS-HRRR pair (the example's closest published setup: a forecast input on a
~1,000 x 1,800 regional grid): regression U-Net on the full image, then the patched residual diffusion (EDM) on 448 x 448
patches, 4 patches per sample, model_size normal, total batch 256, lr 2e-4, amp-bf16. Changes, all of them:
  - dataset: our reader (cd_dataset.py::E2v4Box) and the E2-v4 export; validation = 64 rows of 2022
  - model: `regression` and `patched_diffusion` instead of the lead-time-aware variants (no lead-time label; GL512 has
    none either)
  - batch_size_per_gpu fixed (REG_BPG, DIFF_BPG) so the global batch 256 is reached by gradient accumulation on A100 64 GB
  - training_duration (samples): REG_SAMPLES and DIFF_SAMPLES; defaults = the published values (1,000,000 and
    10,000,000); a shorter run is allowed only to meet the 27 Nov date, and the paper states it
Input channels: GL512's own input variables (its in_lres list after `drop`), from GL512's config file.

    python3 make_cd_configs.py --gl512-config db_global_512_300k.yaml --data OUT.zarr --repo PHYSICSNEMO_CHECKOUT \
        --dataset-py cd_dataset.py --run-dir RUN [--reg-samples N] [--diff-samples N] [--reg-bpg 2] [--diff-bpg 4]
"""
import argparse
from pathlib import Path

import yaml

OUT10 = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gl512-config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dataset-py", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reg-samples", type=int, default=1_000_000)
    ap.add_argument("--diff-samples", type=int, default=10_000_000)
    ap.add_argument("--reg-bpg", type=int, default=2)
    ap.add_argument("--diff-bpg", type=int, default=4)
    a = ap.parse_args()
    import zarr
    gl = yaml.safe_load(open(a.gl512_config))
    inl = gl["dataloader"]["training"]["datasets"]["in_lres"]
    drop = list(inl.get("drop") or [])
    root = zarr.open_group(a.data, mode="r")
    allin = list(root.attrs["input_variables"])
    assert list(root.attrs["output_variables"]) == OUT10
    sel = inl.get("select")
    in_vars = [v for v in allin if v not in drop and (not sel or v in sel)]
    inv = list(root.attrs["invariant_variables"])
    data = str(Path(a.data).resolve())
    ds = {"type": f"{Path(a.dataset_py).resolve()}::E2v4Box", "data_path": data,
          "stats_path": str(Path(data).with_suffix(".stats.json")), "input_variables": in_vars,
          "output_variables": OUT10, "invariant_variables": inv, "split": "train"}
    val = {"split": "valid", "max_rows": 64}
    wandb = {"mode": "offline", "results_dir": f"{a.run_dir}/wandb", "watch_model": False}
    reg = {
        "hydra": {"job": {"chdir": False, "name": "e2v4_regression"}, "run": {"dir": f"{a.run_dir}/hydra/${{hydra:job.name}}"},
                  "searchpath": ["pkg://conf/base"]},
        "defaults": [{"dataset": "custom"}, {"model": "regression"}, {"model_size": "normal"}, {"training": "${model}"}, "_self_"],
        "dataset": ds, "validation": val,
        "training": {"hp": {"training_duration": a.reg_samples, "total_batch_size": 256, "batch_size_per_gpu": a.reg_bpg},
                     "perf": {"fp_optimizations": "amp-bf16", "dataloader_workers": 4},
                     "io": {"checkpoint_dir": f"{a.run_dir}/reg", "save_checkpoint_freq": 50_000, "validation_freq": 10_000,
                            "print_progress_freq": 2_560}},
        "wandb": wandb,
    }
    diff = {
        "hydra": {"job": {"chdir": False, "name": "e2v4_diffusion"}, "run": {"dir": f"{a.run_dir}/hydra/${{hydra:job.name}}"},
                  "searchpath": ["pkg://conf/base"]},
        "defaults": [{"dataset": "custom"}, {"model": "patched_diffusion"}, {"model_size": "normal"}, {"training": "${model}"}, "_self_"],
        "dataset": ds, "validation": val,
        "training": {"hp": {"training_duration": a.diff_samples, "total_batch_size": 256, "batch_size_per_gpu": a.diff_bpg,
                            "patch_shape_x": 448, "patch_shape_y": 448, "patch_num": 4},
                     "perf": {"fp_optimizations": "amp-bf16", "dataloader_workers": 4},
                     "io": {"checkpoint_dir": f"{a.run_dir}/diff", "regression_checkpoint_path": "${oc.env:CD_REG_CKPT}",
                            "save_checkpoint_freq": 500_000, "validation_freq": 50_000, "print_progress_freq": 2_560}},
        "wandb": wandb,
    }
    conf = Path(a.repo) / "examples" / "weather" / "corrdiff" / "conf"
    assert (conf / "base" / "model" / "patched_diffusion.yaml").exists(), conf
    for name, c in (("config_training_e2v4_regression.yaml", reg), ("config_training_e2v4_diffusion.yaml", diff)):
        (conf / name).write_text(yaml.safe_dump(c, sort_keys=False, width=4096))
        print("wrote", conf / name)
    print(f"input variables ({len(in_vars)}): {in_vars}; dropped by GL512: {drop}; invariants: {inv}")


if __name__ == "__main__":
    main()
