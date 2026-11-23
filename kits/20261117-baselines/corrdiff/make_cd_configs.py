#!/usr/bin/env python3
"""Hydra configs of the CorrDiff baseline for the pinned PhysicsNeMo example (examples/weather/corrdiff, tag v2.2.2),
written into its conf folder (default) or into --out-conf-dir (then train.py needs --config-dir=<that folder>; the
`pkg://conf/base` search path still finds the example's own groups when train.py runs from the example folder).
Reference recipe: the GEFS-HRRR pair (the example's closest published setup: a forecast input on a ~1,000 x 1,800
regional grid): regression U-Net on the full image, then the patched residual diffusion (EDM) on 448 x 448 patches,
4 patches per sample, model_size normal, lr 2e-4, amp-bf16. Departures from NVIDIA's GEFS-HRRR files, all of them:
  - dataset: our reader (cd_dataset.py::E2v4Box) and the E2-v4 export; validation = 64 rows of 2022 (the first 64,
    29 Jun - 3 Aug; --val-spread: 64 rows spread over the whole year)
  - model: `regression` and `patched_diffusion` instead of NVIDIA's `lt_aware_ce_regression` and
    `lt_aware_patched_diffusion` (no lead-time label, no precipitation-probability head; GL512 has none either)
  - training group: the configs keep NVIDIA's `training: ${model}`, so swapping the model group also swapped the
    training group: training/patched_diffusion.yaml instead of training/lt_aware_patched_diffusion.yaml, and
    training/regression.yaml instead of training/lt_aware_ce_regression.yaml. That swap is how the diffusion lost
    NVIDIA's lr_rampup 1,000,000 (patched_diffusion.yaml has none, base_all gives 0; with lr_rampup 0 and lr_decay 0.7
    train.py compounds the decay at every update after lr_decay_rate samples and the rate collapses to zero, see the
    assert below), and how the total batch went from NVIDIA's literal 1 (total_batch_size 1, batch_size_per_gpu 1 in
    both lt_aware groups) to base_all's 256; their io frequencies (print 1, checkpoint 5, validation 1 sample,
    validation_steps 1000) and checkpoint levels (2 regression, 1 diffusion) went the same way.
  - schedule: every stage now writes total_batch_size, batch_size_per_gpu, lr, lr_rampup, lr_decay, lr_decay_rate,
    training_duration, save_checkpoint_freq, validation_freq explicitly (values not given = the example's base files,
    read from --repo); diffusion P_mean / P_std only when given (train.py passes them to the residual loss; NVIDIA's
    files never set them: loss defaults P_mean 0, P_std 1.2). The generator refuses lr_decay < 1 with lr_rampup <= 0,
    and with lr_decay < 1 it needs lr_rampup and lr_decay_rate given explicitly (or --scale-schedule-to-batch).
    --scale-schedule-to-batch (diffusion): ramp-up and decay interval = the published 1,000,000 / 500,000 samples
    (read as values for batch 256) x total_batch_size / 256, the same schedule counted in optimizer updates
  - batch_size_per_gpu fixed (REG_BPG, DIFF_BPG) so the global batch is reached by gradient accumulation
  - activation checkpointing (perf.songunet_checkpoint_level, PhysicsNeMo's own option; same model and loss): the
    regression U-Net on the full 864 x 1728 image runs out of memory on A100 64 GB at level 0 (batch 1 and 2) and at
    level 2 / batch 2; level 3 / batch 1 fits (24.6 GB steady, 0.57 s per sample per node; smokes 18 Nov, Leonardo)
  - diffusion: hp.max_patch_per_gpu 4 (NVIDIA's own option, used by their hrrr_patched_diffusion_opt config: the 4 patches
    of each sample run in 4 sub-iterations, same model and loss) + checkpoint level 1: 60.7 GB peak, 0.38 s per sample
    per node. Level 0 or 1 without it, and max_patch 8 / level 2, run out of memory on A100 64 GB
  - training_duration (samples): REG_SAMPLES and DIFF_SAMPLES; defaults = the published values (1,000,000 and
    10,000,000); a shorter run is allowed only to meet the 27 Nov date, and the paper states it
  - io: checkpoints every 50,000 (regression) / 500,000 (diffusion) samples, validation every 10,000 / 50,000, progress
    every 2,560; wandb offline; hydra run folder under RUN
  - --time-inputs (opt-in): dataset.time_inputs, 5 per-row time channels appended after the 46 (cd_dataset.py); a
    regression checkpoint trained without them has 46 input channels and does not load into a config with them
Input channels: GL512's own input variables (its in_lres list after `drop`), from GL512's config file.
--name-suffix S: files config_training_e2v4S_{regression,diffusion}.yaml, run folders RUN/reg_S, RUN/diff_S (the
unsuffixed files and RUN/reg, RUN/diff are left alone).

    python3 make_cd_configs.py --gl512-config db_global_512_300k.yaml --data OUT.zarr --repo PHYSICSNEMO_CHECKOUT \
        --dataset-py cd_dataset.py --run-dir RUN [--out-conf-dir DIR] [--name-suffix S] [--time-inputs] [--val-spread]
        [--reg-samples N] [--diff-samples N] [--reg-bpg 1] [--diff-bpg 4] [--reg-total-batch 256] [--diff-total-batch 256]
        [--{reg,diff}-lr X] [--{reg,diff}-lr-rampup N] [--{reg,diff}-lr-decay X] [--{reg,diff}-lr-decay-rate N]
        [--{reg,diff}-save-freq N] [--{reg,diff}-val-freq N] [--diff-p-mean X] [--diff-p-std X] [--scale-schedule-to-batch]
        [--reg-ckpt-level 3] [--diff-ckpt-level 1] [--diff-max-patch 4]
"""
import argparse
from pathlib import Path

import yaml

OUT10 = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"]
PUBLISHED_BATCH, PUBLISHED_RAMPUP, PUBLISHED_DECAY_RATE = 256, 1_000_000, 500_000
SCHEDULE = ("total_batch_size", "batch_size_per_gpu", "lr", "lr_rampup", "lr_decay", "lr_decay_rate", "training_duration")


def base_hp(conf: Path, group: str) -> dict:
    """training.hp of conf/base/training/<group>.yaml over base_all.yaml (the group = the model name)."""
    hp = dict(yaml.safe_load(open(conf / "base" / "training" / "base_all.yaml"))["hp"])
    hp.update((yaml.safe_load(open(conf / "base" / "training" / f"{group}.yaml")) or {}).get("hp") or {})
    return hp


def schedule(stage: str, group: str, conf: Path, a, scale: bool) -> dict:
    """Resolved schedule of one stage (arguments over the example's base files), checked and printed."""
    hp = base_hp(conf, group)
    p = stage[:3] if stage == "regression" else "diff"
    given = {"total_batch_size": getattr(a, f"{p}_total_batch"), "batch_size_per_gpu": getattr(a, f"{p}_bpg"),
             "lr": getattr(a, f"{p}_lr"), "lr_rampup": getattr(a, f"{p}_lr_rampup"), "lr_decay": getattr(a, f"{p}_lr_decay"),
             "lr_decay_rate": getattr(a, f"{p}_lr_decay_rate"), "training_duration": getattr(a, f"{p}_samples")}
    if scale:
        assert given["lr_rampup"] is None and given["lr_decay_rate"] is None, \
            f"{stage}: --scale-schedule-to-batch sets lr_rampup and lr_decay_rate; do not give them as well"
        b = given["total_batch_size"]
        given["lr_rampup"] = PUBLISHED_RAMPUP * b // PUBLISHED_BATCH
        given["lr_decay_rate"] = PUBLISHED_DECAY_RATE * b // PUBLISHED_BATCH
        assert given["lr_rampup"] * PUBLISHED_BATCH == PUBLISHED_RAMPUP * b and \
            given["lr_decay_rate"] * PUBLISHED_BATCH == PUBLISHED_DECAY_RATE * b, f"total_batch_size {b}: not exact in samples"
    s = {k: (given[k] if given[k] is not None else hp[k]) for k in SCHEDULE}
    assert float(s["lr_decay"]) >= 1 or (given["lr_rampup"] is not None and given["lr_decay_rate"] is not None), (
        f"{stage}: lr_decay {float(s['lr_decay']):g} < 1 needs the sample-based schedule given explicitly: "
        f"--{'reg' if stage == 'regression' else 'diff'}-lr-rampup N and --{'reg' if stage == 'regression' else 'diff'}"
        "-lr-decay-rate N (samples), or --scale-schedule-to-batch")
    s["lr"], s["lr_decay"] = float(s["lr"]), float(s["lr_decay"])
    s["lr_rampup"], s["lr_decay_rate"] = int(float(s["lr_rampup"])), int(float(s["lr_decay_rate"]))
    assert not (s["lr_decay"] < 1 and s["lr_rampup"] <= 0), (
        f"{stage}: lr_decay {s['lr_decay']} < 1 with lr_rampup {s['lr_rampup']} <= 0 is refused: PhysicsNeMo train.py "
        "(v2.2.2, lines 651-664) resets the rate at each update only when lr_rampup > 0, so the decay compounds at every "
        "update after lr_decay_rate samples and the rate collapses to zero. Give --{0}-lr-rampup N (> 0), or "
        "--{0}-lr-decay 1, or --scale-schedule-to-batch".format(p))
    b = s["total_batch_size"]
    assert b % s["batch_size_per_gpu"] == 0, f"{stage}: total_batch_size {b} not a multiple of batch_size_per_gpu"
    gpus = [n for n in (1, 2, 4, 8, 12, 16, 32, 64) if b % (s["batch_size_per_gpu"] * n) == 0]
    print(f"{stage}: total_batch_size {b} batch_size_per_gpu {s['batch_size_per_gpu']} lr {s['lr']:g} | "
          f"lr_rampup {s['lr_rampup']:,} samples = {s['lr_rampup'] / b:,.0f} updates | lr_decay {s['lr_decay']:g} every "
          f"{s['lr_decay_rate']:,} samples = {s['lr_decay_rate'] / b:,.0f} updates | training_duration "
          f"{s['training_duration']:,} samples = {s['training_duration'] / b:,.0f} updates | GPU counts without a "
          f"batch mismatch: {gpus}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gl512-config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dataset-py", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-conf-dir", default=None, help="default: REPO/examples/weather/corrdiff/conf")
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--time-inputs", action="store_true", help="dataset.time_inputs: 5 per-row time channels")
    ap.add_argument("--val-spread", action="store_true", help="validation rows spread over the whole 2022 split")
    ap.add_argument("--val-rows", type=int, default=64)
    ap.add_argument("--reg-samples", type=int, default=1_000_000)
    ap.add_argument("--diff-samples", type=int, default=10_000_000)
    ap.add_argument("--reg-total-batch", type=int, default=256)
    ap.add_argument("--diff-total-batch", type=int, default=256)
    ap.add_argument("--reg-bpg", type=int, default=1)
    ap.add_argument("--diff-bpg", type=int, default=4)
    for p in ("reg", "diff"):
        ap.add_argument(f"--{p}-lr", type=float, default=None)
        ap.add_argument(f"--{p}-lr-rampup", type=int, default=None, help="samples")
        ap.add_argument(f"--{p}-lr-decay", type=float, default=None)
        ap.add_argument(f"--{p}-lr-decay-rate", type=int, default=None, help="samples")
    ap.add_argument("--reg-save-freq", type=int, default=50_000)
    ap.add_argument("--diff-save-freq", type=int, default=500_000)
    ap.add_argument("--reg-val-freq", type=int, default=10_000)
    ap.add_argument("--diff-val-freq", type=int, default=50_000)
    ap.add_argument("--diff-p-mean", type=float, default=None)
    ap.add_argument("--diff-p-std", type=float, default=None)
    ap.add_argument("--scale-schedule-to-batch", action="store_true",
                    help="diffusion: lr_rampup / lr_decay_rate = 1,000,000 / 500,000 samples x total_batch_size / 256")
    ap.add_argument("--reg-ckpt-level", type=int, default=3)
    ap.add_argument("--diff-ckpt-level", type=int, default=1)
    ap.add_argument("--diff-max-patch", type=int, default=4)
    a = ap.parse_args()
    import zarr
    conf = Path(a.repo) / "examples" / "weather" / "corrdiff" / "conf"
    assert (conf / "base" / "model" / "patched_diffusion.yaml").exists(), conf
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
    if a.time_inputs:
        ds["time_inputs"] = True
    val = {"split": "valid", "max_rows": a.val_rows}
    if a.val_spread:
        val["spread"] = True
    sfx = a.name_suffix
    reg_dir, diff_dir = (f"{a.run_dir}/reg_{sfx}", f"{a.run_dir}/diff_{sfx}") if sfx else (f"{a.run_dir}/reg", f"{a.run_dir}/diff")
    wandb = {"mode": "offline", "results_dir": f"{a.run_dir}/wandb", "watch_model": False}
    rs = schedule("regression", "regression", conf, a, scale=False)
    dsch = schedule("diffusion", "patched_diffusion", conf, a, scale=a.scale_schedule_to_batch)
    dhp = dict(dsch, patch_shape_x=448, patch_shape_y=448, patch_num=4, max_patch_per_gpu=a.diff_max_patch)
    if a.diff_p_mean is not None:
        dhp["P_mean"] = a.diff_p_mean
    if a.diff_p_std is not None:
        dhp["P_std"] = a.diff_p_std
    reg = {
        "hydra": {"job": {"chdir": False, "name": f"e2v4{sfx}_regression"},
                  "run": {"dir": f"{a.run_dir}/hydra/${{hydra:job.name}}"}, "searchpath": ["pkg://conf/base"]},
        "defaults": [{"dataset": "custom"}, {"model": "regression"}, {"model_size": "normal"}, {"training": "${model}"}, "_self_"],
        "dataset": ds, "validation": val,
        "training": {"hp": rs,
                     "perf": {"fp_optimizations": "amp-bf16", "dataloader_workers": 4,
                              "songunet_checkpoint_level": a.reg_ckpt_level},
                     "io": {"checkpoint_dir": reg_dir, "save_checkpoint_freq": a.reg_save_freq,
                            "validation_freq": a.reg_val_freq, "print_progress_freq": 2_560}},
        "wandb": wandb,
    }
    diff = {
        "hydra": {"job": {"chdir": False, "name": f"e2v4{sfx}_diffusion"},
                  "run": {"dir": f"{a.run_dir}/hydra/${{hydra:job.name}}"}, "searchpath": ["pkg://conf/base"]},
        "defaults": [{"dataset": "custom"}, {"model": "patched_diffusion"}, {"model_size": "normal"}, {"training": "${model}"}, "_self_"],
        "dataset": ds, "validation": val,
        "training": {"hp": dhp,
                     "perf": {"fp_optimizations": "amp-bf16", "dataloader_workers": 4,
                              "songunet_checkpoint_level": a.diff_ckpt_level},
                     "io": {"checkpoint_dir": diff_dir, "regression_checkpoint_path": "${oc.env:CD_REG_CKPT}",
                            "save_checkpoint_freq": a.diff_save_freq, "validation_freq": a.diff_val_freq,
                            "print_progress_freq": 2_560}},
        "wandb": wandb,
    }
    out = Path(a.out_conf_dir) if a.out_conf_dir else conf
    out.mkdir(parents=True, exist_ok=True)
    for name, c in ((f"config_training_e2v4{sfx}_regression.yaml", reg), (f"config_training_e2v4{sfx}_diffusion.yaml", diff)):
        (out / name).write_text(yaml.safe_dump(c, sort_keys=False, width=4096))
        print("wrote", out / name)
    print(f"input variables ({len(in_vars)}): {in_vars}; dropped by GL512: {drop}; invariants: {inv}; "
          f"time inputs: {a.time_inputs}; validation: {val}")


if __name__ == "__main__":
    main()
