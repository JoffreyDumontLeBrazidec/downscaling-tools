"""``eval.cli predict``: generate predictions for a lane and a checkpoint.

Manual mode calls ``eval.predict.main`` in a subprocess (wrapped in ``srun`` for
model parallelism); prepml mode hands over to ``eval.predict.prepml``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.cli._bundles import _resolve_predict_input_root, _verify_predict_input_bundles
from eval.cli._common import (
    LOG, Command, add_common_args, add_lane_override_args, add_prepare_args, add_prepml_args,
)

SUMMARY = "Generate predictions only, without scoring them."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("predict", help=SUMMARY, description=SUMMARY)
    add_common_args(p)
    p.add_argument("--checkpoint", required=True, help="Path to the model checkpoint.")
    add_lane_override_args(p)
    add_prepare_args(p)
    add_prepml_args(p)
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <scratch>/eval/<lane>/run_<TS>). "
             "Predictions go to <output-dir>/predictions.",
    )
    return p


def cmd_predict(args: argparse.Namespace, lane_config: dict, host_config: dict, output_dir: Path) -> None:
    """Run predictions via subprocess call to eval.predict.main."""
    mode = getattr(args, "mode", "manual")
    if mode == "prepml":
        from eval.predict.prepml import prepml_predict
        input_root = _resolve_predict_input_root(
            args, lane_config, host_config, output_dir,
            prepare_bundles=True,
            allow_host_fallback=False,
        )
        if not input_root:
            raise SystemExit(
                "PrepML predict requires truth-aware bundles for prediction assembly. "
                "Pass --source-grib-root to build them, --bundle-dir to reuse them, "
                "or set predict.input_root in the lane config."
            )
        if input_root:
            lane_config.setdefault("predict", {})["input_root"] = input_root
        prepml_predict(
            checkpoint=args.checkpoint,
            lane_config=lane_config,
            host_config=host_config,
            output_dir=output_dir,
            expver=getattr(args, "expver", None),
            runner_override=getattr(args, "prepml_runner", None),
            lane=getattr(args, "lane", ""),
        )
        return

    predict_cfg = lane_config["predict"]
    checkpoint = args.checkpoint

    # Auto-resolve inference-* companion to base checkpoint for manual mode.
    # PrepML mode uses the inference checkpoint directly (handled above).
    ckpt_path = Path(checkpoint)
    if ckpt_path.name.startswith("inference-") and ckpt_path.name.endswith(".ckpt"):
        base_name = ckpt_path.name.replace("inference-", "", 1)
        base_path = ckpt_path.parent / base_name
        if base_path.exists():
            LOG.warning(
                "Auto-resolved inference companion to base checkpoint: %s -> %s",
                ckpt_path.name, base_name,
            )
            checkpoint = str(base_path)
        else:
            raise FileNotFoundError(
                f"Inference companion checkpoint passed but base checkpoint not found: "
                f"{base_path}. Manual predict requires the base (non-inference) checkpoint."
            )

    input_root = _resolve_predict_input_root(
        args, lane_config, host_config, output_dir, prepare_bundles=True,
    )
    _verify_predict_input_bundles(lane_config, input_root)

    members_str = ",".join(str(m) for m in predict_cfg["members"])
    steps_str = ",".join(str(s) for s in predict_cfg["steps"])
    dates_str = ",".join(predict_cfg["dates"])
    bundle_pairs = predict_cfg.get("bundle_pairs", "")
    if isinstance(bundle_pairs, list):
        bundle_pairs = ",".join(
            f"{item.get('date')}:{item.get('step')}" if isinstance(item, dict) else str(item)
            for item in bundle_pairs
        )

    predictions_dir = output_dir / "predictions"

    cmd = [
        sys.executable, "-m", "eval.predict.main",
        "--name-ckpt", str(checkpoint),
        "--out-dir", str(predictions_dir),
        "--members", members_str,
        "--steps", steps_str,
        "--dates", dates_str,
        "--input-root", input_root,
        "--allow-existing-out-dir",
    ]
    if bundle_pairs:
        cmd += ["--bundle-pairs", str(bundle_pairs)]

    num_gpus_per_model = predict_cfg.get("num_gpus_per_model")
    if num_gpus_per_model is not None:
        cmd += ["--num-gpus-per-model", str(int(num_gpus_per_model))]

    # Pass sampler config from lane YAML if present, overriding predict.main defaults
    sampler_cfg = predict_cfg.get("sampler")
    if sampler_cfg:
        cmd += ["--extra-args-json", json.dumps(sampler_cfg)]

    local_scope_cfg = predict_cfg.get("local_scope")
    if local_scope_cfg:
        cmd += ["--local-scope-json", json.dumps(local_scope_cfg)]

    # Wrap in srun for multi-GPU model parallelism. Requires an outer sbatch
    # allocation; falls back to single-process when not in SLURM.
    # No --gpus-per-task: each rank needs all node GPUs visible so the model
    # loader can do torch.cuda.set_device(cuda:<local_rank>) without binding.
    #
    # C8(ii): cmd_predict is the single owner of the predict srun. If we are
    # already inside an srun step (SLURM_STEP_ID set) — e.g. the rendered sbatch
    # itself launched `srun python -m eval.cli predict` — do NOT add another srun,
    # which would nest `srun srun` and break. (The renderer no longer wraps
    # predict, so in the normal pipeline this guard simply confirms ownership.)
    already_in_srun = os.environ.get("SLURM_STEP_ID") is not None
    if (
        num_gpus_per_model
        and int(num_gpus_per_model) > 1
        and os.environ.get("SLURM_JOB_ID")
        and not already_in_srun
    ):
        n = str(int(num_gpus_per_model))
        cmd = ["srun", "--ntasks", n, "--ntasks-per-node", n] + cmd

    LOG.info("Running predictions: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    LOG.info("Predictions written to %s", predictions_dir)


def run(args: argparse.Namespace, session) -> None:
    cmd_predict(args, session.lane_config, session.host_config, session.output_dir)


COMMANDS = (Command("predict", "pipeline", SUMMARY, register, run, needs_lane=True),)
