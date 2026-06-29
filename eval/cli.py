"""Unified CLI entry point for the evaluation framework.

Canonical invocation: python -m eval.cli <subcommand>

Subcommands:
    run         Full pipeline: predict + evaluate + scoreboard
    predict     Generate predictions only (subprocess call to eval.predict.main)
    evaluate    Run evaluators on existing predictions
    scoreboard  Generate scoreboard from existing evaluation results
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.config.loader import (
    default_host_for_stage,
    load_host,
    load_lane,
    validate_lane_host_compatible,
)
from eval.paths import resolve_eval_root
from eval.evaluators.tc.comparison_contract import require_lane_analysis_reference
from eval import lean_layout

LOG = logging.getLogger(__name__)

ALL_EVALUATORS = [
    "tc", "spectra", "surface", "region_plot",
    "sigma", "sigma_loss", "mechanistic", "intermediate",
    "spectra_ecmwf", "mlflow",
    "precip_dist", "precip_events",
    "interp", "probabilistic",
]

DEFAULT_HOST = "atos_ac"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    parser.add_argument(
        "--lane", required=True,
        help="Lane name (must match a YAML file in eval/config/lanes/).",
    )
    parser.add_argument(
        "--host", default=None,
        help=f"Host config name (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print resolved config as JSON and exit without running.",
    )


def _add_evaluator_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add --only and --include-diagnostics for evaluator selection."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", default=None,
        help="Comma-separated list of evaluators to run (overrides lane YAML groups).",
    )
    group.add_argument(
        "--include-diagnostics", action="store_true", default=False,
        help="Run default + diagnostics evaluator groups from lane YAML.",
    )


def _add_lane_override_args(parser: argparse.ArgumentParser) -> None:
    """Add lane-overridable args.  All use default=None for precedence detection."""
    parser.add_argument(
        "--members", default=None,
        help="Comma-separated member indices (e.g. 1,2,3). Overrides lane YAML predict.members.",
    )
    parser.add_argument(
        "--steps", default=None,
        help="Comma-separated forecast steps (e.g. 24,48). Overrides lane YAML predict.steps.",
    )
    parser.add_argument(
        "--dates", default=None,
        help="Comma-separated dates YYYYMMDD (e.g. 20230826,20230827). Overrides lane YAML predict.dates.",
    )
    parser.add_argument(
        "--weather-states", default=None,
        help=(
            "Comma-separated weather_state names (e.g. 10u,2t,z_500). Overrides lane YAML "
            "predict.weather_states. Honored by --mode prepml's resolver as the highest-priority "
            "source; manual mode still resolves from checkpoint output via surface-plus-core-pl."
        ),
    )


def _add_prepare_args(parser: argparse.ArgumentParser) -> None:
    """Add truth-aware bundle-building args."""
    parser.add_argument(
        "--source-grib-root", default=None,
        help="Root directory of source GRIB files for truth-aware bundle building.",
    )
    parser.add_argument(
        "--bundle-dir", default=None,
        help=(
            "Bundle directory. With --source-grib-root: output for newly built bundles. "
            "Without --source-grib-root: existing bundle directory used as input_root "
            "(rebuild is skipped). Default: <output-dir>/bundles."
        ),
    )
    parser.add_argument(
        "--num-gpus-per-model", type=int, default=None,
        help="GPUs per model replica. Overrides lane YAML predict.num_gpus_per_model.",
    )
    parser.add_argument(
        "--num-chunks", type=int, default=None,
        help=(
            "Override ANEMOI_INFERENCE_NUM_CHUNKS and its _PROCESSOR/_MAPPER "
            "variants in the inference env. Chunks attention in block/mapper "
            "layers to fit on fewer GPUs. Falls back to lane YAML predict.env."
        ),
    )


def _add_prepml_args(parser: argparse.ArgumentParser) -> None:
    """Add PrepML-specific args to run and predict subcommands."""
    parser.add_argument(
        "--mode", choices=["manual", "prepml"], default="manual",
        help="Prediction backend: manual (bundle-based) or prepml (MARS/FDB). Default: manual.",
    )
    parser.add_argument(
        "--expver", default=None,
        help="PrepML expver. If omitted in prepml mode, uses debug expver from lane config.",
    )
    parser.add_argument(
        "--prepml-runner", default=None,
        help="Override PrepML runner/venv path from lane config.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.cli",
        description="Unified CLI for the evaluation framework.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Full pipeline: predict + evaluate + scoreboard.")
    _add_common_args(p_run)
    p_run.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    _add_evaluator_filter_args(p_run)
    _add_lane_override_args(p_run)
    _add_prepare_args(p_run)
    _add_prepml_args(p_run)
    p_run.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Allow re-running over existing evaluator outputs.",
    )

    # --- predict ---
    p_predict = subparsers.add_parser("predict", help="Generate predictions only.")
    _add_common_args(p_predict)
    p_predict.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    _add_lane_override_args(p_predict)
    _add_prepare_args(p_predict)
    _add_prepml_args(p_predict)
    p_predict.add_argument(
        "--output-dir", default=None,
        help="Override output directory (defaults to <scratch>/eval/<lane>/run_<TS>). "
             "Predictions go to <output-dir>/predictions.",
    )

    # --- prepare ---
    p_prepare = subparsers.add_parser("prepare", help="Build truth-aware bundles only (no prediction).")
    _add_common_args(p_prepare)
    _add_lane_override_args(p_prepare)
    p_prepare.add_argument(
        "--source-grib-root", required=True,
        help="Root directory of source GRIB files.",
    )
    p_prepare.add_argument(
        "--bundle-dir", default=None,
        help="Output directory for built bundles (default: <output-dir>/bundles).",
    )

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Run evaluators on existing predictions.")
    _add_common_args(p_eval)
    p_eval.add_argument(
        "--predictions-dir", required=True,
        help="Directory containing prediction .nc files.",
    )
    _add_evaluator_filter_args(p_eval)
    _add_lane_override_args(p_eval)
    p_eval.add_argument(
        "--checkpoint", default=None,
        help="Path to model checkpoint (for evaluators that require it, e.g. mechanistic).",
    )
    p_eval.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Allow re-running over existing evaluator outputs.",
    )
    p_eval.add_argument(
        "--plot-only", action="store_true", default=False,
        help="Skip run() and score(); only re-render plot() against existing results_dir. "
             "Useful for cheap re-plot after fixing plotting code (e.g. intermediate states).",
    )
    p_eval.add_argument(
        "--output-dir", default=None,
        help="Override output directory (defaults to <scratch>/eval/<lane>/run_<TS>). "
             "Use this with --plot-only to target an existing run directory.",
    )
    p_eval.add_argument(
        "--run-label", default="",
        help="Short display label for this run (used in TC/plot legends). "
             "Overrides the automatic fallback from the predictions directory name.",
    )

    # --- scoreboard ---
    p_sb = subparsers.add_parser("scoreboard", help="Generate scoreboard from evaluation results.")
    _add_common_args(p_sb)
    p_sb.add_argument(
        "--eval-dir", required=True,
        help="Root evaluation directory containing evaluator outputs.",
    )
    _add_evaluator_filter_args(p_sb)

    # --- report ---
    p_report = subparsers.add_parser("report", help="Generate HTML report for an evaluation run.")
    p_report.add_argument("--run-dir", required=True, help="Root directory of the evaluation run.")
    p_report.add_argument("--output", default=None, help="Output HTML path (default: <run-dir>/report.html).")

    # --- prepml-cleanup ---
    # List tracked prepml experiments and force-run their ecFlow `run/delete/*`
    # tasks. Bypasses lane/host config loading — operates purely on the ledger
    # and ecFlow.
    p_pcleanup = subparsers.add_parser(
        "prepml-cleanup",
        help="List and clean tracked prepml experiments via ecFlow.",
    )
    p_pcleanup.add_argument(
        "--list", action="store_true",
        help="Print the ledger (with ecFlow state) and exit.",
    )
    p_pcleanup.add_argument(
        "--expver", action="append", default=[],
        help="Clean a specific expver. Repeat for multiple.",
    )
    p_pcleanup.add_argument(
        "--scope", choices=("fdb", "all"), default="fdb",
        help="Which run/delete tasks to force-run. fdb (default) matches the announcement; "
             "all = fdb + mars + s3 + quaver + workdir (catalogue is always preserved).",
    )
    p_pcleanup.add_argument(
        "--dry-run", action="store_true",
        help="Print the ecflow_client commands that would fire; do not execute.",
    )
    p_pcleanup.add_argument(
        "--yes", action="store_true",
        help="Skip the final confirmation prompt.",
    )
    p_pcleanup.add_argument(
        "--no-ecflow", action="store_true",
        help="Skip the ecFlow state probe when listing (faster, offline-safe).",
    )

    # --- videogen ---
    # Backend: eval._backends.videogen (modular MP4 generator).
    # Scenes aren't enumerated here so eval.cli's import stays cheap; the
    # videogen entrypoint validates --scene against its own SCENES registry.
    p_videogen = subparsers.add_parser(
        "videogen", help="Render MP4 videos of downscaling predictions.",
    )
    p_videogen.add_argument("--scene", required=True,
                            help="Scene name (see eval._backends.videogen.scenes.SCENES).")
    p_videogen.add_argument("--mode", choices=("preview", "all"), default="preview")
    p_videogen.add_argument("--preview-valid", default=None,
                            help="Valid time YYYY-MM-DD for preview mode.")
    p_videogen.add_argument("--predictions-dir", default=None,
                            help="Override scene's predictions_dir.")
    p_videogen.add_argument("--output-dir", default=None,
                            help="Override scene's output_dir.")
    p_videogen.add_argument("--ckpt-label", default=None,
                            help="Override scene's ckpt_label (cosmetic).")

    return parser


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _parse_int_csv(raw: str) -> list[int]:
    """Parse comma-separated integers, sorted ascending."""
    return sorted(int(tok.strip()) for tok in raw.split(",") if tok.strip())


def _parse_str_csv(raw: str) -> list[str]:
    """Parse comma-separated strings, preserving order."""
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _build_lane_overrides(args: argparse.Namespace) -> dict:
    """Build overrides dict from CLI args that are not None."""
    predict_overrides: dict = {}
    if getattr(args, "members", None) is not None:
        predict_overrides["members"] = _parse_int_csv(args.members)
    if getattr(args, "steps", None) is not None:
        predict_overrides["steps"] = _parse_int_csv(args.steps)
    if getattr(args, "dates", None) is not None:
        predict_overrides["dates"] = _parse_str_csv(args.dates)
    if getattr(args, "weather_states", None) is not None:
        predict_overrides["weather_states"] = _parse_str_csv(args.weather_states)
    if getattr(args, "num_gpus_per_model", None) is not None:
        predict_overrides["num_gpus_per_model"] = int(args.num_gpus_per_model)
    # Note: --num-chunks is NOT propagated here. The loader's _deep_merge is only
    # shallow at the second level, so injecting {"env": {...}} here would clobber
    # the lane YAML's full predict.env block. Instead, --num-chunks is applied
    # directly to lane_config["predict"]["env"] after load_lane returns.
    if predict_overrides:
        return {"predict": predict_overrides}
    return {}


def _resolve_evaluators(args: argparse.Namespace, lane_config: dict) -> list[str]:
    """Three-step evaluator resolution.

    1. --only: run exactly those evaluators.
    2. --include-diagnostics: default + diagnostics groups.
    3. Otherwise: default group only.
    """
    evaluator_groups = lane_config.get("evaluator_groups", {})

    if getattr(args, "only", None) is not None:
        requested = _parse_str_csv(args.only)
        unknown = [e for e in requested if e not in ALL_EVALUATORS]
        if unknown:
            raise SystemExit(
                f"Unknown evaluator(s) in --only: {unknown}. "
                f"Valid evaluators: {ALL_EVALUATORS}"
            )
        return requested

    if getattr(args, "include_diagnostics", False):
        default_group = evaluator_groups.get("default", [])
        diag_group = evaluator_groups.get("diagnostics", [])
        # Preserve order, avoid duplicates
        combined: list[str] = list(default_group)
        for e in diag_group:
            if e not in combined:
                combined.append(e)
        return combined

    return list(evaluator_groups.get("default", []))


def _get_git_commit() -> str:
    """Return current git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _resolve_output_dir(host_config: dict, lane_name: str) -> Path:
    """Build output directory: <scratch_root>/eval/<lane>/run_<YYYYMMDDTHHMMSS>/"""
    scratch_root = Path(host_config["scratch_root"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return scratch_root / "eval" / lane_name / f"run_{timestamp}"


def _config_file_paths(lane_name: str, host_name: str) -> dict:
    """Return paths to the YAML config files that were loaded."""
    config_dir = Path(__file__).resolve().parent / "config"
    return {
        "lane": str(config_dir / "lanes" / f"{lane_name}.yaml"),
        "host": str(config_dir / "hosts" / f"{host_name}.yaml"),
    }


def _build_effective_config(
    args: argparse.Namespace,
    lane_config: dict,
    host_config: dict,
    lane_name: str,
    host_name: str,
    overrides: dict,
    evaluators: list[str],
    output_dir: Path,
) -> dict:
    """Build the effective config dict for emission."""
    code_root = host_config.get("code_root", "unknown")
    return {
        "lane": lane_name,
        "host": host_name,
        "checkpoint": getattr(args, "checkpoint", None),
        "predictions_dir": getattr(args, "predictions_dir", None),
        "eval_dir": getattr(args, "eval_dir", None),
        "resolved": lane_config,
        "overrides": overrides,
        "cli_args": sys.argv[1:],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "code_root": code_root,
        "config_file_paths": _config_file_paths(lane_name, host_name),
        "output_dir": str(output_dir),
        "evaluators": evaluators,
        "evaluators_run": [],
        "mode": getattr(args, "mode", "manual"),
        "expver": getattr(args, "expver", None),
    }


def _write_effective_config(config: dict, output_dir: Path) -> Path:
    """Write effective_config.json to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "effective_config.json"
    path.write_text(json.dumps(config, indent=2, default=str) + "\n")
    return path


def _update_effective_config_completion(
    output_dir: Path, evaluators_run: list[str],
) -> None:
    """Update effective_config.json with completion info."""
    path = output_dir / "effective_config.json"
    if path.exists():
        config = json.loads(path.read_text())
    else:
        config = {}
    config["evaluators_run"] = evaluators_run
    config["completion_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(config, indent=2, default=str) + "\n")


def _predict_bundle_pairs(predict_cfg: dict) -> list:
    """Return bundle_pairs from predict config as a list."""
    bundle_pairs_raw = predict_cfg.get("bundle_pairs", [])
    if isinstance(bundle_pairs_raw, str):
        return [bp.strip() for bp in bundle_pairs_raw.split(",") if bp.strip()]
    return list(bundle_pairs_raw)


SERIAL_PREPARE_RANK_ENV_VARS = (
    "SLURM_PROCID",
    "PMI_RANK",
    "PMIX_RANK",
    "OMPI_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_RANK",
    "RANK",
    "LOCAL_RANK",
)


def _distributed_rank_context_vars() -> dict[str, str]:
    return {
        key: value
        for key in SERIAL_PREPARE_RANK_ENV_VARS
        if (value := os.environ.get(key)) not in (None, "")
    }


# C8(iii): world-size env vars. A rank env var being set is harmless when the
# world is a single task (ntasks==1). We only refuse when there is genuinely
# more than one parallel task AND this is not rank 0.
SERIAL_PREPARE_WORLD_SIZE_ENV_VARS = (
    "SLURM_NTASKS",
    "SLURM_STEP_NUM_TASKS",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "WORLD_SIZE",
)


def _max_declared_world_size() -> int:
    """Largest world-size hint across known launchers (0 if none declared)."""
    sizes = [0]
    for key in SERIAL_PREPARE_WORLD_SIZE_ENV_VARS:
        value = os.environ.get(key)
        if value:
            try:
                sizes.append(int(value))
            except ValueError:
                pass
    return max(sizes)


def _all_ranks_zero(rank_vars: dict[str, str]) -> bool:
    """True if every declared rank var is rank 0."""
    for value in rank_vars.values():
        try:
            if int(value) != 0:
                return False
        except ValueError:
            return False
    return True


def _assert_serial_prepare_context() -> None:
    rank_vars = _distributed_rank_context_vars()
    if not rank_vars:
        return
    # C8(iii): allow when the launcher reports a single task (ntasks==1), or when
    # every rank var is 0 and no launcher declares more than one task. Bundle
    # prepare is inherently serial; a 1-task srun/rank-0 context is fine.
    world_size = _max_declared_world_size()
    if world_size <= 1 and _all_ranks_zero(rank_vars):
        return
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(rank_vars.items()))
    raise SystemExit(
        "Refusing serial bundle preparation inside a multi-task distributed "
        f"context ({rendered}; world_size={world_size}). Run "
        "`python -m eval.cli prepare` once with a single task (ntasks==1) or "
        "outside `srun`, then run prediction with `--bundle-dir <prepared-bundles>`."
    )


def _verify_predict_input_bundles(lane_config: dict, input_root: str) -> None:
    """Fail before prediction when a prepare lane is pointed at bad bundles."""
    if not lane_config.get("prepare"):
        return
    from eval.prepare.builder import verify_bundles

    predict_cfg = lane_config["predict"]
    verify_bundles(
        lane_config,
        Path(input_root),
        dates=list(predict_cfg.get("dates", [])),
        steps=[int(s) for s in predict_cfg.get("steps", [])],
        members=[int(m) for m in predict_cfg.get("members", [])],
        bundle_pairs=_predict_bundle_pairs(predict_cfg),
    )


def _resolve_predict_input_root(
    args: argparse.Namespace,
    lane_config: dict,
    host_config: dict,
    output_dir: Path,
    *,
    prepare_bundles: bool,
    allow_host_fallback: bool = True,
) -> str:
    """Resolve the prediction input_root and optionally build truth-aware bundles."""
    predict_cfg = lane_config["predict"]
    source_grib_root = getattr(args, "source_grib_root", None) or ""
    bundle_dir_arg = getattr(args, "bundle_dir", None)

    if lane_config.get("prepare") and source_grib_root:
        bundle_dir = Path(bundle_dir_arg) if bundle_dir_arg else output_dir / "bundles"
        if prepare_bundles:
            _assert_serial_prepare_context()
            from eval.prepare.builder import build_bundles

            LOG.info("=== Phase 0: Bundle preparation ===")
            build_bundles(
                lane_config=lane_config,
                bundle_dir=bundle_dir,
                source_grib_root=source_grib_root,
                dates=list(predict_cfg.get("dates", [])),
                steps=[int(s) for s in predict_cfg.get("steps", [])],
                members=[int(m) for m in predict_cfg.get("members", [])],
                bundle_pairs=_predict_bundle_pairs(predict_cfg),
                verification_path=output_dir / "bundle_build_verification.json",
            )
        return str(bundle_dir)

    if bundle_dir_arg:
        # Use pre-built bundles as input_root; skip rebuild.
        return str(bundle_dir_arg)

    # Resolve input_root: lane config takes precedence over host DATA_DIR.
    input_root = predict_cfg.get("input_root", "")
    if input_root:
        return str(input_root)

    if not allow_host_fallback:
        return ""

    env_setup = host_config.get("environment_setup", {})
    exports = env_setup.get("exports", {})
    return str(exports.get("DATA_DIR", ""))


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

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


def _write_evaluator_status(
    output_dir: Path, name: str, status: str, detail: str = "",
) -> None:
    """C4: record a per-evaluator status (ran/skipped/failed) into the run dir.

    Statuses accumulate in ``<output_dir>/evaluators/status.json`` so an
    operator (or a later automated check) can see exactly which evaluators ran,
    which were skipped, and which failed. Best-effort: never raise.
    """
    try:
        status_path = output_dir / "evaluators" / "status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
        entry: dict[str, str] = {
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if detail:
            entry["detail"] = detail
        data[name] = entry
        status_path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    except Exception:
        LOG.warning("Could not write evaluator status for '%s' (non-fatal)", name)


def _declared_evaluators(
    lane_config: dict, evaluators: list[str], *, checkpoint: str | None,
) -> list[str]:
    """C4: the set of evaluators the lane DECLARES and that *should* produce output.

    Starts from the requested set (already resolved from the lane's
    ``evaluator_groups.default`` upstream) and drops evaluators that are
    legitimately not expected to run in this invocation:
      - unknown evaluators (not in ALL_EVALUATORS),
      - evaluators whose ``requires`` cannot be satisfied (e.g. need a
        checkpoint when none was passed).
    Anything left is expected to produce output; a declared-but-missing
    evaluator is therefore a real gap, not an intentional skip.
    """
    declared: list[str] = []
    for name in evaluators:
        if name not in ALL_EVALUATORS:
            continue
        try:
            mod = importlib.import_module(f"eval.evaluators.{name}")
        except ImportError:
            # Import failure is itself a failure (collected separately); count it
            # as declared so the run still fails on it.
            declared.append(name)
            continue
        requires = getattr(mod, "EVALUATOR_SPEC", {}).get("requires", [])
        if "checkpoint" in requires and not checkpoint:
            # Cannot run without a checkpoint — not an unexpected gap.
            continue
        declared.append(name)
    return declared


def _run_evaluators(
    predictions_dir: Path,
    lane_config: dict,
    evaluators: list[str],
    output_dir: Path,
    *,
    overwrite: bool = False,
    plot_only: bool = False,
    checkpoint: str | None = None,
    run_label: str = "",
) -> list[str]:
    """Run selected evaluators on existing predictions. Returns list of evaluators that ran."""
    evaluators_run: list[str] = []
    failures: list[str] = []

    for name in evaluators:
        if name not in ALL_EVALUATORS:
            LOG.warning(
                "Skipping unknown evaluator '%s'. Valid: %s", name, ALL_EVALUATORS
            )
            continue

        # Import evaluator module
        try:
            mod = importlib.import_module(f"eval.evaluators.{name}")
        except ImportError as exc:
            LOG.error(
                "Cannot import evaluator 'eval.evaluators.%s'. "
                "Check that the module exists and has no import errors.",
                name,
            )
            failures.append(f"{name}.import: {exc}")
            _write_evaluator_status(output_dir, name, "failed", detail=str(exc))
            continue

        spec = getattr(mod, "EVALUATOR_SPEC", {})

        # Check requirements
        requires = spec.get("requires", [])
        if "checkpoint" in requires and not checkpoint:
            LOG.warning(
                "Skipping evaluator '%s': requires 'checkpoint' but none provided. "
                "Pass --checkpoint to include it.",
                name,
            )
            _write_evaluator_status(
                output_dir, name, "skipped", detail="requires checkpoint (none provided)"
            )
            continue

        # Determine results directory
        results_dir = output_dir / "evaluators" / name
        eval_config = lane_config.get(name, {})

        # C3: completion is tracked by a `.complete` marker written only after a
        # fully successful run/score/plot. A bare results_dir is NOT proof of
        # completion — a crash or a racing parallel evaluator can leave an empty
        # dir, which previously caused a silent skip.
        complete_marker = results_dir / ".complete"

        if plot_only:
            if not results_dir.exists():
                LOG.warning(
                    "Evaluator '%s' --plot-only: results_dir does not exist (%s). Skipping.",
                    name, results_dir,
                )
                continue
            LOG.info("Re-plotting evaluator (plot-only): %s", name)
        else:
            # C3: skip only when the completion marker exists (and not overwriting).
            if complete_marker.exists() and not overwrite:
                LOG.warning(
                    "Evaluator '%s' already completed at %s (.complete marker present). "
                    "Use --overwrite to re-run. Skipping.",
                    name, results_dir,
                )
                # Already-complete evaluators still count as produced output, so
                # the C4 declared-vs-run diff below does not flag them as missing.
                evaluators_run.append(name)
                _write_evaluator_status(output_dir, name, "skipped")
                continue

            # C3: a dir without the marker is stale (crash/race) — wipe and re-run.
            # --overwrite forces the same path so a fresh run always starts clean.
            if results_dir.exists():
                import shutil
                if not complete_marker.exists():
                    LOG.warning(
                        "Evaluator '%s' results_dir exists without .complete marker "
                        "(stale/partial). Removing and re-running: %s",
                        name, results_dir,
                    )
                shutil.rmtree(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            LOG.info("Running evaluator: %s", name)

        # Run (skipped in plot-only mode)
        run_fn = getattr(mod, "run", None)
        if run_fn is not None and not plot_only:
            try:
                run_fn(
                    predictions_dir, lane_config, eval_config,
                    output_dir=results_dir, overwrite=overwrite,
                    checkpoint=checkpoint,
                    run_label=run_label,
                )
            except Exception as exc:
                LOG.error("Evaluator '%s' run() failed", name, exc_info=True)
                failures.append(f"{name}.run: {exc}")
                _write_evaluator_status(output_dir, name, "failed", detail=str(exc))
                continue

        # Score
        score_fn = getattr(mod, "score", None)
        if score_fn is not None:
            try:
                scores = score_fn(
                    results_dir, lane_config, eval_config,
                    predictions_dir=predictions_dir,
                )
                if scores:
                    metrics_path = results_dir / "metrics.json"
                    metrics_path.write_text(
                        json.dumps(scores, indent=2, default=str) + "\n"
                    )
            except Exception as exc:
                LOG.error("Evaluator '%s' score() failed", name, exc_info=True)
                failures.append(f"{name}.score: {exc}")
                _write_evaluator_status(output_dir, name, "failed", detail=str(exc))
                continue

        # Plot
        plot_fn = getattr(mod, "plot", None)
        if plot_fn is not None:
            try:
                plot_fn(results_dir, lane_config, eval_config, output_dir=results_dir)
            except Exception as exc:
                LOG.error("Evaluator '%s' plot() failed", name, exc_info=True)
                failures.append(f"{name}.plot: {exc}")
                _write_evaluator_status(output_dir, name, "failed", detail=str(exc))
                continue

        # C3: write the completion marker only now, after run/score/plot all
        # succeeded, so a future invocation can trust it for skip decisions.
        if not plot_only:
            try:
                complete_marker.write_text(
                    datetime.now(timezone.utc).isoformat() + "\n"
                )
            except OSError:
                LOG.warning("Could not write .complete marker for '%s'", name)
        evaluators_run.append(name)
        _write_evaluator_status(output_dir, name, "ran")
        LOG.info("Evaluator '%s' completed. Output: %s", name, results_dir)

    # C4: an evaluator that the lane DECLARES but that produced no output (was
    # skipped for a missing requirement, an empty default group entry, etc.)
    # leaves a silent gap. Diff the declared set against what actually ran and
    # treat any declared-but-missing evaluator as a failure so the run cannot
    # "complete" with a hole in it.
    declared = _declared_evaluators(lane_config, evaluators, checkpoint=checkpoint)
    missing = [e for e in declared if e not in evaluators_run]
    for e in missing:
        LOG.error(
            "Declared evaluator '%s' produced no output (skipped or never ran).", e
        )
        _write_evaluator_status(output_dir, e, "skipped")
        failures.append(f"{e}: declared but produced no output")

    if failures:
        failure_lines = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"Evaluator failure(s):\n{failure_lines}")

    return evaluators_run


def _resolve_run_root(output_dir: Path) -> Path:
    """Resolve the run root from output_dir. See ``eval.lean_layout``."""
    return lean_layout.resolve_run_root(output_dir)


def _consolidate_plots(output_dir: Path) -> None:
    """Project the evaluator tree into the lean run-root bundle.

    Delegates to ``eval.lean_layout.project_lean_layout``, which lays down the
    top-level deliverables, ``plots/<name>/``, ``data/`` and an assembled
    ``metrics.json`` as a non-destructive, idempotent symlink view over
    ``evaluators/<name>/`` — replacing both the old flat plot copy here and the
    standalone ``finalize_lean_eval_layout.sbatch`` reorg step.

    Best-effort: the projection never raises, so a hiccup can't fail a run whose
    metrics and completion marker are already written.
    """
    lean_layout.project_lean_layout(output_dir)


def _run_scoreboard(
    eval_dir: Path,
    lane_config: dict,
    evaluators: list[str],
    output_dir: Path,
) -> None:
    """Generate scoreboard from evaluation results."""
    from eval.scoreboard.aggregator import aggregate_scores
    from eval.scoreboard.formatter import to_csv, to_markdown, to_pretty_text

    scores = aggregate_scores(eval_dir, lane_config, evaluators=evaluators)
    if not scores:
        raise RuntimeError(
            "Scoreboard produced no scores. "
            f"eval_dir={eval_dir} evaluators={evaluators}"
        )

    # Write outputs
    scoreboard_dir = output_dir / "scoreboard"
    scoreboard_dir.mkdir(parents=True, exist_ok=True)

    csv_path = to_csv(scores, scoreboard_dir / "scores.csv")
    md_path = to_markdown(scores, scoreboard_dir / "scores.md")
    text = to_pretty_text(scores)

    LOG.info("Scoreboard CSV:      %s", csv_path)
    LOG.info("Scoreboard Markdown: %s", md_path)
    print("\n--- Scoreboard ---")
    print(text)
    print()


def cmd_prepare(args: argparse.Namespace, lane_config: dict, host_config: dict, output_dir: Path) -> None:
    """Build truth-aware bundles only (no prediction)."""
    _assert_serial_prepare_context()
    from eval.prepare.builder import build_bundles

    prepare_cfg = lane_config.get("prepare")
    if not prepare_cfg:
        raise SystemExit(f"Lane '{args.lane}' has no 'prepare:' section in its config.")

    source_grib_root = args.source_grib_root
    bundle_dir_arg = getattr(args, "bundle_dir", None)
    bundle_dir = Path(bundle_dir_arg) if bundle_dir_arg else output_dir / "bundles"

    predict_cfg = lane_config.get("predict", {})
    bundle_pairs_raw = predict_cfg.get("bundle_pairs", [])
    if isinstance(bundle_pairs_raw, str):
        bundle_pairs_raw = [bp.strip() for bp in bundle_pairs_raw.split(",") if bp.strip()]

    build_bundles(
        lane_config=lane_config,
        bundle_dir=bundle_dir,
        source_grib_root=source_grib_root,
        dates=list(predict_cfg.get("dates", [])),
        steps=[int(s) for s in predict_cfg.get("steps", [])],
        members=[int(m) for m in predict_cfg.get("members", [])],
        bundle_pairs=list(bundle_pairs_raw),
        verification_path=bundle_dir.parent / "bundle_build_verification.json",
    )
    LOG.info("Bundle preparation complete. Bundles in: %s", bundle_dir)


def cmd_run(args: argparse.Namespace, lane_config: dict, host_config: dict, evaluators: list[str], output_dir: Path) -> None:
    """Full pipeline: predict + evaluate + scoreboard."""
    predictions_dir = output_dir / "predictions"
    checkpoint = args.checkpoint

    # Step 1: Predict
    LOG.info("=== Phase 1/3: Predictions ===")
    cmd_predict(args, lane_config, host_config, output_dir)

    # Step 2: Evaluate
    LOG.info("=== Phase 2/3: Evaluators ===")
    evaluators_run = _run_evaluators(
        predictions_dir, lane_config, evaluators, output_dir,
        overwrite=getattr(args, "overwrite", False),
        checkpoint=checkpoint,
    )

    # Step 3: Scoreboard
    LOG.info("=== Phase 3/3: Scoreboard ===")
    _run_scoreboard(output_dir, lane_config, evaluators, output_dir)

    # Step 4: record completion FIRST, then consolidate plots.
    # C2: completion must always be recorded; plot consolidation is cosmetic and
    # non-fatal, so it runs last so a plotting hiccup can never block the marker.
    _update_effective_config_completion(output_dir, evaluators_run)
    _consolidate_plots(output_dir)


def _apply_host_module_loads(host_config: dict) -> None:
    """C5 (a): best-effort apply host ``environment_setup.module_loads``.

    The inline eval subprocess needs the env that ``module load <mod>`` sets up
    (e.g. ``ecmwf-toolbox`` provides metview, used by regridded TC). When this
    process was launched outside the rendered sbatch (which already loads the
    modules), those vars are absent. We source the module system, run the
    loads, dump the resulting env, and import any *new/changed* vars into
    ``os.environ`` so child evaluators inherit them.

    Best-effort: if the module system isn't available or anything fails, we log
    and move on — the regridded assertion in _assert_metview_for_regridded_tc is
    the hard guard against a silent degrade.
    """
    module_loads = (
        host_config.get("environment_setup", {}).get("module_loads", []) or []
    )
    if not module_loads:
        return
    # Skip if modules already appear loaded (LOADEDMODULES is set by the module
    # system); avoids spawning a shell on every invocation inside sbatch.
    if os.environ.get("LOADEDMODULES"):
        return
    load_cmd = " && ".join(f"module load {m}" for m in module_loads)
    script = (
        "source /etc/profile.d/modules.sh 2>/dev/null || "
        "source /usr/share/Modules/init/bash 2>/dev/null || true; "
        f"{load_cmd} >/dev/null 2>&1; env"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", script],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        LOG.warning(
            "Could not apply host module_loads %s (non-fatal); regridded TC will "
            "be guarded by an explicit metview check.", module_loads, exc_info=True,
        )
        return
    if result.returncode != 0:
        LOG.warning("module load returned %d; continuing", result.returncode)
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Only import vars not already set by the user/sbatch so explicit
        # overrides win, mirroring the host_exports policy.
        if key and key not in os.environ:
            os.environ[key] = value


def _tc_support_implies_regridded(lane_config: dict) -> bool:
    """Return True when the lane's TC support mode needs metview (regridded path)."""
    tc_cfg = lane_config.get("tc") or {}
    mode = str(tc_cfg.get("support_mode", "")).strip().lower()
    return mode in ("regridded", "both")


def _assert_metview_for_regridded_tc(
    lane_config: dict, evaluators: list[str],
) -> None:
    """C5 (b): hard-fail instead of silently degrading regridded TC to native.

    Regridded/both TC support requires metview (from the ``ecmwf-toolbox``
    module). If the module wasn't loaded the import fails deep inside the TC
    backend and the result quietly degrades to native support (a different,
    incomparable measurement). Assert importability up front with a clear,
    actionable error.
    """
    if "tc" not in evaluators:
        return
    if not _tc_support_implies_regridded(lane_config):
        return
    try:
        import metview  # noqa: F401
    except Exception as exc:
        mode = str((lane_config.get("tc") or {}).get("support_mode", "")).strip()
        raise SystemExit(
            "TC support_mode="
            f"{mode!r} requires metview, but it is not importable: "
            f"{exc}. Load the host's ecmwf-toolbox module (e.g. "
            "`module load ecmwf-toolbox`) before running, or run inside the "
            "rendered sbatch which loads it. Refusing to silently degrade "
            "regridded TC to native."
        ) from exc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Parse args, resolve config, dispatch to subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # --- Report subcommand (no config needed) ---
    if args.subcommand == "report":
        from eval.report import generate_report
        run_dir = Path(args.run_dir)
        output = Path(args.output) if args.output else None
        report_path = generate_report(run_dir, output)
        LOG.info("Report written to %s", report_path)
        return

    # --- prepml-cleanup subcommand (no lane/host config needed) ---
    if args.subcommand == "prepml-cleanup":
        from eval.predict.prepml_cleanup import main as prepml_cleanup_main
        forwarded: list[str] = []
        if args.list:
            forwarded.append("--list")
        for ev in args.expver:
            forwarded += ["--expver", ev]
        forwarded += ["--scope", args.scope]
        if args.dry_run:
            forwarded.append("--dry-run")
        if args.yes:
            forwarded.append("--yes")
        if args.no_ecflow:
            forwarded.append("--no-ecflow")
        raise SystemExit(prepml_cleanup_main(forwarded))

    # --- Videogen subcommand (no lane/host config needed) ---
    if args.subcommand == "videogen":
        from eval._backends.videogen.__main__ import main as videogen_main
        forwarded = ["--scene", args.scene, "--mode", args.mode]
        if args.preview_valid:
            forwarded += ["--preview-valid", args.preview_valid]
        if args.predictions_dir:
            forwarded += ["--predictions-dir", str(args.predictions_dir)]
        if args.output_dir:
            forwarded += ["--output-dir", str(args.output_dir)]
        if args.ckpt_label:
            forwarded += ["--ckpt-label", args.ckpt_label]
        videogen_main(forwarded)
        return

    # --- Resolve config ---
    lane_name = args.lane

    lane_overrides = _build_lane_overrides(args)

    try:
        lane_config = load_lane(lane_name, overrides=lane_overrides or None)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Lane config not found: '{lane_name}'. "
            f"Available lanes are YAML files in eval/config/lanes/. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Failed to load lane config '{lane_name}': {exc}") from exc

    host_name = args.host or default_host_for_stage(lane_config, args.subcommand) or DEFAULT_HOST
    try:
        validate_lane_host_compatible(lane_name, lane_config, host_name, stage=args.subcommand)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    try:
        host_config = load_host(host_name)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Host config not found: '{host_name}'. "
            f"Available hosts are YAML files in eval/config/hosts/. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Failed to load host config '{host_name}': {exc}") from exc

    # --- Export host-declared env vars so subprocesses (predict.main, evaluators) see them ---
    # The host YAML's environment_setup.exports lists vars like DATA_DIR, GRID_DIR,
    # RESIDUAL_STATISTICS_DIR that OmegaConf interpolations and model loaders depend on.
    # Existing values in os.environ take precedence (so user overrides still work).
    host_exports = host_config.get("environment_setup", {}).get("exports", {}) or {}
    for key, value in host_exports.items():
        if key not in os.environ:
            os.environ[key] = str(value)

    # --- C5 (a): apply host module_loads so the inline eval subprocess gets the
    # same modules the sbatch would (e.g. ecmwf-toolbox -> metview for TC). ---
    _apply_host_module_loads(host_config)

    # --- Export lane-declared inference env vars (e.g. ANEMOI_INFERENCE_NUM_CHUNKS) ---
    # Apply the CLI --num-chunks override on top of lane predict.env so the
    # dry-run output and downstream subprocesses see the same merged value.
    predict_section = lane_config.get("predict")
    if isinstance(predict_section, dict):
        predict_env = dict(predict_section.get("env") or {})
        if getattr(args, "num_chunks", None) is not None:
            chunk_value = str(int(args.num_chunks))
            predict_env["ANEMOI_INFERENCE_NUM_CHUNKS"] = chunk_value
            predict_env["ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR"] = chunk_value
            predict_env["ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER"] = chunk_value
            predict_section["env"] = predict_env
        for key, value in predict_env.items():
            os.environ[key] = str(value)

    # --- Propagate --steps to evaluator sections ---
    # When --steps is passed, override not just predict.steps but also any
    # evaluator-specific steps (e.g. spectra.steps, spectra_ecmwf.steps) so
    # evaluators don't request forecast steps that don't exist in predictions.
    if getattr(args, "steps", None) is not None and args.subcommand in ("evaluate", "run"):
        cli_steps = _parse_int_csv(args.steps)
        for section_name, section_val in lane_config.items():
            if section_name != "predict" and isinstance(section_val, dict) and "steps" in section_val:
                section_val["steps"] = cli_steps
                lane_overrides.setdefault(section_name, {})["steps"] = cli_steps

    # --- Resolve evaluators (for subcommands that need them) ---
    evaluators: list[str] = []
    if args.subcommand in ("run", "evaluate", "scoreboard"):
        evaluators = _resolve_evaluators(args, lane_config)

    # --- Resolve output dir ---
    if args.subcommand == "scoreboard" and hasattr(args, "eval_dir") and args.eval_dir:
        output_dir = Path(args.eval_dir)
    elif args.subcommand == "evaluate" and hasattr(args, "predictions_dir") and args.predictions_dir:
        # Place evaluator outputs alongside predictions, unless --output-dir overrides
        explicit_out = getattr(args, "output_dir", None)
        output_dir = Path(explicit_out) if explicit_out else Path(args.predictions_dir).parent
    elif args.subcommand == "predict" and getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
    elif args.subcommand == "prepare":
        bundle_dir_arg = getattr(args, "bundle_dir", None)
        output_dir = Path(bundle_dir_arg).parent if bundle_dir_arg else _resolve_output_dir(host_config, lane_name)
    else:
        output_dir = _resolve_output_dir(host_config, lane_name)

    if args.subcommand in ("run", "predict") and "predict" in lane_config:
        input_root = _resolve_predict_input_root(
            args, lane_config, host_config, output_dir,
            prepare_bundles=False,
            allow_host_fallback=getattr(args, "mode", "manual") != "prepml",
        )
        if input_root:
            lane_config.setdefault("predict", {})["input_root"] = input_root

    if args.subcommand in ("run", "evaluate") and "tc" in lane_config:
        require_lane_analysis_reference(
            lane_name, (lane_config.get("tc") or {}).get("analysis_expid"),
        )

    # --- Build effective config ---
    effective = _build_effective_config(
        args, lane_config, host_config,
        lane_name, host_name, lane_overrides,
        evaluators, output_dir,
    )

    # --- Dry run ---
    if args.dry_run:
        print(json.dumps(effective, indent=2, default=str))
        if getattr(args, "mode", "manual") == "prepml" and args.subcommand in ("run", "predict"):
            from eval.predict.prepml_config import generate_prepml_config
            from eval.predict.prepml import resolve_expver
            try:
                resolved_expver = resolve_expver(getattr(args, "expver", None), lane_config)
                prepml_cfg = generate_prepml_config(
                    lane_config=lane_config,
                    checkpoint_path=getattr(args, "checkpoint", ""),
                    runner_override=getattr(args, "prepml_runner", None),
                )
                import yaml
                print("\n--- PrepML Config Preview ---")
                print(yaml.dump(prepml_cfg, default_flow_style=False, sort_keys=False))
                print(f"Expver: {resolved_expver}")
            except Exception as exc:
                print(f"\n--- PrepML Config Preview (error) ---\n{exc}")
        return

    # --- Preflight: write effective config ---
    config_path = _write_effective_config(effective, output_dir)
    LOG.info("Effective config written to %s", config_path)

    # --- Validate evaluator names ---
    unknown_evals = [e for e in evaluators if e not in ALL_EVALUATORS]
    if unknown_evals:
        raise SystemExit(
            f"Unknown evaluator(s) in lane config evaluator_groups: {unknown_evals}. "
            f"Valid evaluators: {ALL_EVALUATORS}"
        )

    # --- C5 (b): for run/evaluate with regridded/both TC, require metview now
    # so we fail loudly instead of silently degrading to native support. ---
    if args.subcommand in ("run", "evaluate"):
        _assert_metview_for_regridded_tc(lane_config, evaluators)

    # --- Dispatch ---
    if args.subcommand == "run":
        cmd_run(args, lane_config, host_config, evaluators, output_dir)
    elif args.subcommand == "predict":
        cmd_predict(args, lane_config, host_config, output_dir)
    elif args.subcommand == "prepare":
        cmd_prepare(args, lane_config, host_config, output_dir)
    elif args.subcommand == "evaluate":
        predictions_dir = Path(args.predictions_dir)
        evaluators_run = _run_evaluators(
            predictions_dir, lane_config, evaluators, output_dir,
            overwrite=getattr(args, "overwrite", False),
            plot_only=getattr(args, "plot_only", False),
            checkpoint=getattr(args, "checkpoint", None),
            run_label=getattr(args, "run_label", ""),
        )
        # C2: record completion FIRST (always), then consolidate plots (non-fatal).
        _update_effective_config_completion(output_dir, evaluators_run)
        _consolidate_plots(output_dir)
    elif args.subcommand == "scoreboard":
        eval_dir = Path(args.eval_dir)
        _run_scoreboard(eval_dir, lane_config, evaluators, output_dir)


if __name__ == "__main__":
    main()
