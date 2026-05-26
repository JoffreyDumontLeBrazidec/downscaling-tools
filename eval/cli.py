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

from eval.config.loader import (
    default_host_for_stage,
    load_host,
    load_lane,
    validate_lane_host_compatible,
)
from eval.paths import resolve_eval_root

LOG = logging.getLogger(__name__)

ALL_EVALUATORS = [
    "tc", "spectra", "surface", "region_plot",
    "sigma", "mechanistic", "intermediate",
    "spectra_ecmwf", "mlflow",
    "precip_dist", "precip_events",
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


def _assert_serial_prepare_context() -> None:
    rank_vars = _distributed_rank_context_vars()
    if not rank_vars:
        return
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(rank_vars.items()))
    raise SystemExit(
        "Refusing serial bundle preparation inside a distributed rank context "
        f"({rendered}). Run `python -m eval.cli prepare` once outside `srun`, "
        "then run prediction with `--bundle-dir <prepared-bundles>`."
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
    if num_gpus_per_model and int(num_gpus_per_model) > 1 and os.environ.get("SLURM_JOB_ID"):
        n = str(int(num_gpus_per_model))
        cmd = ["srun", "--ntasks", n, "--ntasks-per-node", n] + cmd

    LOG.info("Running predictions: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    LOG.info("Predictions written to %s", predictions_dir)


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

    for name in evaluators:
        if name not in ALL_EVALUATORS:
            LOG.warning(
                "Skipping unknown evaluator '%s'. Valid: %s", name, ALL_EVALUATORS
            )
            continue

        # Import evaluator module
        try:
            mod = importlib.import_module(f"eval.evaluators.{name}")
        except ImportError:
            LOG.error(
                "Cannot import evaluator 'eval.evaluators.%s'. "
                "Check that the module exists and has no import errors.",
                name,
            )
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
            continue

        # Determine results directory
        results_dir = output_dir / "evaluators" / name
        eval_config = lane_config.get(name, {})

        if plot_only:
            if not results_dir.exists():
                LOG.warning(
                    "Evaluator '%s' --plot-only: results_dir does not exist (%s). Skipping.",
                    name, results_dir,
                )
                continue
            LOG.info("Re-plotting evaluator (plot-only): %s", name)
        else:
            # Check overwrite guard
            if results_dir.exists() and not overwrite:
                LOG.warning(
                    "Evaluator '%s' output already exists at %s. "
                    "Use --overwrite to re-run. Skipping.",
                    name, results_dir,
                )
                continue

            if results_dir.exists() and overwrite:
                import shutil
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
            except Exception:
                LOG.error("Evaluator '%s' run() failed", name, exc_info=True)
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
            except Exception:
                LOG.error("Evaluator '%s' score() failed", name, exc_info=True)

        # Plot
        plot_fn = getattr(mod, "plot", None)
        if plot_fn is not None:
            try:
                plot_fn(results_dir, lane_config, eval_config, output_dir=results_dir)
            except Exception:
                LOG.error("Evaluator '%s' plot() failed", name, exc_info=True)

        evaluators_run.append(name)
        LOG.info("Evaluator '%s' completed. Output: %s", name, results_dir)

    return evaluators_run


def _resolve_run_root(output_dir: Path) -> Path:
    """Resolve the run root from output_dir.

    output_dir must be either the run root itself (containing evaluators/)
    directly) or <run_root>/data (the canonical layout when evaluate is
    called with --predictions-dir).  Any other structure raises ValueError
    so callers cannot silently misplace outputs.
    """
    if output_dir.name == "data":
        return output_dir.parent
    if (output_dir / "evaluators").exists() or (output_dir / "predictions").exists():
        return output_dir
    raise ValueError(
        f"Cannot resolve run root from output_dir={output_dir!r}. "
        "Expected either <run_root>/data or a directory containing evaluators/ or predictions/."
    )


def _consolidate_plots(output_dir: Path) -> None:
    """Copy all PDFs and PNGs from evaluators/* to <run_root>/plots/.

    Always writes to <run_root>/plots/ regardless of how output_dir is
    structured, so plots are never buried inside data/.
    """
    import shutil
    run_root = _resolve_run_root(output_dir)
    plots_dir = run_root / "plots"
    # Wipe and rebuild so stale files from previous runs or parallel
    # evaluators never linger.  Each call produces a complete snapshot.
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    evaluators_dir = output_dir / "evaluators"
    if not evaluators_dir.exists():
        return
    for src in sorted(evaluators_dir.rglob("*.pdf")) + sorted(evaluators_dir.rglob("*.png")):  # type: ignore[operator]
        shutil.copy2(src, plots_dir / src.name)
    LOG.info("Plots consolidated to %s (%d files)", plots_dir, len(list(plots_dir.iterdir())))


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

    # Step 4: Consolidate all plots to <run_root>/plots/
    _consolidate_plots(output_dir)

    # Update effective config with completion info
    _update_effective_config_completion(output_dir, evaluators_run)


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
        _consolidate_plots(output_dir)
        _update_effective_config_completion(output_dir, evaluators_run)
    elif args.subcommand == "scoreboard":
        eval_dir = Path(args.eval_dir)
        _run_scoreboard(eval_dir, lane_config, evaluators, output_dir)


if __name__ == "__main__":
    main()
