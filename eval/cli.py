"""Unified CLI entry point for the evaluation framework.

Canonical invocation: python -m eval.cli <subcommand>

Subcommands:
    run         Full pipeline: predict + evaluate + scoreboard
    predict     Generate predictions only (subprocess call to eval.predict.main)
    evaluate    Run evaluators on existing predictions
    scoreboard  Generate scoreboard from existing evaluation results
    tctracker   Produce ECMWF tctracker basin-track archives from a PrepML/FDB
                expver AND its references (ctrl expver, target ENFO, input EEFO)
                on one shared tracking support (--track-sources, --months)
    tccompare   Compare tctracker track sets across those sources and render
                the month-scale TC track figure suite + metrics JSON

The tracker pair (tctracker + tccompare) is the month-scale, track-based TC
diagnostic panel for prepml campaigns. It never feeds scoreboards: TC verdicts
stay with the box-based raw-extremes `tc` evaluator on the canonical support.
Operational runbook (read this before tracker work):
/home/ecm5702/dev/docs/epics/completed_epics/tc_track/TCTRACKER_EVAL_CLI.md
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

LOG = logging.getLogger(__name__)

ALL_EVALUATORS = [
    "tc", "spectra", "surface", "region_plot",
    "sigma", "sigma_loss", "mechanistic", "intermediate",
    "spectra_ecmwf", "spectra_ecmwf_v2", "mlflow",
    "precip_dist", "precip_events", "precip_scores",
    "interp", "probabilistic", "spread_proxy",
    "quaver", "local_global",
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
    p_run.add_argument("--output-dir", default=None, help="Override output directory (defaults to <scratch>/eval/<lane>/run_<TS>).")
    p_run.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Allow re-running over existing evaluator outputs.",
    )
    p_run.add_argument(
        "--vs-baseline", action="store_true", default=False,
        help="After the scoreboard step, diff this run against the lane BASELINE "
             "(top of the lane scoreboard) and write scoreboard/vs_baseline.md.",
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
    p_eval.add_argument(
        "--expver", default=None,
        help="PrepML/FDB expver of the run being evaluated. When set, the quaver "
             "probabilistic scorecard is run automatically (FDB-based).",
    )
    p_eval.add_argument(
        "--vs-baseline", action="store_true", default=False,
        help="After evaluating, diff this run's scoreboard scores against the lane BASELINE "
             "(top of the lane scoreboard) and write scoreboard/vs_baseline.md.",
    )

    # --- scoreboard ---
    p_sb = subparsers.add_parser("scoreboard", help="Generate scoreboard from evaluation results.")
    _add_common_args(p_sb)
    p_sb.add_argument(
        "--eval-dir", required=True,
        help="Root evaluation directory containing evaluator outputs.",
    )
    _add_evaluator_filter_args(p_sb)
    p_sb.add_argument(
        "--vs-baseline", action="store_true", default=False,
        help="Also diff the scores against the lane BASELINE (top of the lane scoreboard) "
             "and write scoreboard/vs_baseline.md.",
    )

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

    # --- evolution ---
    p_evo = subparsers.add_parser(
        "evolution",
        help="Plot how an experiment is evolving vs a reference run, the EEFO input and the "
             "ENFO target (one row per weather state, one column per metric family).",
    )
    p_evo.add_argument("--exp", action="append", required=True,
                       help="LABEL=/path/to/ladder.json (repeatable). `baseline:<lane>` resolves "
                            "to the lane baseline's archived ladder card.")
    p_evo.add_argument("--ref", default=None,
                       help="LABEL=/path/to/ladder.json -- reference RUN, drawn as its own curve. "
                            "`baseline:<lane>` (e.g. baseline:o96_o320) resolves to the lane "
                            "BASELINE's archived ladder card — the standard during-run and "
                            "end-of-run comparison.")
    p_evo.add_argument("--input", dest="input_ref", default=None,
                       help="LABEL=/path/to/flat.json -- the INPUT anchor (REQUIRED)")
    p_evo.add_argument("--target", dest="target_ref", default=None,
                       help="LABEL=/path/to/flat.json -- the TARGET anchor (REQUIRED)")
    p_evo.add_argument("--hline", action="append", default=[],
                       help="LABEL=/path/to/flat.json -- any further flat anchor (repeatable)")
    p_evo.add_argument("--allow-missing-references", action="store_true",
                       help="bootstrap a lane with no reference yet; stamps the gap on the figure")
    p_evo.add_argument("--rows", default=None, help="comma-separated weather states")
    p_evo.add_argument("--columns", default=None, help="comma-separated metric families")
    p_evo.add_argument("--region", default="n.hem")
    p_evo.add_argument("--title", default=None, help="optional figure title (default: none)")
    p_evo.add_argument("--out", required=True)
    p_evo.add_argument("--allow-mixed-support", action="store_true")

    # --- tctracker ---
    p_tctracker = subparsers.add_parser(
        "tctracker",
        help="Produce and verify ECMWF tctracker basin track archives from a PrepML/FDB expver and its references (--track-sources).",
        description=(
            "Produce, verify, and parse ECMWF tctracker basin-track archives. "
            "Default: one rd expver. With --track-sources, the same tracker "
            "settings also run over ctrl/target/input references so every "
            "track set shares ONE support; operational references are cached "
            "under <scratch>/eval/tcrefs/ and reused across campaigns. "
            "Runbook: docs/epics/completed_epics/tc_track/TCTRACKER_EVAL_CLI.md "
            "(month-scale section). Compare the results with `eval.cli tccompare`."
        ),
    )
    _add_common_args(p_tctracker)
    _add_lane_override_args(p_tctracker)
    p_tctracker.add_argument("--expver", required=True, help="PrepML/FDB expver to track, e.g. j761.")
    p_tctracker.add_argument("--output-dir", default=None, help="Tracker run root (default: <scratch>/eval/<lane>/tctracker/<expver>).")
    p_tctracker.add_argument("--time", default=None, help="Forecast cycle hour, e.g. 00.")
    p_tctracker.add_argument("--start-step", type=int, default=None, help="First forecast step passed to tctracker -s.")
    p_tctracker.add_argument("--end-step", type=int, default=None, help="Last forecast step passed to tctracker -f.")
    p_tctracker.add_argument("--step-interval", type=int, default=None, help="Forecast step interval passed to tctracker -i.")
    p_tctracker.add_argument("--grid", type=int, default=None, help="Output grid resolution passed to tctracker -r.")
    p_tctracker.add_argument("--class", dest="fdb_class", default=None, help="FDB class passed to tctracker -C.")
    p_tctracker.add_argument("--type", dest="fdb_type", default=None, help="FDB type passed to tctracker -T.")
    p_tctracker.add_argument("--stream", default=None, help="FDB stream passed to tctracker -S.")
    p_tctracker.add_argument("--vorticity", choices=("true", "false"), default=None, help="Whether tctracker should read vorticity (-v).")
    p_tctracker.add_argument("--model-keyword", default=None, help="Value exported as model_keyword before invoking tctracker.")
    p_tctracker.add_argument("--module", default=None, help="Environment module to load before invoking tctracker (default: tctracker).")
    p_tctracker.add_argument("--overwrite", action="store_true", default=False, help="Re-run targets even if their tar already exists.")
    p_tctracker.add_argument("--verify-only", action="store_true", default=False, help="Only verify existing tars/manifests; do not run tctracker.")
    p_tctracker.add_argument("--parse-only", action="store_true", default=False, help="Only parse existing tracks; do not run tctracker.")
    p_tctracker.add_argument("--slurm-script", default=None, help="Write a resumable sbatch script per source to this path (role-suffixed when multi-source) and exit.")
    p_tctracker.add_argument("--role", default="model", help="Role label for this expver's tracks (default: model).")
    p_tctracker.add_argument(
        "--track-sources", default=None,
        help=(
            "Comma-separated roles to track in one invocation, e.g. "
            "'model,ctrl=j95z,target,input'. Bare 'target'/'input' resolve from "
            "lane tctracker.sources / prepml blocks; reference (non-rd) sources "
            "are cached under <scratch>/eval/tcrefs/ and reused across expvers."
        ),
    )
    p_tctracker.add_argument("--months", default=None, help="Comma-separated YYYYMM months expanded to daily dates (alternative to --dates).")
    p_tctracker.add_argument("--no-check-fdb", action="store_true", default=False, help="Skip the FDB completeness preflight for rd expvers.")
    p_tctracker.add_argument("--track-incomplete", action="store_true", default=False, help="Track partial/empty FDB dates too (default: skip them with a warning).")

    # --- tccompare ---
    p_tcc = subparsers.add_parser(
        "tccompare",
        help="Compare tctracker track sets (model vs ctrl/target/input) and render the TC track figure suite.",
        description=(
            "Compare track sets produced by `eval.cli tctracker` and render the "
            "month-scale TC figure suite (track maps, density vs target, "
            "intensity log-PDF + ratio, counts, step intensity, case panels) "
            "plus tc_tracks_metrics.json. Pin --dates to the intersection of "
            "complete dates when sources have unequal coverage. Diagnostic "
            "panel only — TC verdicts stay with the box-based raw-extremes tc "
            "evaluator. Runbook: docs/epics/completed_epics/tc_track/"
            "TCTRACKER_EVAL_CLI.md (month-scale section)."
        ),
    )
    _add_common_args(p_tcc)
    p_tcc.add_argument(
        "--sources", required=True,
        help="Comma-separated role=value specs: rd expver (model=j9f3), ref class:stream:expver (target=od:enfo:0001), absolute run-root path, or a bare role resolved from lane defaults (target,input).",
    )
    p_tcc.add_argument("--months", required=True, help="Comma-separated YYYYMM months in scope.")
    p_tcc.add_argument(
        "--dates", default=None,
        help=(
            "Restrict ALL sources to these init dates (comma YYYYMMDD). Use for "
            "paired-window comparisons when sources have unequal date coverage "
            "(different weather in scope would confound the distributions)."
        ),
    )
    p_tcc.add_argument("--basins", default="atl", help="Comma-separated basins (default: atl).")
    p_tcc.add_argument("--label", default=None, help="Campaign label for the output dir (default: months joined).")
    p_tcc.add_argument("--out", default=None, help="Output dir (default: <scratch>/eval/<lane_short>/tctracks/<label>).")
    p_tcc.add_argument("--reparse", action="store_true", default=False, help="Re-parse source tars even if parsed tables exist.")
    p_tcc.add_argument("--no-plots", action="store_true", default=False, help="Metrics only; skip figure rendering.")
    p_tcc.add_argument("--top-k-cases", type=int, default=3, help="Deepest-target case pages per case basin (default: 3).")
    p_tcc.add_argument("--case-basins", default="atl", help="Comma-separated basins that get per-storm case pages (default: atl).")
    p_tcc.add_argument("--plot-only", action="store_true", default=False, help="Re-render the report from cached parsed tables + existing tc_tracks_metrics.json (no re-scoring).")
    p_tcc.add_argument("--per-month-pages", action="store_true", default=False, help="Also render one focus-basin stats page per month (default: pooled pages only).")

    # --- membermaps ---
    from eval._backends.region_plotting.plot_member_wind_maps import build_arg_parser as _membermaps_parser
    subparsers.add_parser(
        "membermaps",
        parents=[_membermaps_parser(add_help=False)],
        help="Render single-member 10 m wind-speed cutout maps (EEFO input / ENFO truth / prediction arms) from predictions_*.nc or GRIB files.",
        description=(
            "Render the single-member 10 m wind-speed map set used for "
            "member-level case inspection: EEFO input, operational ENFO truth "
            "and one prediction panel per --run, all with a shared colour "
            "scale, projection and title style. Sources are the retrieved "
            "predictions_*.nc files (which embed x/y/y_pred); --grib panels "
            "cover steps absent from predictions (e.g. step 0 read from "
            "FDB/MARS). Diagnostic maps only — no scoring. Sits alongside the "
            "TC contour suite (plot_tc_contours_from_predictions)."
        ),
    )

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


def _with_prepml_quaver(evaluators: list[str], args: argparse.Namespace) -> list[str]:
    """Auto-include the quaver scorecard for prepml evaluations (expver set).

    Quaver is FDB-based and only meaningful when the run published an ensemble to
    FDB under an expver. We avoid even listing it for manual runs. Applied to the
    default / --include-diagnostics paths; --only stays explicit.
    """
    if getattr(args, "expver", None) and "quaver" not in evaluators:
        return [*evaluators, "quaver"]
    return evaluators


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
        return _with_prepml_quaver(combined, args)

    return _with_prepml_quaver(list(evaluator_groups.get("default", [])), args)


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

    C2: plotting/consolidation must NEVER fail the run. Metrics and the
    completion marker are already written by the time this runs; a missing
    source file (race/crash) previously surfaced as a copy2->copystat
    FileNotFoundError that made the evaluate job exit nonzero AFTER metrics
    were computed, cancelling the afterok scoreboard. Every step here is
    therefore best-effort: log and continue instead of raising.
    """
    import contextlib
    import shutil
    try:
        run_root = _resolve_run_root(output_dir)
        plots_dir = run_root / "plots"
        # Wipe and rebuild so stale files from previous runs or parallel
        # evaluators never linger.  Each call produces a complete snapshot.
        if plots_dir.exists():
            shutil.rmtree(plots_dir, ignore_errors=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        evaluators_dir = output_dir / "evaluators"
        if not evaluators_dir.exists():
            return
        copied = 0
        for src in sorted(evaluators_dir.rglob("*.pdf")) + sorted(evaluators_dir.rglob("*.png")):  # type: ignore[operator]
            # Use copyfile (data only, no copystat) + suppress OSError so a
            # file vanishing mid-copy can never abort the run.
            with contextlib.suppress(OSError):
                shutil.copyfile(src, plots_dir / src.name)
                copied += 1
        LOG.info("Plots consolidated to %s (%d files)", plots_dir, copied)
    except Exception:
        # Last-resort guard: consolidation is cosmetic and must not fail the run.
        LOG.warning("Plot consolidation failed (non-fatal)", exc_info=True)


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



def cmd_tctracker(args: argparse.Namespace, lane_config: dict, host_config: dict, output_dir: Path) -> None:
    """Run, verify, or parse ECMWF tctracker archives for one or more sources.

    Default = the single --expver under --role (back-compatible). With
    --track-sources, the same tracker settings run over every requested role
    (model expver + ctrl/target/input references) so all tracks share ONE
    support; reference tars land in the shared tcrefs cache.
    """
    import dataclasses

    from eval._backends.tctracker import (
        build_config, completeness_report, expand_months, parse_atlantic_tracks,
        parse_sources_arg, render_slurm_script, resolve_source_configs,
        run_batch, verify_outputs, write_atlantic_summary,
        write_verification_summary,
    )
    from eval._backends.tctracker.tables import parse_run_root

    if getattr(args, "months", None) and not getattr(args, "dates", None):
        args.dates = ",".join(expand_months(args.months))

    base_config = build_config(args, lane_config, host_config, output_dir)
    roles = parse_sources_arg(getattr(args, "track_sources", None))
    if roles:
        model_override = roles.get("model")
        if model_override and model_override != base_config.expver:
            raise SystemExit("--track-sources model=<expver> must match --expver")
        sources = resolve_source_configs(base_config, roles, lane_config, host_config)
    else:
        sources = [(getattr(args, "role", "model") or "model", base_config.expver, base_config)]

    if getattr(args, "slurm_script", None):
        base_path = Path(args.slurm_script)
        for role, source_id, config in sources:
            script = render_slurm_script(
                config,
                code_root=host_config["code_root"],
                venv_activate=host_config["environment_setup"]["venv_activate"],
            )
            script_path = base_path if len(sources) == 1 else base_path.with_name(
                f"{base_path.stem}_{role}_{source_id}{base_path.suffix or '.sbatch'}"
            )
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script, encoding="utf-8")
            script_path.chmod(script_path.stat().st_mode | 0o755)
            LOG.info("tctracker sbatch script (%s=%s) written to %s", role, source_id, script_path)
        return

    failures: list[str] = []
    for role, source_id, config in sources:
        LOG.info("=== tctracker source %s=%s (%s/%s/%s) -> %s",
                 role, source_id, config.fdb_class, config.stream, config.expver,
                 config.output_dir)
        if not getattr(args, "verify_only", False) and not getattr(args, "parse_only", False):
            # Warn-only FDB completeness preflight for rd expvers: partial or
            # empty dates are skipped by default (a tracker run on a half-
            # written date would silently produce truncated tracks).
            if config.fdb_class == "rd" and not getattr(args, "no_check_fdb", False):
                report = completeness_report(config)
                config.manifests_dir.mkdir(parents=True, exist_ok=True)
                (config.manifests_dir / "fdb_completeness.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8",
                )
                if report["checked"] and not getattr(args, "track_incomplete", False):
                    keep = tuple(d for d in config.dates if d in set(report["complete"]))
                    if keep != config.dates:
                        LOG.warning("%s=%s: tracking %d/%d complete dates",
                                    role, source_id, len(keep), len(config.dates))
                        config = dataclasses.replace(config, dates=keep)
            if not config.dates:
                LOG.warning("%s=%s: no complete dates to track; skipping source", role, source_id)
                continue
            try:
                run_batch(config)
            except RuntimeError as exc:
                failures.append(f"{role}={source_id}: {exc}")

        verification = verify_outputs(config)
        md_path, json_path = write_verification_summary(config, verification)
        LOG.info("verification (%s=%s) written to %s", role, source_id, md_path)
        if verification["issues"] and not getattr(args, "parse_only", False):
            failures.append(f"{role}={source_id}: {len(verification['issues'])} verification issue(s); see {json_path}")

        # Parse the WHOLE run root, not just this invocation's targets:
        # member-sliced production jobs run concurrently against one run root,
        # and a per-config parse would leave whichever member finished last.
        parsed_dir = parse_run_root(config.output_dir, role=role, source_id=source_id)
        LOG.info("parsed tables (%s=%s) written to %s", role, source_id, parsed_dir)
        if role == "model":  # keep the historical Atlantic summary artifacts
            tracks = parse_atlantic_tracks(config)
            write_atlantic_summary(config, tracks)

    if failures:
        raise RuntimeError("tctracker source failures:\n" + "\n".join(failures))


def cmd_tccompare(args: argparse.Namespace, lane_config: dict, host_config: dict, output_dir: Path) -> None:
    """Compare track sets from multiple tctracker sources and render figures."""
    from eval.evaluators.tctracks.runner import run as tccompare_run

    months = [m.strip() for m in str(args.months).split(",") if m.strip()]
    basins = [b.strip() for b in str(args.basins).split(",") if b.strip()]
    dates = [d.strip() for d in str(args.dates).split(",") if d.strip()] if getattr(args, "dates", None) else None
    tccompare_run(
        sources_arg=args.sources,
        months=months,
        basins=basins,
        dates=dates,
        lane_name=args.lane,
        lane_config=lane_config,
        host_config=host_config,
        out_dir=output_dir,
        reparse=getattr(args, "reparse", False),
        no_plots=getattr(args, "no_plots", False),
        top_k_cases=getattr(args, "top_k_cases", 3),
        plot_only=getattr(args, "plot_only", False),
        per_month_pages=getattr(args, "per_month_pages", False),
        case_basins=[b.strip() for b in str(getattr(args, "case_basins", "") or "").split(",") if b.strip()] or None,
    )

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

    # --- membermaps subcommand (no lane/host config needed) ---
    if args.subcommand == "membermaps":
        from eval._backends.region_plotting.plot_member_wind_maps import run as membermaps_run
        raise SystemExit(membermaps_run(args))

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

    # --- Evolution subcommand (reads ladder cards; no lane/host config needed) ---
    if args.subcommand == "evolution":
        from eval.jobs.evolution import main as evolution_main

        def _resolve_card_spec(spec: str) -> str:
            """`baseline:<lane>` or `LABEL=baseline:<lane>` -> the lane baseline's archived
            ladder card (LABEL defaults to baseline-<ckpt8>)."""
            label, _, path = spec.rpartition("=")
            if path.startswith("baseline:"):
                from eval.baseline import baseline_ladder_card
                auto_label, card = baseline_ladder_card(path.split(":", 1)[1])
                return f"{label or auto_label}={card}"
            return spec

        forwarded: list[str] = ["--out", str(args.out), "--region", args.region]
        for e in args.exp:
            forwarded += ["--exp", _resolve_card_spec(e)]
        if args.ref:
            forwarded += ["--ref", _resolve_card_spec(args.ref)]
        if args.input_ref:
            forwarded += ["--input", args.input_ref]
        if args.target_ref:
            forwarded += ["--target", args.target_ref]
        for h in args.hline:
            forwarded += ["--hline", h]
        if args.allow_missing_references:
            forwarded.append("--allow-missing-references")
        if args.rows:
            forwarded += ["--rows", args.rows]
        if args.columns:
            forwarded += ["--columns", args.columns]
        if args.title:
            forwarded += ["--title", args.title]
        if args.allow_mixed_support:
            forwarded.append("--allow-mixed-support")
        evolution_main(forwarded)
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

    # --- Surface host-declared stage caveats (WARN ONLY, never a gate) ---
    # Hosts may declare `stage_warnings: {<stage-name>: "text"}`. A host can be perfectly
    # valid for one checkpoint class and degrading for another, so this warns and
    # continues rather than refusing (defaults-not-validators doctrine).
    _stage_warning = (host_config.get("stage_warnings") or {}).get(args.subcommand)
    if _stage_warning:
        LOG.warning(
            "host %r on stage %r: %s", host_name, args.subcommand, " ".join(str(_stage_warning).split())
        )

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
    elif args.subcommand in ("run", "predict") and getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
    elif args.subcommand == "prepare":
        bundle_dir_arg = getattr(args, "bundle_dir", None)
        output_dir = Path(bundle_dir_arg).parent if bundle_dir_arg else _resolve_output_dir(host_config, lane_name)
    elif args.subcommand == "tctracker":
        from eval._backends.tctracker.pipeline import default_output_dir
        explicit_out = getattr(args, "output_dir", None)
        output_dir = Path(explicit_out) if explicit_out else default_output_dir(
            host_config, lane_name, lane_config, getattr(args, "expver")
        )
    elif args.subcommand == "tccompare":
        from eval._backends.tctracker.pipeline import _lane_short_name
        label = getattr(args, "label", None) or "_".join(
            m.strip() for m in str(args.months).split(",") if m.strip()
        )
        explicit_out = getattr(args, "out", None)
        output_dir = Path(explicit_out) if explicit_out else (
            Path(host_config["scratch_root"]) / "eval"
            / _lane_short_name(lane_name, lane_config) / "tctracks" / label
        )
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
        if args.subcommand == "tctracker":
            from eval._backends.tctracker import build_config, dry_run_payload
            effective["tctracker"] = dry_run_payload(
                build_config(args, lane_config, host_config, output_dir)
            )
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
    elif args.subcommand == "tctracker":
        cmd_tctracker(args, lane_config, host_config, output_dir)
    elif args.subcommand == "tccompare":
        cmd_tccompare(args, lane_config, host_config, output_dir)

    # --- vs-baseline: every score is read relative to the lane BASELINE (top of the
    # lane scoreboard). Written AFTER the scoreboard step so scores.csv exists.
    # Warn-only: a missing baseline/scores must not fail an otherwise-good eval. ---
    if getattr(args, "vs_baseline", False):
        from eval.baseline import write_vs_baseline
        try:
            write_vs_baseline(output_dir, lane_name, run_label=getattr(args, "run_label", ""))
        except SystemExit as exc:
            LOG.warning("--vs-baseline skipped: %s", exc)


if __name__ == "__main__":
    main()
