"""``eval.cli evaluate``: run evaluators on existing predictions.

Also holds ``_run_evaluators``, the loop that imports each evaluator package and
calls its ``run``, ``score`` and ``plot`` (the contract is in
``eval/evaluators/base.py``); ``eval.cli run`` reuses it after predicting.
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval import lean_layout
from eval.cli._common import (
    ALL_EVALUATORS, LOG, Command, _update_effective_config_completion, _parse_csv_or_none,
    add_common_args, add_lane_override_args, evaluator_registry,
)
from eval.cli._selection import add_evaluator_filter_args

SUMMARY = "Run evaluators on existing predictions."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("evaluate", help=SUMMARY, description=SUMMARY)
    add_common_args(p)
    p.add_argument(
        "--predictions-dir", required=True,
        help="Directory containing the prediction .nc files.",
    )
    add_evaluator_filter_args(p)
    add_lane_override_args(p)
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to the model checkpoint, for evaluators that need it (for example sigma_loss).",
    )
    p.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Allow re-running over existing evaluator outputs.",
    )
    p.add_argument(
        "--plot-only", action="store_true", default=False,
        help="Skip run() and score(); re-render plot() against the existing results. "
             "Cheap way to re-plot after a plotting fix.",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <scratch>/eval/<lane>/run_<TS>). "
             "Use it with --plot-only to target an existing run directory.",
    )
    p.add_argument(
        "--run-label", default="",
        help="Short display label for this run, used in TC and plot legends "
             "(default: derived from the predictions directory name).",
    )
    p.add_argument(
        "--expver", default=None,
        help="PrepML/FDB expver of the run being evaluated. When set, the quaver "
             "probabilistic scorecard runs automatically (FDB-based).",
    )
    p.add_argument(
        "--vs-baseline", action="store_true", default=False,
        help="After evaluating, diff this run's scores against the lane BASELINE "
             "(top of the lane scoreboard) and write scoreboard/vs_baseline.md.",
    )
    return p


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


def _host_mismatch(name: str) -> str | None:
    """Why ``name`` cannot run on this machine, or None when it can.

    Some evaluators depend on a tool that exists on one Atos cluster only (the
    registry's ``host_prefix``; for example gptosp for spectra_ecmwf_v2 is on AC
    only). On any other host the evaluator is skipped with a warning rather than
    failing the whole evaluation.
    """
    import socket

    entry = evaluator_registry.get(name)
    prefix = entry.host_prefix if entry else None
    if not prefix:
        return None
    hostname = socket.gethostname()
    if hostname.startswith(prefix):
        return None
    return (
        f"it needs a host whose name starts with '{prefix}' "
        f"(current host: {hostname}); run it on Atos {prefix.upper()} with --only {name}"
    )


def _lane_block(lane_config: dict, name: str) -> dict:
    """The lane's ``<name>:`` block, or the block under a former name of a renamed evaluator.

    Tracked lanes written before a rename keep the old block name (a canonical lane
    edit changes a certified hash), so the old block is read when the lane has none
    under the new name.
    """
    if name in lane_config:
        return lane_config.get(name) or {}
    for old_name in evaluator_registry.former_names(name):
        if old_name in lane_config:
            LOG.warning(
                "Lane block '%s:' is deprecated; read it for '%s'. Rename the block to '%s:'.",
                old_name, name, name,
            )
            return lane_config.get(old_name) or {}
    return {}


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
        if _host_mismatch(name):
            # Cannot run on this machine; skipped with a warning, not a gap.
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
    stages: list[str] | None = None,
) -> list[str]:
    """Run selected evaluators on existing predictions. Returns list of evaluators that ran."""
    evaluators_run: list[str] = []
    failures: list[str] = []

    for name in evaluators:
        if evaluator_registry.is_retired(name):
            LOG.warning("Skipping retired evaluator. %s", evaluator_registry.retired_message(name))
            continue
        if name not in ALL_EVALUATORS:
            LOG.warning(
                "Skipping unknown evaluator '%s'. Valid: %s", name, ALL_EVALUATORS
            )
            continue

        mismatch = _host_mismatch(name)
        if mismatch:
            LOG.warning("Skipping evaluator '%s': %s", name, mismatch)
            _write_evaluator_status(output_dir, name, "skipped", detail=mismatch)
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
        eval_config = dict(_lane_block(lane_config, name))
        if stages is not None:
            eval_config["stages"] = list(stages)

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


def run(args: argparse.Namespace, session) -> None:
    predictions_dir = Path(args.predictions_dir)
    evaluators_run = _run_evaluators(
        predictions_dir, session.lane_config, session.evaluators, session.output_dir,
        overwrite=getattr(args, "overwrite", False),
        plot_only=getattr(args, "plot_only", False),
        checkpoint=getattr(args, "checkpoint", None),
        run_label=getattr(args, "run_label", ""),
        stages=_parse_csv_or_none(getattr(args, "stages", None)),
    )
    # C2: record completion FIRST (always), then consolidate plots (non-fatal).
    _update_effective_config_completion(session.output_dir, evaluators_run)
    _consolidate_plots(session.output_dir)


COMMANDS = (Command("evaluate", "pipeline", SUMMARY, register, run, needs_lane=True),)
