"""Choosing which evaluators a command runs (``--only``, ``--include-diagnostics``)."""
from __future__ import annotations

import argparse
import sys

from eval.cli._common import ALL_EVALUATORS, LOG, _parse_str_csv, evaluator_registry


def add_evaluator_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add --only, --include-diagnostics and --stages for evaluator selection."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", default=None,
        help="Comma-separated evaluators to run, instead of the lane's default group. "
             "Run `python -m eval.cli list` for the names.",
    )
    group.add_argument(
        "--include-diagnostics", action="store_true", default=False,
        help="Run the lane's default group plus its diagnostics group.",
    )
    parser.add_argument(
        "--stages", default=None,
        help=(
            "Comma-separated stage names passed to every selected evaluator as "
            "eval_config['stages'], overriding the lane YAML. Evaluators that work in one "
            "piece ignore it; evaluators whose measurements differ greatly in cost use it "
            "to run in separate jobs. Give concurrent stages separate --output-dir trees, "
            "because an evaluator's results directory is cleaned before a fresh run."
        ),
    )


def _with_prepml_fdb_evaluators(evaluators: list[str], args: argparse.Namespace) -> list[str]:
    """Auto-include the quaver scorecard for prepml evaluations (expver set).

    quaver reads the ensemble from FDB under an expver, so it is only meaningful
    when the run published one; we avoid even listing it for manual runs. quaver
    is the canonical probabilistic scorecard and the only one covering upper air.
    (obs_crps, its cheap surface-only counterpart, was retired on 2026-09-28 in
    favour of quaver.) Applied to the default / --include-diagnostics paths;
    --only stays explicit.
    """
    if not getattr(args, "expver", None):
        return evaluators
    out = list(evaluators)
    for name in ("quaver",):
        if name not in out:
            out.append(name)
    return out


def _drop_retired_from_lane_group(evaluators: list[str], group: str) -> list[str]:
    """Skip retired evaluators that a lane YAML group still lists, with a warning.

    Tracked lanes were repointed when the evaluators were retired; this keeps an
    old or untracked lane running instead of failing on a name that is gone.
    """
    kept: list[str] = []
    for name in evaluators:
        if evaluator_registry.is_retired(name):
            LOG.warning(
                "Lane evaluator group '%s' lists a retired evaluator; skipping it. %s",
                group, evaluator_registry.retired_message(name),
            )
            continue
        kept.append(name)
    return kept


def _resolve_evaluators(args: argparse.Namespace, lane_config: dict) -> list[str]:
    """Three-step evaluator resolution.

    1. --only: run exactly those evaluators. A retired evaluator named here is a
       tombstone: print its replacement and exit with status 1.
    2. --include-diagnostics: default + diagnostics groups.
    3. Otherwise: default group only.

    Retired evaluators found in a lane group are skipped with a warning.
    """
    evaluator_groups = lane_config.get("evaluator_groups", {})

    if getattr(args, "only", None) is not None:
        requested = _parse_str_csv(args.only)
        retired = [e for e in requested if evaluator_registry.is_retired(e)]
        if retired:
            for name in retired:
                print(f"ERROR: {evaluator_registry.retired_message(name)}", file=sys.stderr)
            raise SystemExit(1)
        unknown = [e for e in requested if e not in ALL_EVALUATORS]
        if unknown:
            raise SystemExit(
                f"Unknown evaluator(s) in --only: {unknown}. "
                f"Valid evaluators: {ALL_EVALUATORS}"
            )
        return requested

    default_group = _drop_retired_from_lane_group(
        list(evaluator_groups.get("default", [])), "default"
    )
    if getattr(args, "include_diagnostics", False):
        diag_group = _drop_retired_from_lane_group(
            list(evaluator_groups.get("diagnostics", [])), "diagnostics"
        )
        # Preserve order, avoid duplicates
        combined: list[str] = list(default_group)
        for e in diag_group:
            if e not in combined:
                combined.append(e)
        return _with_prepml_fdb_evaluators(combined, args)

    return _with_prepml_fdb_evaluators(default_group, args)
