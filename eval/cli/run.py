"""``eval.cli run``: predict, evaluate and build the scoreboard in one command."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.cli._common import (
    LOG, Command, _update_effective_config_completion, add_common_args, add_lane_override_args,
    add_prepare_args, add_prepml_args,
)
from eval.cli._selection import add_evaluator_filter_args
from eval.cli.evaluate import _consolidate_plots, _run_evaluators
from eval.cli.predict import cmd_predict
from eval.cli.scoreboard import _run_scoreboard

SUMMARY = "Predict, evaluate and build the scoreboard in one command."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("run", help=SUMMARY, description=SUMMARY)
    add_common_args(p)
    p.add_argument("--checkpoint", required=True, help="Path to the model checkpoint.")
    add_evaluator_filter_args(p)
    add_lane_override_args(p)
    add_prepare_args(p)
    add_prepml_args(p)
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <scratch>/eval/<lane>/run_<TS>).",
    )
    p.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Allow re-running over existing evaluator outputs.",
    )
    p.add_argument(
        "--vs-baseline", action="store_true", default=False,
        help="After the scoreboard step, diff this run against the lane BASELINE "
             "(top of the lane scoreboard) and write scoreboard/vs_baseline.md.",
    )
    return p


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


def run(args: argparse.Namespace, session) -> None:
    cmd_run(args, session.lane_config, session.host_config, session.evaluators, session.output_dir)


COMMANDS = (Command("run", "pipeline", SUMMARY, register, run, needs_lane=True),)
