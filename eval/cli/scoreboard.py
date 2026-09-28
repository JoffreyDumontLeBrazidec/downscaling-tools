"""``eval.cli scoreboard``: turn evaluator results into ``scores.csv`` and ``scores.md``."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.cli._common import LOG, Command, add_common_args
from eval.cli._selection import add_evaluator_filter_args

SUMMARY = "Build the scoreboard from existing evaluation results."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("scoreboard", help=SUMMARY, description=SUMMARY)
    add_common_args(p)
    p.add_argument(
        "--eval-dir", required=True,
        help="Root evaluation directory that contains the evaluator outputs.",
    )
    add_evaluator_filter_args(p)
    p.add_argument(
        "--vs-baseline", action="store_true", default=False,
        help="Also diff the scores against the lane BASELINE (top of the lane scoreboard) "
             "and write scoreboard/vs_baseline.md.",
    )
    return p


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


def run(args: argparse.Namespace, session) -> None:
    _run_scoreboard(Path(args.eval_dir), session.lane_config, session.evaluators, session.output_dir)


COMMANDS = (Command("scoreboard", "pipeline", SUMMARY, register, run, needs_lane=True),)
