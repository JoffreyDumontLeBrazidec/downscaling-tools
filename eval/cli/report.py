"""``eval.cli report``: render the HTML report of an evaluation run."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.cli._common import LOG, Command

SUMMARY = "Generate the HTML report of an evaluation run."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("report", help=SUMMARY, description=SUMMARY)
    p.add_argument("--run-dir", required=True, help="Root directory of the evaluation run.")
    p.add_argument("--output", default=None, help="Output HTML path (default: <run-dir>/report.html).")
    return p


def run(args: argparse.Namespace) -> None:
    from eval.report import generate_report
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else None
    report_path = generate_report(run_dir, output)
    LOG.info("Report written to %s", report_path)


COMMANDS = (Command("report", "pipeline", SUMMARY, register, run),)
