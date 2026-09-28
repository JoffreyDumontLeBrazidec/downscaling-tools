"""``eval.cli tccompare``: compare tctracker track sets and render the TC track figures.

Diagnostic panel only. TC verdicts stay with the box-based raw-extremes ``tc``
evaluator. Runbook: docs/epics/completed_epics/tc_track/TCTRACKER_EVAL_CLI.md
(month-scale section) in the project docs on hpc-login.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.cli._common import Command, add_common_args

SUMMARY = "Compare tctracker track sets and render the TC track figures."
DESCRIPTION = (
    "Compare the track sets produced by `eval.cli tctracker` and render the month-scale "
    "TC figure suite (track maps, density against the target, intensity log-PDF and ratio, "
    "counts, step intensity, case panels) plus tc_tracks_metrics.json. Pin --dates to the "
    "intersection of complete dates when the sources have unequal coverage."
)


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("tccompare", help=SUMMARY, description=DESCRIPTION)
    add_common_args(p)
    p.add_argument(
        "--sources", required=True,
        help="Comma-separated role=value specs. A value is an rd expver (model=j9f3), a reference "
             "class:stream:expver (target=od:enfo:0001), an absolute run-root path, or a bare role "
             "name (target, input) resolved from the lane defaults.",
    )
    p.add_argument("--months", required=True, help="Comma-separated YYYYMM months in scope.")
    p.add_argument(
        "--dates", default=None,
        help=(
            "Restrict ALL sources to these init dates (comma-separated YYYYMMDD). Use it for "
            "paired-window comparisons when the sources cover different dates, because "
            "different weather in scope would confound the distributions."
        ),
    )
    p.add_argument("--basins", default="atl", help="Comma-separated basins (default: atl).")
    p.add_argument("--label", default=None, help="Campaign label used in the output directory name (default: the months joined).")
    p.add_argument("--out", default=None, help="Output directory (default: <scratch>/eval/<lane_short>/tctracks/<label>).")
    p.add_argument("--reparse", action="store_true", default=False, help="Re-parse the source tars even if parsed tables exist.")
    p.add_argument("--no-plots", action="store_true", default=False, help="Compute the metrics only; skip the figures.")
    p.add_argument("--top-k-cases", type=int, default=3, help="Deepest-target case pages per case basin (default: 3).")
    p.add_argument("--case-basins", default="atl", help="Comma-separated basins that get per-storm case pages (default: atl).")
    p.add_argument("--plot-only", action="store_true", default=False, help="Re-render the report from cached parsed tables and the existing tc_tracks_metrics.json, without re-scoring.")
    p.add_argument("--per-month-pages", action="store_true", default=False, help="Also render one focus-basin statistics page per month (default: pooled pages only).")
    return p


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


def run(args: argparse.Namespace, session) -> None:
    cmd_tccompare(args, session.lane_config, session.host_config, session.output_dir)


COMMANDS = (Command("tccompare", "tc_tracks", SUMMARY, register, run, needs_lane=True),)
