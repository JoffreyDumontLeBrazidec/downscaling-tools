"""``eval.cli obs``: observation references for the observation-based evaluators.

    obs fetch     Atos only. Station observations from STVL (2t, 10ff, msl, 12 h and 24 h tp) with the
                  model orography heights and HRES at the stations, for the obs_points evaluator. Run it
                  with the module python3 (``module load python3 vtb ecmwf-toolbox``), not the eval venv.
    obs compare   Any cluster. Paired comparison of two obs_points results (run A minus run B), date by
                  date: mean difference, SE over dates and date wins, as paired_runs.csv, paired_runs.md
                  and figures.

The caches are small directories of CSV files plus a cache_meta.json manifest; copy them to the
cluster that holds the predictions and point the lane block at them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.cli._common import Command

SUMMARY = "Fetch observation caches (Atos) and compare two runs against observations."


def register(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser("obs", help=SUMMARY, description=__doc__.split("\n\n")[1],
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="obs_cmd", required=True, metavar="<obs command>")

    from eval.evaluators.obs_points.core import fetch as station_fetch
    f = sub.add_parser("fetch", help="Station observation cache from STVL (Atos only).",
                       description=station_fetch.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    station_fetch.add_args(f)

    c = sub.add_parser("compare", help="Paired comparison of two obs_points results (A minus B).")
    c.add_argument("--a", required=True, help="obs_points results directory of run A (holds scores_per_date.csv).")
    c.add_argument("--b", required=True, help="obs_points results directory of run B.")
    c.add_argument("--label-a", default=None, help="Display label of run A (default: its run_meta.json run_label).")
    c.add_argument("--label-b", default=None, help="Display label of run B.")
    c.add_argument("--row", default="model", help="Row compared (default model; input, truth or hres also work).")
    c.add_argument("--out", required=True, help="Output directory (created).")
    c.add_argument("--plot-only", action="store_true", help="Redraw the figures from an existing paired_runs.csv.")
    return p


def _compare(args) -> None:
    import pandas as pd

    from eval.evaluators.obs_points.core.summary import compare_runs, write_compare_md
    from eval.evaluators.obs_points.plotter import plot_compare

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        a_dir, b_dir = Path(args.a), Path(args.b)
        meta_a = json.loads((a_dir / "run_meta.json").read_text()) if (a_dir / "run_meta.json").exists() else {}
        meta_b = json.loads((b_dir / "run_meta.json").read_text()) if (b_dir / "run_meta.json").exists() else {}
        la = args.label_a or meta_a.get("run_label") or a_dir.parent.parent.name
        lb = args.label_b or meta_b.get("run_label") or b_dir.parent.parent.name
        # round_trip: the exact floats run() wrote, so that a tie between two runs stays a tie
        read = lambda d: pd.read_csv(d / "scores_per_date.csv", dtype={"date": str}, float_precision="round_trip")  # noqa: E731
        tab, checks = compare_runs(read(a_dir), read(b_dir), row=args.row)
        tab.to_csv(out / "paired_runs.csv", index=False)
        meta = {"a": str(a_dir), "b": str(b_dir), "label_a": la, "label_b": lb, "row": args.row, "checks": checks,
                "members": [meta_a.get("members"), meta_b.get("members")]}
        (out / "compare_meta.json").write_text(json.dumps(meta, indent=2))
        write_compare_md(tab, meta, out / "paired_runs.md")
        for k, v in checks.items():
            print(f"check {k}: {v}")
    meta = json.loads((out / "compare_meta.json").read_text())
    written = plot_compare(out, pd.read_csv(out / "paired_runs.csv"), meta)
    print(f"wrote {out / 'paired_runs.csv'} and {len(written)} figure files in {out}")


def run(args: argparse.Namespace) -> None:
    if args.obs_cmd == "fetch":
        from eval.evaluators.obs_points.core.fetch import run_from_args
        out = run_from_args(args)
        print(f"station observation cache written: {out}")
    elif args.obs_cmd == "compare":
        _compare(args)


COMMANDS = (Command("obs", "observations", SUMMARY, register, run),)
