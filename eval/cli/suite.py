"""``eval.cli suite``: the classic plot suite, the same pages every time plots are asked for.

Owner, 23 Nov 2026. Pages (``eval.suite``): 01-02 spectra +24/+72 h (target, input, model), 03 TC 10 members,
04 intense-rain 10 members, 05 input/target/model by area, 06 fair CRPS and spread at stations on one page,
07 tp stills, 08 tp PDF against input and target, 09 TC pressure and wind against input, target and IBTrACS.
Two pages of the agent's choice (10, 11) are added by hand. It reads no lane or host configuration.

Example (4 km lane, 2025 TC set)::

    python -m eval.cli suite --cases o2560_tc_2025 \\
        --run G300=.../pred/G300e2_m --run G150=.../pred/G150e2_m \\
        --obs G300=.../obs_points/G300e2_m --obs G150=.../obs_points/G150e2_m \\
        --tc G300=.../tc_structure/G300e2_m --tc G150=.../tc_structure/G150e2_m --tc-cache .../tc_cache \\
        --out .../suite_G300_vs_G150
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval.cli._common import Command

SUMMARY = "Draw the classic plot suite (the fixed pages asked for every time) of one or more runs."


def _kv(items, what) -> dict[str, Path]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"eval.cli suite: --{what} wants LABEL=DIR, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = Path(v).expanduser()
    return out


def register(subparsers) -> argparse.ArgumentParser:
    from eval.suite.cases import available_case_lists
    from eval.suite.runner import PAGES

    p = subparsers.add_parser("suite", help=SUMMARY, description=__doc__.split("\n\n")[1],
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                   help="Predictions folder (predictions_<date>_step<LLL>.nc, one draw per input member). Repeat "
                        "per run; the first run gives the input and target rows, and the model of the map pages.")
    p.add_argument("--cases", required=True,
                   help=f"Case list: a name in eval/config/suite ({', '.join(available_case_lists())}) or a path.")
    p.add_argument("--out", required=True, type=Path, help="Folder for the pages (created).")
    p.add_argument("--obs", action="append", metavar="LABEL=DIR",
                   help="obs_points results folder (summary.csv) per run, for page 06.")
    p.add_argument("--tc", action="append", metavar="LABEL=DIR",
                   help="tc_structure results folder (cases.csv) per run, for page 09.")
    p.add_argument("--tc-cache", type=Path, default=None, help="tc_obs cache (best_track.csv) for IBTrACS on page 09.")
    p.add_argument("--attach-target", type=Path, default=None, metavar="TRUTH_DIR",
                   help="Before drawing, WRITE a target y into every prediction file that has none, from "
                        "TRUTH_DIR/truth_<valid YYYYMMDDHH>.nc (y(grid_point, weather_state), same points and state "
                        "names; NaN allowed). Missing truth file = all-NaN target. See eval/suite/target.py.")
    p.add_argument("--maps-all-runs", action="store_true",
                   help="Map pages (03, 04, 05, 07) show every run; default only the first (one model per page).")
    p.add_argument("--only", nargs="+", choices=PAGES, default=None, help="Draw only these pages.")
    return p


def run(args: argparse.Namespace) -> dict:
    from eval.suite.cases import load_case_list
    from eval.suite.runner import build_suite

    if args.attach_target:
        from eval.suite.target import attach_target

        for label, d in _kv(args.run, "run").items():
            print(f"attach-target {label}: {attach_target(d, args.attach_target)}")
    record = build_suite(_kv(args.run, "run"), load_case_list(args.cases), args.out, obs=_kv(args.obs, "obs"),
                         tc=_kv(args.tc, "tc"), tc_cache=args.tc_cache, only=args.only,
                         maps_all_runs=args.maps_all_runs)
    for p in record["pages"]:
        print(f"{p['page']} {p['title']}: {'; '.join(p['files']) or '-'} | {p['note']}")
    print(f"index: {Path(args.out) / 'SUITE.md'}")
    return record


COMMANDS = (Command("suite", "figures", SUMMARY, register, run),)
