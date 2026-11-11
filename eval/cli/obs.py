"""``eval.cli obs``: observation references for the observation-based evaluators.

    obs fetch     Atos only. Station observations from STVL (2t, 10ff, msl, 12 h and 24 h tp) with the
                  model orography heights and HRES at the stations, for the obs_points evaluator. Run it
                  with the module python3 (``module load python3 vtb ecmwf-toolbox``), not the eval venv.
    obs fetch-tc  Atos only. Tropical cyclone references for the tc_obs evaluator: IBTrACS best-track rows of
                  the configured storms (anemoi zarr; a NOAA IBTrACS CSV too when a local path is given) and
                  the ECMWF operational tracker (HRES and ENS) for the given dates, decoded from MARS BUFR.
    obs fetch-imerg  Atos (or ac6-100) only. IMERG Final V07B 12 h precipitation on the E2 O1280 points for the
                  given dates and leads, one small netCDF file for the obs_imerg evaluator.
    obs compare   Any cluster. Paired comparison of two obs_points (or obs_imerg) results (run A minus run B), date by
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

    t = sub.add_parser("fetch-tc", help="Tropical cyclone reference cache: IBTrACS and the ECMWF tracker (Atos only).")
    t.add_argument("--storms", required=True,
                   help="Comma-separated event=IBTrACS-id pairs, event names as in tc_structure (franklin=4648,...).")
    t.add_argument("--dates", required=True, help="Comma-separated YYYYMMDD forecast dates (00 UTC) for the tracker.")
    t.add_argument("--streams", default="oper,enfo", help="Tracker streams (default oper,enfo).")
    t.add_argument("--ibtracs-zarr", default=None, help="anemoi IBTrACS zarr (default: the ai-ml dataset on /ec/ai).")
    t.add_argument("--noaa-csv", default=None, help="Optional local NOAA IBTrACS CSV (adds best-track wind). Never downloaded.")
    t.add_argument("--out", required=True, help="Cache directory (created).")

    from eval.evaluators.obs_imerg.core import fetch as imerg_fetch
    i = sub.add_parser("fetch-imerg", help="IMERG 12 h precipitation cut on the E2 points (Atos only).",
                       description=imerg_fetch.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    imerg_fetch.add_args(i)

    c = sub.add_parser("compare", help="Paired comparison of two obs_points (or obs_imerg) results (A minus B).")
    c.add_argument("--a", default=None, help="Results directory of run A (holds scores_per_date.csv); not needed "
                                             "with --plot-only.")
    c.add_argument("--b", default=None, help="Results directory of run B; not needed with --plot-only.")
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
    if args.plot_only and not (out / "paired_runs.csv").exists():
        raise SystemExit(f"obs compare --plot-only: no paired_runs.csv in {out}")
    if not args.plot_only and not (args.a and args.b):
        raise SystemExit("obs compare: --a and --b are required unless --plot-only is given")
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
                "members": [meta_a.get("members"), meta_b.get("members")],
                "evaluator": meta_a.get("evaluator", "obs_points"), "where": meta_a.get("where", "at stations"),
                "sample_unit": meta_a.get("sample_unit", "stations"),
                "tp_obs_var": meta_a.get("tp_obs_var") or "tp"}
        if (meta_a.get("tp_obs_var") or "tp") != (meta_b.get("tp_obs_var") or "tp"):
            checks["tp_obs_var_differs"] = [meta_a.get("tp_obs_var"), meta_b.get("tp_obs_var")]
        # the region shown: "all" for station runs, else the first region the run configured (obs_imerg)
        order = list(meta_a.get("regions") or meta_a.get("config", {}).get("regions") or [])
        present = set(tab.region) if len(tab) else set()
        meta["region"] = "all" if "all" in present else next((r for r in order if r in present), None)
        (out / "compare_meta.json").write_text(json.dumps(meta, indent=2))
        write_compare_md(tab, meta, out / "paired_runs.md", region=meta["region"])
        for k, v in checks.items():
            print(f"check {k}: {v}")
    meta = json.loads((out / "compare_meta.json").read_text())
    written = plot_compare(out, pd.read_csv(out / "paired_runs.csv"), meta, region=meta.get("region"))
    print(f"wrote {out / 'paired_runs.csv'} and {len(written)} figure files in {out}")


def run(args: argparse.Namespace) -> None:
    if args.obs_cmd == "fetch":
        from eval.evaluators.obs_points.core.fetch import run_from_args
        out = run_from_args(args)
        print(f"station observation cache written: {out}")
    elif args.obs_cmd == "fetch-tc":
        from eval.evaluators.obs_points.core.fetch import require_atos
        from eval.evaluators.tc_obs.core.reference import IBTRACS_ZARR, fetch_tc
        require_atos("obs fetch-tc")
        storms = dict(kv.split("=") for kv in args.storms.split(","))
        out = fetch_tc({k.strip().lower(): int(v) for k, v in storms.items()}, args.dates.split(","), args.out,
                       zarr_path=args.ibtracs_zarr or IBTRACS_ZARR, noaa_csv=args.noaa_csv,
                       streams=tuple(args.streams.split(",")))
        print(f"tropical cyclone reference cache written: {out}")
    elif args.obs_cmd == "fetch-imerg":
        from eval.evaluators.obs_imerg.core.fetch import run_from_args as imerg_run
        print(f"IMERG cut written: {imerg_run(args)}")
    elif args.obs_cmd == "compare":
        _compare(args)


COMMANDS = (Command("obs", "observations", SUMMARY, register, run),)
