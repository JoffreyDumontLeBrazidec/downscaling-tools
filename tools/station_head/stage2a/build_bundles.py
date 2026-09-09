"""Build the stage 2a input bundles from the extracted AIFS GRIB files.

One bundle is one (initialisation, lead, member). The bundle carries the AIFS
O320 forecast for that member, the O1280 land-sea mask and orography, and the
coordinates of both grids; it carries no truth field, because the station head is
scored against station reports rather than against the analysis grid, and leaving
the truth out keeps a bundle near half a gigabyte instead of two gigabytes.

The bundles of one initialisation go in their own directory. That is deliberate:
the filename template of the AIFS lane hard-codes the string time0000, and our
calendar has both a 00 UTC and a 12 UTC initialisation on the same date, so two
initialisations of one date would otherwise write the same filename. A directory
per initialisation removes the collision without changing the lane.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

CASES = "/home/ecm5702/agent-work/20260909-station-head-adapter/notes/cases_firstcut.csv"
GRIB_ROOT = Path("/home/ecm5702/perm/station-head-adapter/grib_cache_firstcut")
BUNDLE_ROOT = Path("/home/ecm5702/scratch/eval/station_head_2a/bundles")
HRES_STATIC = GRIB_ROOT / "static" / "o1280_an_lsm_z.grib"
TPL = "aifs_o320_0001_date{date}_time0000_mem{member:02d}_step{step:03d}h_input_bundle.nc"


def build_one(date: str, time: str, step: int, member: int) -> Path:
    src = GRIB_ROOT / (date + time)
    sfc = src / ("aifs_o320_0001_date" + date + "_time" + time + "_sfc.grib")
    pl = src / ("aifs_o320_0001_date" + date + "_time" + time + "_pl.grib")
    for p in (sfc, pl, HRES_STATIC):
        if not p.exists():
            raise FileNotFoundError(str(p))
    out_dir = BUNDLE_ROOT / (date + time)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / TPL.format(date=date, member=member, step=step)
    if out.exists() and out.stat().st_size > 0:
        print("skip (exists): " + out.name, flush=True)
        return out
    tmp = out.with_name("." + out.name + ".tmp")
    cmd = [
        sys.executable, "-m", "manual_inference.prediction.predict", "build-bundle",
        "--lres-sfc-grib", str(sfc),
        "--lres-pl-grib", str(pl),
        "--hres-grib", str(HRES_STATIC),
        "--hres-static-grib", str(HRES_STATIC),
        "--allow-missing-target-unsafe",
        "--step-hours", str(step),
        "--member", str(member),
        "--out", str(tmp),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    tmp.replace(out)
    print("built: %s %d bytes" % (out, out.stat().st_size), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--nslices", type=int, default=1)
    ap.add_argument("--only-case", default=None, help="case_id, e.g. 2026051212_step006")
    ap.add_argument("--only-members", default=None, help="comma separated member list")
    ap.add_argument("--split", default=None, help="training or validation")
    args = ap.parse_args()

    cases = pd.read_csv(CASES)
    if args.split:
        cases = cases[cases["split"] == args.split]
    if args.only_case:
        cases = cases[cases["case_id"] == args.only_case]
        if not len(cases):
            raise SystemExit("no such case: " + args.only_case)
    rows = list(cases.itertuples())
    mine = rows[args.slice :: args.nslices]
    print("task %d/%d: %d case(s)" % (args.slice, args.nslices, len(mine)), flush=True)

    failures = []
    for r in mine:
        members = ([int(m) for m in args.only_members.split(",")]
                   if args.only_members else list(range(1, int(r.n_members) + 1)))
        for m in members:
            try:
                build_one(str(r.date), "%02d" % int(r.time), int(r.lead_h), m)
            except Exception as exc:
                print("FAILED %s member %d: %s" % (r.case_id, m, exc), flush=True)
                failures.append((r.case_id, m, str(exc)))
    print("done: %d failure(s)" % len(failures), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
