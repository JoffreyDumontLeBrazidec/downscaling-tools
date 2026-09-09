"""Extract the members the station head needs out of the surviving MARS cache.

The MARS retriever cache of the AIFS ENS version 2 dataset build holds, for each
of the 223 initialisations, one surface file and one pressure-level file with all
50 members and both leads (6 and 12 hours). The station head only ever uses
members 1 to 10, so this script copies just those messages into a durable place on
perm, in the per-initialisation layout the bundle builder expects, and re-encodes
the surface file to GRIB edition 1 with a surface level, which is what the AIFS
lane of the evaluation harness expects. It also copies one O1280 land-sea-mask and
orography file, which is date-independent and serves as the high-resolution static
input of every bundle.

One task of the SLURM array handles a contiguous slice of the initialisations.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

CACHE_INDEX = "/home/ecm5702/agent-work/20260909-station-head-adapter/notes/cache_index.csv"
OUT_ROOT = Path("/home/ecm5702/perm/station-head-adapter/grib_cache_firstcut")


def load_index() -> list[dict]:
    with open(CACHE_INDEX) as fh:
        return list(csv.DictReader(fh))


def forecast_files(rows: list[dict]) -> dict[tuple[str, str], dict[str, str]]:
    """Map (date, time) to {levtype: path} for the 50-member forecast retrievals."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        if r["n_members"] != "50" or r["steps"] != "6/12":
            continue
        out.setdefault((r["date"], r["time"]), {})[r["levtype"]] = r["path"]
    return out


def static_file(rows: list[dict]) -> str:
    for r in rows:
        if r["params"] == "lsm/z":
            return r["path"]
    raise RuntimeError("no lsm/z file in the cache index")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def do_one(date: str, time: str, paths: dict[str, str], members: int) -> None:
    dest = OUT_ROOT / f"{date}{time}"
    dest.mkdir(parents=True, exist_ok=True)
    numbers = "/".join(str(m) for m in range(1, members + 1))

    sfc_out = dest / f"aifs_o320_0001_date{date}_time{time}_sfc.grib"
    if not sfc_out.exists():
        # the un-re-encoded intermediate is written to the node-local temporary
        # directory so that nothing provisional is ever created on perm
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.grib"
            run(["grib_copy", "-w", f"number={numbers}", paths["sfc"], str(raw)])
            tmp = dest / f".{sfc_out.name}.tmp"
            run(["grib_set", "-s", "edition=1,typeOfLevel=surface,level=0", str(raw), str(tmp)])
            tmp.replace(sfc_out)

    pl_out = dest / f"aifs_o320_0001_date{date}_time{time}_pl.grib"
    if not pl_out.exists():
        tmp = dest / f".{pl_out.name}.tmp"
        run(["grib_copy", "-w", f"number={numbers}", paths["pl"], str(tmp)])
        tmp.replace(pl_out)

    counts = {}
    for label, path in (("sfc", sfc_out), ("pl", pl_out)):
        res = subprocess.run(["grib_count", str(path)], check=True, capture_output=True, text=True)
        counts[label] = int(res.stdout.strip())
    (dest / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print("%s %s: sfc=%d pl=%d messages" % (date, time, counts["sfc"], counts["pl"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, required=True, help="SLURM array task index, from 0")
    ap.add_argument("--nslices", type=int, required=True)
    ap.add_argument("--members", type=int, default=10)
    args = ap.parse_args()

    rows = load_index()
    fc = forecast_files(rows)
    keys = sorted(fc)

    if args.slice == 0:
        static_dest = OUT_ROOT / "static" / "o1280_an_lsm_z.grib"
        static_dest.parent.mkdir(parents=True, exist_ok=True)
        if not static_dest.exists():
            src = static_file(rows)
            tmp = static_dest.with_suffix(".tmp")
            run(["cp", src, str(tmp)])
            tmp.replace(static_dest)
            print(f"static hres GRIB copied from {src}")

    mine = keys[args.slice :: args.nslices]
    print(f"task {args.slice}/{args.nslices}: {len(mine)} initialisations")
    for date, time in mine:
        do_one(date, time, fc[(date, time)], args.members)


if __name__ == "__main__":
    main()
