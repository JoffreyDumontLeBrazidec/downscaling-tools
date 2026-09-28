"""Gather every stage 2a case whose prediction file is finished and which has not
been gathered yet.

The inference of the first cut runs as two SLURM array jobs, so at any moment some
prediction files are complete, some are still being written, and some do not exist.
This driver reads the manifest that `manifest.py` writes, keeps the cases whose
prediction is finished and whose gathered parquet file is missing, and calls the
gather of `gather.py` on each of them. It never touches a prediction file that is
still being written: before gathering, it probes the file again with the very same
check the manifest uses, that is the file must exist, be larger than a floor a
truncated write could not reach, open as NetCDF, and carry at least the number of
ensemble members the case asked for.

The work is split into shards so that it can run as a SLURM array. Shard `i` of
`n` takes the cases whose position in the pending list has remainder `i` modulo
`n`, which spreads the large validation cases over the tasks rather than giving
them all to one.

Usage:
    python gather_driver.py --list
    python gather_driver.py --shard 0 --nshards 8
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gather as G  # noqa: E402
import manifest as M  # noqa: E402

STAGE1 = G.STAGE1
DEFAULT_MANIFEST = M.OUT


def pending(manifest_path: Path, include_gathered: bool = False) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    sel = df[df["prediction_done"].astype(bool)].copy()
    if not include_gathered:
        sel = sel[~sel["gathered_done"].astype(bool)]
    return sel.sort_values(["split", "init", "lead_h"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many cases")
    ap.add_argument("--out-root", default=str(G.GATHER_ROOT))
    ap.add_argument("--regather", action="store_true",
                    help="gather again even if the parquet file already exists")
    ap.add_argument("--list", action="store_true", help="print what would be done and stop")
    args = ap.parse_args()

    todo = pending(Path(args.manifest), include_gathered=args.regather)
    mine = todo.iloc[args.shard::args.nshards].reset_index(drop=True)
    print("manifest %s: %d cases with a finished prediction and no gather; shard %d of %d takes %d"
          % (args.manifest, len(todo), args.shard, args.nshards, len(mine)), flush=True)
    if args.list:
        for r in mine.itertuples():
            print("  " + r.case_id + "  " + r.split)
        return
    if not len(mine):
        print("nothing to do")
        return

    stations = pd.read_parquet(STAGE1 / "static_stations.parquet")
    obs = {}
    for p in G.OBS_PARAMETERS:
        f = STAGE1 / "stations_2026" / (p + ".parquet")
        if f.exists():
            obs[p] = pd.read_parquet(f, columns=["synthetic_index", "stnid", "value"])
    print("static table %d stations; observation tables %s" % (len(stations), sorted(obs)), flush=True)

    done, skipped, failed = 0, 0, 0
    for r in mine.itertuples():
        if args.limit and done >= args.limit:
            break
        pred = Path(r.prediction_path)
        ok, size, why = M.probe(pred, int(r.n_members_wanted))
        if not ok:
            print("SKIP %s: %s (%d bytes)" % (r.case_id, why, size), flush=True)
            skipped += 1
            continue
        out = Path(args.out_root) / (r.case_id + ".parquet")
        if out.exists() and not args.regather:
            print("SKIP %s: already gathered" % r.case_id, flush=True)
            skipped += 1
            continue
        row = pd.Series({"synthetic_index": int(r.synthetic_index), "init": r.init,
                         "lead_h": int(r.lead_h), "valid": r.valid})
        t0 = time.time()
        try:
            G.gather_case(pred, out, stations, row, obs)
            print("OK %s in %.1f s" % (r.case_id, time.time() - t0), flush=True)
            done += 1
        except Exception:
            traceback.print_exc()
            print("FAIL %s" % r.case_id, flush=True)
            failed += 1
    print("shard %d finished: gathered=%d skipped=%d failed=%d" % (args.shard, done, skipped, failed),
          flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
