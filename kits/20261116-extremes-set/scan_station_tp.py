#!/usr/bin/env python3
"""Scan of the STVL 12 h station precipitation, to choose heavy-rain dates on observations (16 Nov 2026).

Atos only (STVL through vtb), reads only. For every 12 h window ending at 00 and 12 UTC between --start and
--end (inclusive, YYYYMMDD) it reads the global station tp with the same call, hard limits (0-300 mm) and
one-report-per-station rule as ``eval.cli obs fetch`` (``obs_at``), and writes

  <out>/counts_<start>_<end>.csv   valid (YYYYMMDDHH = end of the window), n (reports), n_ge10, n_ge20, n_ge50,
                                   n_ge100, max_mm, and zone_<z>_n / _ge20 / _ge50 / _ge100 per zone z:
                                   t = 30 S-30 N, n = north of 30 N, s = south of 30 S
  <out>/exceed_<start>_<end>.csv   every report of at least --keep-mm (default 20): valid, lat, lon, stnid, obs
  <out>/scan_<start>_<end>.done    written last

Run with the module python3 and the downscaling-tools checkout on PYTHONPATH:

    module load python3 vtb ecmwf-toolbox; export TMPDIR=<own folder>
    PYTHONPATH=<downscaling-tools> python3 scan_station_tp.py --start 20240101 --end 20240331 --out <dir>

A window that fails (STVL error) is logged as ERROR and retried once; the scan never stops on one window.
"""
import argparse
import datetime as dt
import time
from pathlib import Path


def zone(lat):
    return "t" if abs(lat) <= 30.0 else ("n" if lat > 30.0 else "s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-mm", type=float, default=20.0)
    a = ap.parse_args()
    from eval.evaluators.obs_points.core.fetch import obs_at, require_atos

    require_atos("scan_station_tp")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.start}_{a.end}"
    box = [-90.0, 90.0, -180.0, 180.0]
    t = dt.datetime.strptime(a.start, "%Y%m%d")
    end = dt.datetime.strptime(a.end, "%Y%m%d") + dt.timedelta(hours=12)
    thr = (10.0, 20.0, 50.0, 100.0)
    head = ["valid", "n", "n_ge10", "n_ge20", "n_ge50", "n_ge100", "max_mm"]
    for z in "tns":
        head += [f"zone_{z}_n", f"zone_{z}_ge20", f"zone_{z}_ge50", f"zone_{z}_ge100"]
    with open(out / f"counts_{tag}.csv", "w") as fc, open(out / f"exceed_{tag}.csv", "w") as fe:
        fc.write(",".join(head) + "\n")
        fe.write("valid,lat,lon,stnid,obs\n")
        while t <= end:
            df, err = None, None
            for attempt in (1, 2):
                try:
                    df = obs_at("tp", t, box)
                    break
                except Exception as exc:  # noqa: BLE001 - one bad window must not stop a two-year scan
                    err = exc
                    time.sleep(20)
            v = t.strftime("%Y%m%d%H")
            if df is None:
                print(f"ERROR {v}: {err!r}", flush=True)
                fc.write(v + "," + ",".join(["nan"] * (len(head) - 1)) + "\n")
            elif len(df) == 0:
                print(f"EMPTY {v}", flush=True)
                fc.write(v + "," + ",".join(["0"] * (len(head) - 1)) + "\n")
            else:
                o = df.obs.to_numpy(float)
                row = [v, len(df)] + [int((o >= x).sum()) for x in thr] + [f"{o.max():.1f}"]
                zz = df.lat.map(zone)
                for z in "tns":
                    oz = o[(zz == z).to_numpy()]
                    row += [len(oz), int((oz >= 20).sum()), int((oz >= 50).sum()), int((oz >= 100).sum())]
                fc.write(",".join(str(x) for x in row) + "\n")
                big = df[df.obs >= a.keep_mm]
                for r in big.itertuples(index=False):
                    fe.write(f"{r.valid},{r.lat:.4f},{r.lon:.4f},{r.stnid},{r.obs:.1f}\n")
                print(f"OK {v}: {len(df)} reports, {int((o >= 50).sum())} at 50 mm or more, max {o.max():.1f}",
                      flush=True)
            fc.flush()
            fe.flush()
            t += dt.timedelta(hours=12)
    (out / f"scan_{tag}.done").write_text(dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ\n"))
    print("SCAN_DONE", tag, flush=True)


if __name__ == "__main__":
    main()
