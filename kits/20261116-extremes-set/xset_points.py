#!/usr/bin/env python3
"""Evaluation points of each start date of the extremes set XSET-1 (16 Nov 2026).

The model is drawn on the whole globe but only these points are written (docs note s4.2). For a start date the O1280
points (indices in the order of the global out_hres store, --gout) are the union of
  1  the nearest point of every station reporting 2t, 10ff, msl or 12 h tp at one of the six valid times (station cache);
  2  every point within TP_DISC_KM (45 km) of a tp station (obs_points' 18 and 36 km disc means need the full disc);
  4  every point within TC_KM (850 km) of each best-track fix of the date's cases (300 km centre search + 500 km profile);
  8  every point within RAIN_KM (700 km) of each heavy-rain event centre of the date.
The native O320 points (order of the global in_lres store, --gin) are the nearest native point of every evaluation point
and of every station (the input row is read at the nearest native point, rule 41).

    python3 xset_points.py --cache CACHE --tc-cases tc_cases.csv --rain-events rain_events.csv --gout STORE --gin STORE \
        --out DIR --date 20240804 [--date ...]

Writes DIR/points_<date>.npz: idx (sorted), why (bit flags above, per point), lres_idx (sorted), counts. Atos or Jupiter
(numpy, scipy, zarr; pandas for the cache).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from xset_common import LEADS, chord, unit_xyz

TP_DISC_KM, TC_KM, RAIN_KM = 45.0, 850.0, 700.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tc-cases", required=True)
    ap.add_argument("--rain-events", default="")
    ap.add_argument("--gout", required=True)
    ap.add_argument("--gin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", action="append", required=True)
    a = ap.parse_args()
    import pandas as pd
    import zarr
    from scipy.spatial import cKDTree

    go, gi = zarr.open_group(a.gout, mode="r"), zarr.open_group(a.gin, mode="r")
    hlat, hlon = np.asarray(go["latitudes"][:], np.float64), np.asarray(go["longitudes"][:], np.float64)
    llat, llon = np.asarray(gi["latitudes"][:], np.float64), np.asarray(gi["longitudes"][:], np.float64)
    hxyz = unit_xyz(hlat, hlon)
    htree, ltree = cKDTree(hxyz), cKDTree(unit_xyz(llat, llon))
    print(f"grids: O1280 {hlat.size} points, O320 {llat.size} points", flush=True)
    obs = {}
    for var in ("2t", "10ff", "msl", "tp"):
        p = Path(a.cache) / f"obs_{var}.csv"
        df = pd.read_csv(p, usecols=["valid", "lat", "lon"], dtype={"valid": str})
        obs[var] = df
        print(f"cache {p.name}: {len(df)} rows", flush=True)
    cases = list(csv.DictReader(open(a.tc_cases)))
    events = list(csv.DictReader(open(a.rain_events))) if a.rain_events else []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in a.date:
        day = dt.datetime.strptime(d, "%Y%m%d")
        valids = {(day + dt.timedelta(hours=L)).strftime("%Y%m%d%H") for L in LEADS}
        why = np.zeros(hlat.size, np.uint8)
        st_xyz = []
        n_st = {}
        for var, df in obs.items():
            s = df[df.valid.isin(valids)][["lat", "lon"]].drop_duplicates().to_numpy(np.float64)
            n_st[var] = int(len(s))
            if not len(s):
                continue
            xyz = unit_xyz(s[:, 0], s[:, 1])
            st_xyz.append(xyz)
            why[htree.query(xyz, k=1)[1]] |= 1
            if var == "tp":
                for hit in htree.query_ball_point(xyz, r=float(chord(TP_DISC_KM))):
                    why[hit] |= 2
        if not n_st.get("tp") or not n_st.get("2t"):
            raise SystemExit(f"{d}: the station cache has no 2t or tp reports at the valid times of this date ({n_st})")
        fixes = [(float(c["lat"]), float(c["lon"])) for c in cases if c["date"] == d]
        for la, lo in fixes:
            why[htree.query_ball_point(unit_xyz([la], [lo])[0], r=float(chord(TC_KM)))] |= 4
        evs = [(float(e["lat"]), float(e["lon"])) for e in events if e["date"] == d]
        for la, lo in evs:
            why[htree.query_ball_point(unit_xyz([la], [lo])[0], r=float(chord(RAIN_KM)))] |= 8
        idx = np.flatnonzero(why).astype(np.int32)
        lres = np.zeros(llat.size, bool)
        lres[ltree.query(hxyz[idx], k=1)[1]] = True
        lres[ltree.query(np.concatenate(st_xyz), k=1)[1]] = True
        lres_idx = np.flatnonzero(lres).astype(np.int32)
        counts = dict(points=int(idx.size), lres_points=int(lres_idx.size), stations=n_st, tc_fixes=len(fixes),
                      rain_events=len(evs), **{f"flag{b}": int(((why[idx] & b) > 0).sum()) for b in (1, 2, 4, 8)})
        np.savez(out / f"points_{d}.npz", idx=idx, why=why[idx], lres_idx=lres_idx, counts=json.dumps(counts),
                 date=d, tp_disc_km=TP_DISC_KM, tc_km=TC_KM, rain_km=RAIN_KM)
        print(f"POINTS {d}: {json.dumps(counts)}", flush=True)
    print("POINTS_DONE", flush=True)


if __name__ == "__main__":
    main()
