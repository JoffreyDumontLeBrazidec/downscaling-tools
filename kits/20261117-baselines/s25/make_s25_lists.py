#!/usr/bin/env python3
"""Lists of the September 2025 development set S25 (17 Nov 2026, thread "Score classical, regression and CorrDiff
baselines"): the 30 start dates of the official prepml month jdyb (2025-09-01..30, 00 UTC), no selection, read with the
XSET-1 kit (leads 12..72 h, 10 members, one draw per input member, rule-31 seeds), so every row (GL512 450k, baselines,
FS-242 freeze, MeanFlow) is scored on the same files.

Cyclone cases: every IBTrACS fix at d+12..d+72 h that counts under XSET-1's rule (select_tc_dates.counts_for: >= 34 kt,
tropical nature, |lat| <= 45, > 12 deg from the date line), for storms with >= 3 counting fixes on that date (XSET-1's
min_fixes). Heavy-rain events: XSET-1's event rule (select_rain_dates.events_of: >= 5 stations with >= 50 mm / 12 h,
linked within 300 km) on the station scan of Jul-Sep 2025. Tag: TC when the date holds a cyclone case, RAIN when it holds
a rain event, both, or NONE. All 49r1 (cycle change 11 Nov 2024).

    python3 make_s25_lists.py --ibtracs ibtracs_2024_2025_extract.csv.gz --scan SCANDIR --out lists
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "20261116-extremes-set"))
from select_rain_dates import P as RP, events_of  # noqa: E402
from select_tc_dates import LEADS, P as TP, counts_for, read_fixes  # noqa: E402

DATES = [(dt.datetime(2025, 9, 1) + dt.timedelta(days=i)) for i in range(30)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fixes, meta, _ = read_fixes(a.ibtracs)
    cases, storms = [], defaultdict(list)
    for day in DATES:
        for sid in sorted(fixes):
            fx = fixes[sid]
            leads = [L for L in LEADS if (f := fx.get(day + dt.timedelta(hours=L))) is not None and counts_for(f, TP)[0]]
            if len(leads) < TP["min_fixes"]:
                continue
            storms[sid].append(day.strftime("%Y%m%d"))
            for L in leads:
                t = day + dt.timedelta(hours=L)
                x = fx[t]
                cases.append([sid, meta[sid]["name"], x["basin"], day.strftime("%Y%m%d"), L, t.strftime("%Y%m%d%H"),
                              x["lat"], x["lon"], x["wind"], x["wind_src"], "" if x["pres"] is None else x["pres"],
                              x["pres_src"], "" if x["rmw"] is None else x["rmw"], "" if x["d2l"] is None else x["d2l"],
                              x["nature"], x["track_type"]])
    with open(out / "tc_cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "name", "basin", "date", "lead", "valid", "lat", "lon", "wind_kt", "wind_src", "pres_hpa",
                    "pres_src", "usa_rmw_nmi", "dist2land_km", "nature", "track_type"])
        w.writerows(cases)
    with open(out / "tc_storms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "name", "season", "basin", "peak_kt", "track_type", "n_dates", "dates"])
        for sid in sorted(storms, key=lambda s: (meta[s]["basin_at_peak"], s)):
            m = meta[sid]
            w.writerow([sid, m["name"], m["season"], m["basin_at_peak"], f"{m['peak']:.0f}", m["track_type"],
                        len(storms[sid]), ";".join(storms[sid])])
    # heavy-rain events of the 30 dates
    rows, seen = defaultdict(list), set()
    for fn in sorted(Path(a.scan).glob("exceed_2025*.csv*")):
        op = gzip.open if fn.name.endswith(".gz") else open
        with op(fn, "rt", newline="") as f:
            for r in csv.DictReader(f):
                o = float(r["obs"])
                key = (r["valid"], r["stnid"], r["lat"], r["lon"])
                if o < RP["thr"] or key in seen:
                    continue
                seen.add(key)
                rows[r["valid"]].append((float(r["lat"]), float(r["lon"]), o))
    events = []
    for day in DATES:
        for L in LEADS:
            v = (day + dt.timedelta(hours=L)).strftime("%Y%m%d%H")
            if v not in rows:
                continue
            x = np.asarray(rows[v], np.float64)
            for e in events_of(x[:, 0], x[:, 1], x[:, 2], RP["link_km"], RP["event_min"]):
                events.append([day.strftime("%Y%m%d"), L, v] + list(e))
    with open(out / "rain_events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "lead", "valid", "n_stations", "lat", "lon", "max_mm", "zone"])
        w.writerows(events)
    ntc, nev = defaultdict(int), defaultdict(int)
    for c in cases:
        ntc[c[3]] += 1
    for e in events:
        nev[e[0]] += 1
    st_per_date = defaultdict(set)
    for c in cases:
        st_per_date[c[3]].add(c[0])
    with open(out / "dates_s25.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "tag", "cycle", "n_storms", "n_tc_cases", "n_rain_events"])
        for day in DATES:
            d = day.strftime("%Y%m%d")
            tag = {(True, True): "both", (True, False): "TC", (False, True): "RAIN", (False, False): "NONE"}[(ntc[d] > 0, nev[d] > 0)]
            w.writerow([d, tag, "49r1", len(st_per_date[d]), ntc[d], nev[d]])
    summary = dict(dates=len(DATES), tc_cases=len(cases), storms=len(storms),
                   basins={b: sum(1 for s in storms if meta[s]["basin_at_peak"] == b) for b in {meta[s]["basin_at_peak"] for s in storms}},
                   rain_events=len(events), dates_with_tc=sum(1 for v in ntc.values() if v), dates_with_rain=sum(1 for v in nev.values() if v),
                   scan_windows_2025=len(rows), tc_rule=TP, rain_rule={k: RP[k] for k in ("thr", "event_min", "link_km")})
    (out / "s25_lists.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
