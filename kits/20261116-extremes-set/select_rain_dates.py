#!/usr/bin/env python3
"""Heavy-rain start dates of the extremes set XSET-1 from the station scan (rule: docs note
epics/fine-scale-o320-o1280/in-progress/20261116_extremes_cyclone_eval_set.md s2.3, written before the scan was read).

Input: the exceed_*.csv files of scan_station_tp.py (valid, lat, lon, stnid, obs; reports of 20 mm per 12 h or more).
An event = EVENT_MIN or more stations with THR mm or more in one 12 h window, linked within LINK_KM of each other
(single linkage). Its zone is that of its mean position: n (north of 30 N), t (30 S-30 N), s (south of 30 S).
A start date d (00 UTC) owns the six windows ending d+12..d+72 h; its value = number of events, ties by stations in events.
Greedy choice: SPACING days or more between chosen dates and from the cyclone dates (--tc-dates), at most PER_MONTH
dates per calendar month; first the best MIN_S dates whose main zone is s, then the best MIN_T whose main zone is t,
then the best MIN_W dates of December to February whose main zone is n (added 16 Nov 11:10 UTC after the first
selection on the real scan held no date between December and February; observations only, no forecast was read;
--min-w 0 gives the first selection), then the best overall up to N_DATES. Cyclone dates that hold an event are listed with tag "both" and are not counted
in N_DATES.

    python3 select_rain_dates.py --scan DIR --tc-dates tc_dates.csv --out DIR

Writes rain_dates.csv, rain_events.csv (every event of every chosen or cyclone date), rain_selection.json.
Standard library + numpy. Deterministic.
"""
import argparse
import csv
import datetime as dt
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LEADS = (12, 24, 36, 48, 60, 72)
P = dict(first="20240101", last="20250927", thr=50.0, event_min=5, link_km=300.0, spacing_days=5, tc_spacing_days=3,
         per_month=2, min_s=3, min_t=6, min_w=3, n_dates=20)


def zone(lat):
    return "t" if abs(lat) <= 30.0 else ("n" if lat > 30.0 else "s")


def events_of(lat, lon, obs, link_km, event_min):
    """Single-linkage clusters of the stations; returns [(n, mean lat, mean lon, max obs, zone)] of clusters of event_min or more."""
    n = lat.size
    if n < event_min:
        return []
    la, lo = np.radians(lat), np.radians(lon)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    chord = 2.0 * np.sin(link_km / 6371.0 / 2.0)
    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1) if n <= 4000 else None
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        near = np.flatnonzero(d2[i] <= chord * chord) if d2 is not None else \
            np.flatnonzero(((xyz - xyz[i]) ** 2).sum(-1) <= chord * chord)
        for j in near:
            if j > i:
                a, b = find(i), find(int(j))
                if a != b:
                    parent[b] = a
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for g in groups.values():
        if len(g) >= event_min:
            c = xyz[g].mean(0)
            c /= np.linalg.norm(c)
            clat, clon = float(np.degrees(np.arcsin(c[2]))), float(np.degrees(np.arctan2(c[1], c[0])))
            out.append((len(g), round(clat, 2), round(clon, 2), float(obs[g].max()), zone(clat)))
    return sorted(out, reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--tc-dates", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocked", default="")
    for k, v in P.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    p = {k: getattr(a, k) for k in P}
    rows = defaultdict(list)
    seen = set()
    files = sorted(glob.glob(str(Path(a.scan) / "exceed_*.csv*")))
    for fn in files:
        import gzip
        op = gzip.open if fn.endswith(".gz") else open
        with op(fn, "rt", newline="") as f:
            for r in csv.DictReader(f):
                o = float(r["obs"])
                if o < p["thr"]:
                    continue
                key = (r["valid"], r["stnid"], r["lat"], r["lon"])
                if key in seen:
                    continue
                seen.add(key)
                rows[r["valid"]].append((float(r["lat"]), float(r["lon"]), o))
    ev = {}
    for valid, rr in rows.items():
        x = np.asarray(rr, np.float64)
        e = events_of(x[:, 0], x[:, 1], x[:, 2], p["link_km"], p["event_min"])
        if e:
            ev[valid] = e
    tc = []
    if a.tc_dates:
        with open(a.tc_dates) as f:
            tc = [dt.datetime.strptime(r["date"], "%Y%m%d") for r in csv.DictReader(f)]
    blocked = {x for x in a.blocked.split(",") if x}

    def date_events(day):
        out = []
        for L in LEADS:
            v = (day + dt.timedelta(hours=L)).strftime("%Y%m%d%H")
            out += [(L, v) + e for e in ev.get(v, [])]
        return out

    d0, d1 = (dt.datetime.strptime(p[k], "%Y%m%d") for k in ("first", "last"))
    cand = {}
    d = d0
    while d <= d1:
        e = date_events(d)
        if e and d.strftime("%Y%m%d") not in blocked and all(abs((d - t).days) >= p["tc_spacing_days"] for t in tc):
            st = defaultdict(int)
            for x in e:
                st[x[6]] += x[2]
            cand[d] = dict(n_events=len(e), n_stations=sum(x[2] for x in e), main=max(st, key=lambda z: (st[z], z)),
                           zones={z: sum(1 for x in e if x[6] == z) for z in "nts"})
        d += dt.timedelta(days=1)
    chosen = []

    def free(day):
        if any(abs((day - c).days) < p["spacing_days"] for c in chosen):
            return False
        return sum(1 for c in chosen if (c.year, c.month) == (day.year, day.month)) < p["per_month"]

    def pick(n, main=None, months=None):
        got = 0
        while got < n:
            best, key = None, None
            for day, c in cand.items():
                if day in chosen or not free(day) or (main and c["main"] != main) or (months and day.month not in months):
                    continue
                k = (c["n_events"], c["n_stations"], -day.toordinal())
                if key is None or k > key:
                    best, key = day, k
            if best is None:
                break
            chosen.append(best)
            got += 1
        return got

    got_s = pick(p["min_s"], "s")
    got_t = pick(p["min_t"], "t")
    got_w = pick(p["min_w"], "n", months=(12, 1, 2))
    pick(p["n_dates"] - len(chosen))
    chosen.sort()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    both = [t for t in tc if date_events(t)]
    with open(out / "rain_dates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "tag", "n_events", "n_event_stations", "main_zone", "events_n", "events_t", "events_s"])
        for day in sorted(set(chosen) | set(both)):
            e = date_events(day)
            st = defaultdict(int)
            for x in e:
                st[x[6]] += x[2]
            w.writerow([day.strftime("%Y%m%d"), "both" if day in both else "RAIN", len(e), sum(x[2] for x in e),
                        max(st, key=lambda z: (st[z], z))] + [sum(1 for x in e if x[6] == z) for z in "nts"])
    with open(out / "rain_events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "lead", "valid", "n_stations", "lat", "lon", "max_mm", "zone"])
        for day in sorted(set(chosen) | set(tc)):
            for x in date_events(day):
                w.writerow([day.strftime("%Y%m%d")] + list(x))
    months = defaultdict(int)
    for c in chosen:
        months[c.strftime("%Y%m")] += 1
    summary = dict(parameters=p, scan_files=[Path(x).name for x in files], windows_with_reports=len(rows),
                   windows_with_events=len(ev), candidate_dates=len(cand), rain_dates=len(chosen), south_first=got_s,
                   tropics_first=got_t, north_winter_first=got_w, main_zone={z: sum(1 for c in chosen if cand[c]["main"] == z) for z in "nts"},
                   per_month=dict(sorted(months.items())), tc_dates_with_events=len(both), tc_dates=len(tc),
                   events_on_rain_dates=sum(cand[c]["n_events"] for c in chosen),
                   event_stations_on_rain_dates=sum(cand[c]["n_stations"] for c in chosen))
    (out / "rain_selection.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
