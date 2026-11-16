#!/usr/bin/env python3
"""Cyclone start dates of the extremes set XSET-1 from the IBTrACS extract (rule: docs note
epics/fine-scale-o320-o1280/in-progress/20261116_extremes_cyclone_eval_set.md s2.2, written before the extract was read).

A storm counts for a start date d (00 UTC) when at least MIN_FIXES of the six valid times d+12..d+72 h hold a
best-track fix with wind >= 34 kt, tropical nature, |lat| <= 45 and more than 12 degrees of longitude from the date
line. Greedy choice of dates: most new storms first, SPACING days or more between chosen dates, until N_STORMS storms
and N_BASINS basins with BASIN_MIN storms each; then second dates for storms of 96 kt or more seen once (up to
MAX_DATES); then, below MIN_DATES, the dates that add the most new storms.

    python3 select_tc_dates.py ibtracs_2024_2025_extract.csv.gz --out DIR [--blocked 20240601,20240612]

Writes DIR/tc_dates.csv, tc_storms.csv, tc_cases.csv and tc_selection.json (the parameters, counts, exclusions).
Standard library only. Deterministic: ties go to the date with more cases, then the earlier date.
"""
import argparse
import csv
import datetime as dt
import gzip
import json
from collections import defaultdict
from pathlib import Path

LEADS = (12, 24, 36, 48, 60, 72)
P = dict(first="20240101", last="20251228", min_fixes=3, wind_kt=34.0, max_abs_lat=45.0, dateline_deg=12.0,
         spacing_days=3, n_storms=30, n_basins=3, basin_min=6, major_kt=96.0, min_dates=22, max_dates=26)


def num(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def read_fixes(path):
    """{sid: {valid datetime: fix dict}} for the 00 and 12 UTC fixes; spur tracks dropped."""
    op = gzip.open if str(path).endswith(".gz") else open
    out, meta, skipped = defaultdict(dict), {}, defaultdict(int)
    with op(path, "rt", newline="") as f:
        for r in csv.DictReader(f):
            tt = r.get("TRACK_TYPE", "").strip()
            if tt.lower().startswith("spur"):
                skipped["spur"] += 1
                continue
            t = dt.datetime.strptime(r["ISO_TIME"].strip()[:19], "%Y-%m-%d %H:%M:%S")
            if t.minute or t.second or t.hour not in (0, 12):
                continue
            usa, wmo = num(r["USA_WIND"]), num(r["WMO_WIND"])
            wind, src = (usa, "USA_WIND") if usa is not None else (wmo, "WMO_WIND:" + r["WMO_AGENCY"].strip())
            up, wp = num(r["USA_PRES"]), num(r["WMO_PRES"])
            pres, psrc = (up, "USA_PRES") if up is not None else (wp, "WMO_PRES:" + r["WMO_AGENCY"].strip())
            sid = r["SID"].strip()
            out[sid][t] = dict(lat=num(r["LAT"]), lon=num(r["LON"]), wind=wind, wind_src=src if wind is not None else "",
                               pres=pres, pres_src=psrc if pres is not None else "", nature=r["NATURE"].strip(),
                               basin=r["BASIN"].strip(), rmw=num(r.get("USA_RMW", "")), d2l=num(r.get("DIST2LAND", "")),
                               track_type=tt)
            m = meta.setdefault(sid, dict(sid=sid, name=r["NAME"].strip(), season=r["SEASON"].strip(), peak=0.0,
                                          basin_at_peak=r["BASIN"].strip(), track_type=tt))
            if wind is not None and wind > m["peak"]:
                m["peak"], m["basin_at_peak"] = wind, r["BASIN"].strip()
    return out, meta, dict(skipped)


def counts_for(fix, p):
    if fix["wind"] is None or fix["wind"] < p["wind_kt"] or fix["lat"] is None or fix["lon"] is None:
        return False, "weak"
    if fix["nature"] not in ("TS", "NR", "MX"):          # tropical; NR / MX = not reported / agencies differ
        return False, "not_tropical"
    if abs(fix["lat"]) > p["max_abs_lat"]:
        return False, "lat"
    lon = (fix["lon"] + 180.0) % 360.0 - 180.0
    if abs(lon) > 180.0 - p["dateline_deg"]:
        return False, "dateline"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extract")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocked", default="", help="comma list of YYYYMMDD that cannot be chosen (e.g. no MARS data)")
    for k, v in P.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    p = {k: getattr(a, k) for k in P}
    fixes, meta, skipped = read_fixes(a.extract)
    d0, d1 = (dt.datetime.strptime(p[k], "%Y%m%d") for k in ("first", "last"))
    blocked = {x for x in a.blocked.split(",") if x}
    # per date: {sid: [leads that count]}
    per_date, excl = {}, defaultdict(set)
    d = d0
    while d <= d1:
        tag = d.strftime("%Y%m%d")
        got = {}
        for sid, fx in fixes.items():
            leads = []
            for L in LEADS:
                f = fx.get(d + dt.timedelta(hours=L))
                if f is None:
                    continue
                ok, why = counts_for(f, p)
                if ok:
                    leads.append(L)
                elif why in ("dateline", "lat") and f["wind"] is not None and f["wind"] >= p["wind_kt"]:
                    excl[why].add(sid)
            if len(leads) >= p["min_fixes"]:
                got[sid] = leads
        if got and tag not in blocked:
            per_date[d] = got
        d += dt.timedelta(days=1)

    chosen, covered = [], defaultdict(list)

    def free(day):
        return all(abs((day - c).days) >= p["spacing_days"] for c in chosen)

    def basins_ok():
        n = defaultdict(int)
        for sid in covered:
            n[meta[sid]["basin_at_peak"]] += 1
        return sum(1 for v in n.values() if v >= p["basin_min"]) >= p["n_basins"], dict(n)

    def take(day):
        chosen.append(day)
        for sid in per_date[day]:
            covered[sid].append(day)

    # phase 1: new storms; while a basin quota is open, new storms of basins under quota count double
    while len(chosen) < p["max_dates"]:
        ok_b, nb = basins_ok()
        if len(covered) >= p["n_storms"] and ok_b:
            break
        best, key = None, None
        for day, got in per_date.items():
            if not free(day):
                continue
            new = [s for s in got if s not in covered]
            if not new:
                continue
            under = sum(1 for s in new if nb.get(meta[s]["basin_at_peak"], 0) < p["basin_min"])
            k = (len(new) + (under if not ok_b else 0), sum(len(v) for v in got.values()), -day.toordinal())
            if key is None or k > key:
                best, key = day, k
        if best is None:
            break
        take(best)
    phase1 = len(chosen)
    # phase 2: a second date for storms of major_kt or more seen once
    while len(chosen) < p["max_dates"]:
        need = {s for s, ds in covered.items() if meta[s]["peak"] >= p["major_kt"] and len(ds) < 2}
        best, key = None, None
        for day, got in per_date.items():
            if not free(day):
                continue
            hit = [s for s in got if s in need]
            if not hit:
                continue
            k = (len(hit), len([s for s in got if s not in covered]), sum(len(v) for v in got.values()), -day.toordinal())
            if key is None or k > key:
                best, key = day, k
        if best is None:
            break
        take(best)
    phase2 = len(chosen)
    # phase 3: up to min_dates, the dates that add the most new storms
    while len(chosen) < p["min_dates"]:
        best, key = None, None
        for day, got in per_date.items():
            if not free(day):
                continue
            new = [s for s in got if s not in covered]
            if not new:
                continue
            k = (len(new), sum(len(v) for v in got.values()), -day.toordinal())
            if key is None or k > key:
                best, key = day, k
        if best is None:
            break
        take(best)
    chosen.sort()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tc_dates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "n_storms", "n_cases", "storms"])
        for day in chosen:
            got = per_date[day]
            w.writerow([day.strftime("%Y%m%d"), len(got), sum(len(v) for v in got.values()),
                        ";".join(f"{meta[s]['name']}:{s}" for s in sorted(got))])
    with open(out / "tc_storms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "name", "season", "basin", "peak_kt", "track_type", "n_dates", "dates"])
        for sid in sorted(covered, key=lambda s: (meta[s]["basin_at_peak"], s)):
            m = meta[sid]
            w.writerow([sid, m["name"], m["season"], m["basin_at_peak"], f"{m['peak']:.0f}", m["track_type"],
                        len(covered[sid]), ";".join(x.strftime("%Y%m%d") for x in sorted(covered[sid]))])
    n_cases = 0
    with open(out / "tc_cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "name", "basin", "date", "lead", "valid", "lat", "lon", "wind_kt", "wind_src", "pres_hpa",
                    "pres_src", "usa_rmw_nmi", "dist2land_km", "nature", "track_type"])
        for day in chosen:
            for sid in sorted(per_date[day]):
                for L in per_date[day][sid]:
                    t = day + dt.timedelta(hours=L)
                    x = fixes[sid][t]
                    w.writerow([sid, meta[sid]["name"], x["basin"], day.strftime("%Y%m%d"), L, t.strftime("%Y%m%d%H"),
                                x["lat"], x["lon"], x["wind"], x["wind_src"], "" if x["pres"] is None else x["pres"],
                                x["pres_src"], "" if x["rmw"] is None else x["rmw"], "" if x["d2l"] is None else x["d2l"],
                                x["nature"], x["track_type"]])
                    n_cases += 1
    ok_b, nb = basins_ok()
    cls = defaultdict(int)
    for s in covered:
        pk = meta[s]["peak"]
        cls["lt64" if pk < 64 else ("64_95" if pk < 96 else "ge96")] += 1
    summary = dict(parameters=p, extract=str(a.extract), blocked=sorted(blocked), dates=len(chosen), dates_phase1=phase1, dates_phase2=phase2,
                   storms=len(covered), cases=n_cases, basins=nb, basin_rule_met=ok_b, peak_classes=dict(cls),
                   provisional_storms=sum(1 for s in covered if meta[s]["track_type"].lower() != "main"),
                   majors_seen_once=sorted(meta[s]["name"] for s, ds in covered.items()
                                           if meta[s]["peak"] >= p["major_kt"] and len(ds) < 2),
                   excluded_storm_fixes={k: len(v) for k, v in excl.items()}, skipped_rows=skipped,
                   storms_in_extract=len(meta))
    (out / "tc_selection.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
