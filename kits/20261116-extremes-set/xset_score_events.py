#!/usr/bin/env python3
"""Cyclone and IMERG scores of one row of the extremes set XSET-1 (16 Nov 2026).

Reads the row's prediction files (xset_rows_date.py / xset_predict_global.py layout) and writes, in --out:

  tc_cases_scored.csv   one line per (case of tc_cases.csv, row), rows = the label of y_pred (--label), and with
                        --with-refs also INPUT (native x at the nearest native point of each output point, rule 41, as
                        tc_structure) and CF9 (the single 9 km field y).
      Measurements are tc_structure's (find_centre with the best-track fix as first guess and a 300 km search disc,
      measure_structure's maximum 10 m wind within 300 km of the centre) and the scores are tc_obs's (fair CRPS against
      the best-track pressure and against the 10-minute wind: USA_WIND x 0.93, WMO_WIND of a 10-minute agency as it is).
      Two member rules, both written:
        *_all    every member counts: its pressure is the minimum within the search disc and its wind the maximum within
                 300 km of its own centre (of the best-track fix when it has no centre). A member without a storm is
                 a bad forecast of the storm, not a missing one. HEADLINE.
        *_found  only the members with a closed centre inside the disc (tc_obs's rule); n_found says how many.
      Position error: distance of the mean centre of the found members from the best-track fix.
  imerg_discs.csv       one line per (date, lead, disc, row): fair twCRPS of 12 h rain above 20 and 50 mm against IMERG
                        Final at every evaluation point within DISC_KM of a cyclone fix (kind tc) or of a heavy-rain
                        event centre (kind rain); mean over points, with point counts and exceedance frequencies.

    python3 xset_score_events.py --pred DIR --label GL512_500k --tc-cases tc_cases.csv --rain-events rain_events.csv \
        --out DIR [--with-refs]

Needs the downscaling-tools checkout on PYTHONPATH (eval.evaluators.tc_structure.core, obs_points.core.scores).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

KT = 0.514444
USA_1MIN_TO_10MIN = 0.93
SEARCH_KM = 300.0
TC_READ_KM = 850.0
DISC_KM = {"tc": 500.0, "rain": 700.0}
TW = (20.0, 50.0)


def wind_ref_ms(wind_kt, src):
    """Best-track wind as a 10-minute wind in m/s (tc_obs.core.reference: USA and HURDAT winds are 1-minute)."""
    if wind_kt is None:
        return np.nan
    one_min = src.startswith("USA") or "hurdat" in src.lower() or "atcf" in src.lower()
    return wind_kt * KT * (USA_1MIN_TO_10MIN if one_min else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tc-cases", required=True)
    ap.add_argument("--rain-events", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-refs", action="store_true")
    a = ap.parse_args()
    import netCDF4
    from eval.evaluators.obs_points.core.scores import fair_crps, fair_twcrps
    from eval.evaluators.tc_structure.core import StructureParams, find_centre, gc_distance_km, measure_structure
    from eval.shared.native_input import nearest_native_index

    params = replace(StructureParams(), search_km=SEARCH_KM)
    cases = defaultdict(list)
    for c in csv.DictReader(open(a.tc_cases)):
        cases[(c["date"], int(c["lead"]))].append(c)
    events = defaultdict(list)
    if a.rain_events:
        for e in csv.DictReader(open(a.rain_events)):
            events[(e["date"], int(e["lead"]))].append(e)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tc_rows, im_rows = [], []
    files = sorted(Path(a.pred).glob("predictions_*_step*.nc"))
    for f in files:
        m = re.match(r"predictions_(\d{8})_step(\d{3})\.nc", f.name)
        date, lead = m.group(1), int(m.group(2))
        cs, es = cases.get((date, lead), []), events.get((date, lead), [])
        if not cs and not es:
            continue
        with netCDF4.Dataset(str(f)) as d:
            d.set_auto_mask(False)
            ws = [str(s) for s in d["weather_state"][:]]
            lat, lon = np.asarray(d["lat_hres"][:], np.float64), np.asarray(d["lon_hres"][:], np.float64)
            centres = [(float(c["lat"]), float(c["lon"]), TC_READ_KM) for c in cs] + \
                      [(float(e["lat"]), float(e["lon"]), DISC_KM["rain"]) for e in es]
            near = np.zeros(lat.size, bool)
            for la, lo, r in centres:
                near |= gc_distance_km(la, lo, lat, lon) <= r
            sel = np.flatnonzero(near)
            lat, lon = lat[sel], lon[sel]
            yp = np.asarray(d["y_pred"][0][:, sel, :], np.float64)                     # (M, n, S)
            fields = {a.label: yp}
            if a.with_refs:
                fields["CF9"] = np.asarray(d["y"][0][sel, :], np.float64)[None]
                nn = nearest_native_index(np.asarray(d["lat_lres"][:]), np.asarray(d["lon_lres"][:]), lat, lon)
                fields["INPUT"] = np.asarray(d["x"][0], np.float64)[:, nn, :]
            imerg = np.asarray(d["imerg_tp12"][:], np.float64)[sel] if "imerg_tp12" in d.variables else None
        iu, iv, ip, it = ws.index("10u"), ws.index("10v"), ws.index("msl"), ws.index("tp")
        # ---- cyclone cases
        for c in cs:
            bl, bo = float(c["lat"]), float(c["lon"])
            dist = gc_distance_km(bl, bo, lat, lon)
            disc = dist <= SEARCH_KM
            bt_p = float(c["pres_hpa"]) if c["pres_hpa"] else np.nan
            bt_w = wind_ref_ms(float(c["wind_kt"]) if c["wind_kt"] else None, c["wind_src"])
            for row, x in fields.items():
                M = x.shape[0]
                p_all, w_all, p_f, w_f, cl, co, rmw = [], [], [], [], [], [], []
                for k in range(M):
                    msl = x[k, :, ip] / 100.0
                    cen = find_centre(lat, lon, msl, np.ones(lat.size, bool), first_guess=(bl, bo), params=params)
                    clat, clon = (cen["lat"], cen["lon"]) if cen["found"] else (bl, bo)
                    sc, _ = measure_structure(lat, lon, x[k, :, iu], x[k, :, iv], clat, clon, params)
                    p_all.append(float(msl[disc].min()))
                    w_all.append(sc["maxwind300_ms"])
                    if cen["found"]:
                        p_f.append(cen["pmin_hpa"]); w_f.append(sc["maxwind300_ms"]); cl.append(cen["lat"]); co.append(cen["lon"])
                        rmw.append(sc["rmw_km"])
                p_all, w_all, p_f, w_f = (np.asarray(v, np.float64) for v in (p_all, w_all, p_f, w_f))
                r = dict(sid=c["sid"], name=c["name"], basin=c["basin"], date=date, lead=lead, valid=c["valid"], row=row,
                         members=M, n_found=len(p_f), bt_lat=bl, bt_lon=bo, bt_pres_hpa=bt_p, bt_wind_ms=bt_w,
                         bt_wind_kt=c["wind_kt"], track_type=c["track_type"], dist2land_km=c["dist2land_km"])

                def put(tag, p, w):
                    if p.size and np.isfinite(bt_p):
                        r[f"pmin_crps_{tag}"] = float(fair_crps(p[:, None], np.array([bt_p]))[0])
                        r[f"pmin_err_{tag}"] = float(p.mean() - bt_p)
                        r[f"pmin_sd_{tag}"] = float(p.std(ddof=1)) if p.size > 1 else 0.0
                    if w.size and np.isfinite(bt_w) and np.isfinite(w).all():
                        r[f"wind_crps_{tag}"] = float(fair_crps(w[:, None], np.array([bt_w]))[0])
                        r[f"wind_err_{tag}"] = float(w.mean() - bt_w)
                        r[f"wind_sd_{tag}"] = float(w.std(ddof=1)) if w.size > 1 else 0.0
                        r[f"wind_max_{tag}"] = float(w.max())

                put("all", p_all, w_all)
                put("found", p_f, w_f)
                if len(p_f):
                    la_, lo_ = np.radians(cl), np.radians(co)
                    v = np.array([(np.cos(la_) * np.cos(lo_)).mean(), (np.cos(la_) * np.sin(lo_)).mean(), np.sin(la_).mean()])
                    mlat, mlon = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))), float(np.degrees(np.arctan2(v[1], v[0])))
                    r["pos_err_mean_km"] = float(gc_distance_km(bl, bo, np.array([mlat]), np.array([mlon]))[0])
                    r["pos_err_member_km"] = float(gc_distance_km(bl, bo, np.asarray(cl), np.asarray(co)).mean())
                    rm = np.asarray(rmw, np.float64)
                    r["rmw_mean_km"] = float(np.nanmean(rm)) if np.isfinite(rm).any() else np.nan
                    if c.get("usa_rmw_nmi"):
                        r["bt_rmw_km"] = float(c["usa_rmw_nmi"]) * 1.852
                tc_rows.append(r)
        # ---- IMERG discs
        if imerg is not None and np.isfinite(imerg).any():
            discs = [("tc", c["sid"], float(c["lat"]), float(c["lon"])) for c in cs] + \
                    [("rain", f"{e['lat']}_{e['lon']}", float(e["lat"]), float(e["lon"])) for e in es]
            for kind, key, la, lo in discs:
                msk = (gc_distance_km(la, lo, lat, lon) <= DISC_KM[kind]) & np.isfinite(imerg)
                if msk.sum() < 100:
                    continue
                ob = imerg[msk]
                for row, x in fields.items():
                    tp = np.clip(x[:, msk, it], 0.0, None) * 1000.0
                    r = dict(date=date, lead=lead, kind=kind, key=key, lat=la, lon=lo, row=row, members=tp.shape[0],
                             n=int(msk.sum()), fcrps=float(fair_crps(tp, ob).mean()), imerg_mean=float(ob.mean()),
                             fc_mean=float(tp.mean()), imerg_max=float(ob.max()), fc_max=float(tp.max()))
                    for t in TW:
                        r[f"twcrps{t:g}"] = float(fair_twcrps(tp, ob, t).mean())
                        r[f"n_obs_gt{t:g}"] = int((ob > t).sum())
                        r[f"fc_freq_gt{t:g}"] = float((tp > t).mean())
                    im_rows.append(r)
        print(f"scored {f.name}: {len(cs)} cyclone cases, {len(es)} rain events", flush=True)

    def dump(path, rows):
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    dump(out / "tc_cases_scored.csv", tc_rows)
    dump(out / "imerg_discs.csv", im_rows)
    print(f"EVENTS_DONE {a.label}: {len(tc_rows)} cyclone lines, {len(im_rows)} IMERG lines from {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
