#!/usr/bin/env python3
"""Paired read per lead of the cyclone members of the extremes set XSET-1 (17 Nov 2026). CPU, seconds.

Question: the cyclone deficit of a model row against the 9 km ensemble grows with lead (note s7.2 point 2). Is it the
input that changes with lead (weaker or broader input storms), or the model that adds less to the same input?

Reads tc_members.csv (xset_score_events.py): one line per (case, row, member). Member k of INPUT is the input of member k
of every model row (one draw per input member) and has the number of member k of ENS9.

    python3 xset_tc_lead_read.py --ens DIR_ENS9 --row GL512_500k=DIR [--row GL512_450k=DIR ...] --out tc_lead_read.md

DIR_ENS9 holds the rows ENS9, INPUT and CF9 (scored with --with-refs). Only the cases present in every row are used.
Tables (wind = maximum 10 m wind within 300 km, 10-minute m/s; pressure = minimum in the 300 km disc, hPa; every member
counts, the *_all rule):
  1  means per lead: best track, INPUT, ENS9, each model; model - ENS9 with a delete-one-date jackknife SE.
  2  what each row adds to its input, per lead: mean over members of (row_k - INPUT_k), wind and pressure.
  3  the same by class of the input member's wind and by lead group (the transfer of the input, without a fit).
  4  pairing per lead: correlation across members, inside a case, of INPUT_k with ENS9_k and with model_k; median
     distance between the centres of member k in two rows (both with a closed centre).
  5  structure per lead: share of members with a closed centre, mean radius of maximum wind.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = (12, 24, 36, 48, 60, 72)
GROUPS = {"+12/+24 h": (12, 24), "+36/+48 h": (36, 48), "+60/+72 h": (60, 72)}
IN_EDGES = (0.0, 15.0, 22.0, 1e9)            # classes of the input member's wind, m/s (36 km field)
IN_NAMES = ("input < 15", "input 15-22", "input >= 22")
KEY = ["sid", "date", "lead", "member"]
CASE = ["sid", "date", "lead"]


def gc_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = (np.radians(np.asarray(v, np.float64)) for v in (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def jack_se(df, col):
    """Delete-one-date jackknife SE of the mean of col over the lines of df (a date = a block)."""
    dates = df["date"].unique()
    if len(dates) < 3:
        return np.nan
    est = np.array([df.loc[df["date"] != d, col].mean() for d in dates])
    n = len(dates)
    return float(np.sqrt((n - 1) / n * ((est - est.mean()) ** 2).sum()))


def within_corr(df, a, b):
    """Correlation of a and b across members inside a case, pooled over cases (deviations from the case means)."""
    g = df.groupby(CASE)
    da, db = df[a] - g[a].transform("mean"), df[b] - g[b].transform("mean")
    den = np.sqrt((da ** 2).sum() * (db ** 2).sum())
    return float((da * db).sum() / den) if den > 0 else np.nan


def f(x, nd=2):
    return "" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def table(head, lines):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(v) for v in ln) + " |" for ln in lines]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ens", required=True, help="score folder of the ENS9 row (rows ENS9, INPUT, CF9)")
    ap.add_argument("--row", action="append", default=[], help="LABEL=DIR of a model row, repeatable")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dt = {"sid": str, "date": str}
    ens = pd.read_csv(Path(a.ens) / "tc_members.csv", dtype=dt)
    models = []
    parts = [ens[ens["row"].isin(["ENS9", "INPUT"])]]
    for spec in a.row:
        lab, d = spec.split("=", 1)
        t = pd.read_csv(Path(d) / "tc_members.csv", dtype=dt)
        parts.append(t[t["row"] == lab])
        models.append(lab)
    rows = ["INPUT", "ENS9"] + models
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(index=KEY, columns="row", values=["wind_ms", "pmin_hpa", "clat", "clon", "rmw_km", "found"],
                            aggfunc="first", dropna=False)
    bt = long.drop_duplicates(CASE).set_index(CASE)[["bt_wind_ms", "bt_pres_hpa"]]
    ok = np.ones(len(wide), bool)
    for r in rows:
        ok &= np.isfinite(wide[("wind_ms", r)].to_numpy(np.float64)) & np.isfinite(wide[("pmin_hpa", r)].to_numpy(np.float64))
    w = wide[ok].copy()
    w.columns = [f"{v}:{r}" for v, r in w.columns]
    w = w.reset_index().join(bt, on=CASE)
    n_case = w.drop_duplicates(CASE).groupby("lead").size()
    md = [f"# Cyclone members per lead, paired read (XSET-1)\n",
          f"Rows: {', '.join(rows)}. {len(w)} member lines, {len(w.drop_duplicates(CASE))} cases, "
          f"{w['sid'].nunique()} storms, {w['date'].nunique()} dates. Wind in m/s (10-minute), pressure in hPa.\n"]

    def sel(L):
        return w if L is None else w[w["lead"] == L]

    # ---- 1 means per lead
    for var, btc, nd in (("wind_ms", "bt_wind_ms", 2), ("pmin_hpa", "bt_pres_hpa", 1)):
        lines = []
        for L in LEADS + (None,):
            s = sel(L)
            s = s[np.isfinite(s[btc])]
            if s.empty:
                continue
            ln = [("all" if L is None else f"+{L}"), len(s.drop_duplicates(CASE)), f(s.drop_duplicates(CASE)[btc].mean(), nd)]
            ln += [f(s[f"{var}:{r}"].mean(), nd) for r in rows]
            for m in models:
                s = s.assign(_d=s[f"{var}:{m}"] - s[f"{var}:ENS9"])
                ln += [f"{f(s['_d'].mean(), 2)} ({f(jack_se(s, '_d'), 2)})"]
            lines.append(ln)
        md += [f"\n## 1. Mean {'wind' if var == 'wind_ms' else 'pressure'} per lead (cases with a best-track value)\n",
               table(["lead", "cases", "best track"] + rows + [f"{m} - ENS9 (SE)" for m in models], lines)]

    # ---- 2 what each row adds to its input
    for var, nd in (("wind_ms", 2), ("pmin_hpa", 2)):
        lines = []
        for L in LEADS + (None,):
            s = sel(L)
            ln = [("all" if L is None else f"+{L}"), len(s)]
            for r in rows[1:]:
                s = s.assign(_d=s[f"{var}:{r}"] - s[f"{var}:INPUT"])
                ln += [f"{f(s['_d'].mean(), nd)} ({f(jack_se(s, '_d'), nd)})"]
            lines.append(ln)
        md += [f"\n## 2. Added to the input member, {'wind' if var == 'wind_ms' else 'pressure'}: mean of row_k - INPUT_k (SE)\n",
               table(["lead", "member lines"] + rows[1:], lines)]

    # ---- 3 by class of the input member's wind and lead group
    cls = np.digitize(w["wind_ms:INPUT"].to_numpy(np.float64), IN_EDGES[1:-1])
    lines = []
    for gname, gl in GROUPS.items():
        for ci, cname in enumerate(IN_NAMES):
            s = w[(cls == ci) & w["lead"].isin(gl)]
            if len(s) < 20:
                continue
            ln = [gname, cname, len(s), f(s["wind_ms:INPUT"].mean())]
            for r in rows[1:]:
                s = s.assign(_d=s[f"wind_ms:{r}"] - s["wind_ms:INPUT"])
                ln += [f"{f(s['_d'].mean())} ({f(jack_se(s, '_d'))})"]
            lines.append(ln)
    md += ["\n## 3. Wind added to the input member, by class of the input member's wind (m/s) and lead group\n",
           "If the model column is flat across the lead groups inside a class while the ENS9 column rises, the model is "
           "blind to the lead. If both fall or both stay, the deficit comes from elsewhere.\n\n",
           table(["leads", "class", "member lines", "mean input wind"] + [f"{r} - INPUT (SE)" for r in rows[1:]], lines)]

    # ---- 4 pairing
    pairs = [("INPUT", "ENS9")] + [("INPUT", m) for m in models] + [("ENS9", m) for m in models]
    lines = []
    for L in LEADS + (None,):
        s = sel(L)
        ln = ["all" if L is None else f"+{L}"]
        for x, y in pairs:
            both = s[(s[f"found:{x}"] == 1) & (s[f"found:{y}"] == 1)]
            dist = gc_km(both[f"clat:{x}"], both[f"clon:{x}"], both[f"clat:{y}"], both[f"clon:{y}"]) if len(both) else np.array([np.nan])
            ln += [f(within_corr(s, f"wind_ms:{x}", f"wind_ms:{y}")), f(within_corr(s, f"pmin_hpa:{x}", f"pmin_hpa:{y}")),
                   f(float(np.nanmedian(dist)), 0)]
        lines.append(ln)
    head = ["lead"]
    for x, y in pairs:
        head += [f"r wind {x}~{y}", f"r pres {x}~{y}", f"centre km {x}~{y}"]
    md += ["\n## 4. Pairing of member k between rows: correlation across members inside a case, median centre distance\n",
           table(head, lines)]

    # ---- 5 structure
    lines = []
    for L in LEADS + (None,):
        s = sel(L)
        ln = ["all" if L is None else f"+{L}"]
        for r in rows:
            fd = s[s[f"found:{r}"] == 1]
            ln += [f(100.0 * (s[f"found:{r}"] == 1).mean(), 0), f(fd[f"rmw_km:{r}"].mean(), 0)]
        lines.append(ln)
    head = ["lead"]
    for r in rows:
        head += [f"{r} closed %", f"{r} RMW km"]
    md += ["\n## 5. Structure per lead: members with a closed centre (%), mean radius of maximum wind (km)\n", table(head, lines)]

    Path(a.out).write_text("\n".join(md))
    w.to_csv(Path(a.out).with_suffix(".members.csv"), index=False)
    print(f"TC_LEAD_READ {a.out}: {len(w)} member lines, rows {rows}")


if __name__ == "__main__":
    main()
