#!/usr/bin/env python3
"""Extremes set XSET-1: are two station score files of the same row equal? (17 Nov 2026)

Used after the station cache was changed (the S25 dates were merged into it and three shared dates fetched again):
the row is scored again on the present cache and compared with its earlier scores_per_date.csv.

    python3 xset_compare_scores.py OLD/obs_points/scores_per_date.csv NEW/obs_points/scores_per_date.csv

Prints STATIONS_SAME when every line and value is equal, else STATIONS_DIFFERENT with, per date, the number of lines
that differ, the largest change of the point count and the largest relative change of a numeric column.
"""
import sys

import numpy as np
import pandas as pd


def main():
    old = pd.read_csv(sys.argv[1], dtype={"date": str})
    new = pd.read_csv(sys.argv[2], dtype={"date": str})
    num = [c for c in old.columns if c in new.columns and pd.api.types.is_numeric_dtype(old[c]) and c not in ("lead", "radius_km")]
    key = [c for c in old.columns if c in new.columns and c not in num]
    print(f"old {len(old)} lines, new {len(new)} lines; key {key}; {len(num)} numeric columns")
    m = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    only = m[m["_merge"] != "both"]
    both = m[m["_merge"] == "both"]
    a = both[[c + "_old" for c in num]].to_numpy(np.float64)
    b = both[[c + "_new" for c in num]].to_numpy(np.float64)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    bad = ~same.all(axis=1)
    if only.empty and not bad.any():
        print(f"STATIONS_SAME {len(both)} lines, {len(num)} numeric columns")
        return 0
    print(f"STATIONS_DIFFERENT: {int(bad.sum())} of {len(both)} common lines differ; {len(only)} lines in one file only")
    rel = np.abs(b - a) / np.maximum(np.abs(a), 1e-12)
    rel[same] = 0.0
    d = both.loc[bad, ["date"]].copy()
    d["max_rel"] = np.nanmax(rel[bad], axis=1)
    ncol = [c for c in num if c in ("n", "n_points", "points", "count", "n_obs")]
    if ncol:
        d["dn"] = (both.loc[bad, ncol[0] + "_new"] - both.loc[bad, ncol[0] + "_old"]).to_numpy()
    tot = both.groupby("date").size()
    for date, g in d.groupby("date"):
        extra = f", point count change {g['dn'].min():+.0f} to {g['dn'].max():+.0f}" if "dn" in g else ""
        print(f"  date {date}: {len(g)} of {tot[date]} lines differ, largest relative change {g['max_rel'].max():.3g}{extra}")
    worst = np.nanmax(rel, axis=0)
    print("  largest relative change per column:", {c: float(f"{w:.3g}") for c, w in zip(num, worst) if w > 0})
    if not only.empty:
        print("  lines in one file only, per date:", only.groupby(["date", "_merge"], observed=True).size().to_dict())
    return 1


if __name__ == "__main__":
    sys.exit(main())
