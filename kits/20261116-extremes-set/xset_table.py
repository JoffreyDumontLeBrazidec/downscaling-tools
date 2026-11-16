#!/usr/bin/env python3
"""The one table of the extremes set XSET-1, with the error bar and the detectable effect of every line.

    python3 xset_table.py --row ENS9=DIR --row GL512_500k=DIR [--row ARM=DIR ...] --ref ENS9 --dates dates_xset1.csv \
        --hist HISTDIR --out DIR

Each DIR holds obs_points/scores_per_date.csv (xset_score_stations.py), tc_cases_scored.csv and imerg_discs.csv
(xset_score_events.py). The INPUT row (native 36 km ensemble) and CF9 (the single 9 km field) are read from the --ref
directory (they are the same in every directory). Conventions (docs note 20261116_extremes_cyclone_eval_set.md s3.2,
detectable-effect note s1):
  * station and IMERG lines: value = sum over dates and leads of (points x score) / sum of points; difference to the
    reference row in % = ratio of sums - 1; SE = delete-one-date jackknife; six leads pooled.
  * cyclone lines: value = mean over cases; difference = mean of the paired case differences (hPa, m/s, km);
    SE = delete-one-storm jackknife.
  * detectable effect (MDE) = (t0.975 + t0.80, units - 1 degrees of freedom) x SE: the true difference found with 80 %
    power by a two-sided 5 % test. eff = (sum of events)^2 / sum of squares over the units (dates or storms).
Writes xset_table.csv (every line) and xset_table.md (the reading table).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = (12, 24, 36, 48, 60, 72)


def tq(n):
    from scipy.stats import t
    return float(t.ppf(0.975, n - 1) + t.ppf(0.80, n - 1)) if n > 1 else np.nan


def jackknife(stat, n):
    """stat(keep) -> value for a boolean keep mask over n units. Returns (value, SE)."""
    full = stat(np.ones(n, bool))
    if n < 3:
        return full, np.nan
    th = np.empty(n)
    for i in range(n):
        k = np.ones(n, bool)
        k[i] = False
        th[i] = stat(k)
    return full, float(np.sqrt((n - 1) / n * np.sum((th - th.mean()) ** 2)))


def eff(x):
    x = np.asarray(x, np.float64)
    return float(x.sum() ** 2 / np.sum(x ** 2)) if np.sum(x ** 2) > 0 else 0.0


def line(block, score, subset, row, ref, value, ref_value, diff, unit, se, n_units, unit_name, n_events=np.nan, eff_units=np.nan):
    return dict(block=block, score=score, subset=subset, row=row, ref=ref, value=value, ref_value=ref_value, diff=diff,
                diff_unit=unit, se=se, mde=tq(n_units) * se if np.isfinite(se) else np.nan, n_units=n_units,
                unit=unit_name, n_events=n_events, eff_units=eff_units)


def ratio_lines(block, score, subset, per_unit, rows, ref, unit_name, events=None):
    """per_unit: DataFrame indexed by unit with columns N and S_<row>. Ratio-of-sums lines for every row against ref."""
    out = []
    n = len(per_unit)
    if n == 0 or f"S_{ref}" not in per_unit:
        return out
    N = per_unit["N"].to_numpy(np.float64)
    ev = (np.nan, np.nan) if events is None else (float(events.sum()), eff(events))
    sb = per_unit[f"S_{ref}"].to_numpy(np.float64)
    for r in rows:
        if f"S_{r}" not in per_unit:
            continue
        sa = per_unit[f"S_{r}"].to_numpy(np.float64)
        val = sa.sum() / N.sum()
        if r == ref:
            _, se = jackknife(lambda k: sa[k].sum() / N[k].sum(), n)
            out.append(line(block, score, subset, r, ref, val, val, 0.0, "value", se, n, unit_name, *ev))
            continue
        d, se = jackknife(lambda k: 100.0 * (sa[k].sum() / sb[k].sum() - 1.0) if sb[k].sum() > 0 else np.nan, n)
        out.append(line(block, score, subset, r, ref, val, sb.sum() / N.sum(), d, "%", se, n, unit_name, *ev))
    return out


def station_lines(tabs, ref, tags):
    """tabs {row label: scores_per_date DataFrame}. INPUT is the 'input' row of the reference table."""
    out = []
    frames = {}
    for lab, df in tabs.items():
        frames[lab] = df[df.row == "model"]
    frames["INPUT"] = tabs[ref][tabs[ref].row == "input"]
    frames["CF9"] = tabs[ref][tabs[ref].row == "truth"]
    rows = list(frames)
    key = ["var", "variant", "radius_km", "region", "lead", "date"]
    specs = [("bulk", "2t", "lapse", 0.0, "fcrps", None), ("bulk", "10ff", "raw", 0.0, "fcrps", None),
             ("bulk", "msl", "raw", 0.0, "fcrps", None), ("bulk", "tp", "raw", 0.0, "fcrps", None)]
    for t in (20, 50, 100):
        for rad in (0.0, 36.0):
            specs.append(("heavy rain", "tp", "raw", rad, f"twcrps{t}", f"n_obs_gt{t}"))
        specs.append(("heavy rain", "tp", "raw", 0.0, f"brier{t}", f"n_obs_gt{t}"))
    for t in (15, 20):
        specs.append(("strong wind", "10ff", "raw", 0.0, f"twcrps{t}", f"n_obs_gt{t}"))
    subsets = {"all dates": None, "RAIN dates": {d for d, g in tags.items() if g in ("RAIN", "both")},
               "TC-only dates": {d for d, g in tags.items() if g == "TC"}}
    regions = sorted(set(tabs[ref].region))
    for block, var, variant, rad, col, evcol in specs:
        for region in regions:
            if block != "bulk" and region not in ("all", "tropics_30S_30N", "north_30N"):
                continue
            for sname, dset in subsets.items():
                if block == "bulk" and sname != "all dates" and region != "all":
                    continue
                per = None
                for lab, df in frames.items():
                    q = df[(df["var"] == var) & (df.variant == variant) & (df.radius_km == rad) & (df.region == region)]
                    if dset is not None:
                        q = q[q.date.isin(dset)]
                    if col not in q or q.empty:
                        continue
                    q = q[key + ["n", col] + ([evcol] if evcol else [])].dropna(subset=[col]).set_index(key)
                    q = q.rename(columns={col: f"s_{lab}", "n": f"n_{lab}", **({evcol: f"e_{lab}"} if evcol else {})})
                    per = q if per is None else per.join(q, how="inner")
                if per is None or per.empty:
                    continue
                nref = per[f"n_{ref}"]
                same = all((per[f"n_{lab}"] == nref).all() for lab in frames if f"n_{lab}" in per)
                agg = pd.DataFrame({"N": nref.groupby(level="date").sum()})
                for lab in frames:
                    if f"s_{lab}" in per:
                        agg[f"S_{lab}"] = (per[f"s_{lab}"] * nref).groupby(level="date").sum()
                events = per[f"e_{ref}"].groupby(level="date").sum() if evcol else None
                name = f"{var} {col}" + (f" disc {rad:g} km" if rad else "") + f" [{region}]" + ("" if same else " (station counts differ)")
                out += ratio_lines(block, name, sname, agg, rows, ref, "dates", events)
                if evcol:   # frequency bias of the tail: forecast frequency / observed frequency
                    fcol = col.replace("twcrps", "fc_freq_gt").replace("brier", "fc_freq_gt")
                    if col.startswith("twcrps") and rad == 0.0:
                        for lab, df in frames.items():
                            q = df[(df["var"] == var) & (df.variant == variant) & (df.radius_km == rad) & (df.region == region)]
                            if dset is not None:
                                q = q[q.date.isin(dset)]
                            if fcol not in q or q.empty:
                                continue
                            g = q.assign(f=q[fcol] * q.n).groupby("date")[["f", evcol]].sum()
                            f, e = g["f"].to_numpy(), g[evcol].to_numpy()
                            if e.sum() <= 0:
                                continue
                            v, se = jackknife(lambda k: f[k].sum() / e[k].sum() if e[k].sum() > 0 else np.nan, len(g))
                            out.append(line(block, f"{var} frequency ratio > {col[6:]} (forecast / observed) [{region}]", sname,
                                            lab, ref, v, np.nan, np.nan, "ratio", se, len(g), "dates", float(e.sum()), eff(e)))
    return out


def tc_lines(tabs, ref):
    out = []
    frames = {lab: df[df.row == lab] for lab, df in tabs.items()}
    frames["INPUT"] = tabs[ref][tabs[ref].row == "INPUT"]
    frames["CF9"] = tabs[ref][tabs[ref].row == "CF9"]
    key = ["sid", "date", "lead"]
    base = frames[ref].set_index(key)
    kt = pd.to_numeric(base.bt_wind_kt, errors="coerce")
    classes = {"all cases": kt > -1, "under 64 kt": kt < 64, "64-95 kt": (kt >= 64) & (kt < 96), "96 kt or more": kt >= 96}
    for b in sorted(set(base.basin)):
        classes[f"basin {b}"] = base.basin == b
    for L in LEADS:
        classes[f"lead +{L} h"] = base.index.get_level_values("lead") == L
    metrics = [("pmin_crps_all", "central pressure fair CRPS, all members", "hPa"), ("wind_crps_all", "max 10 m wind fair CRPS, all members", "m/s"),
               ("pmin_err_all", "central pressure mean error", "hPa"), ("wind_err_all", "max 10 m wind mean error", "m/s"),
               ("pos_err_mean_km", "position error of the mean centre", "km"), ("pmin_sd_all", "central pressure spread (SD)", "hPa"),
               ("wind_sd_all", "max wind spread (SD)", "m/s"), ("n_found", "members with a closed centre", "members"),
               ("pmin_crps_found", "central pressure fair CRPS, found members", "hPa"), ("wind_crps_found", "max wind fair CRPS, found members", "m/s")]
    for col, name, unit in metrics:
        for cname, cmask in classes.items():
            idx = base.index[np.asarray(cmask)]
            for lab, df in frames.items():
                if col not in df or col not in base:
                    continue
                a = df.set_index(key)[col].reindex(idx)
                b = base[col].reindex(idx)
                ok = a.notna() & b.notna()
                a, b = a[ok], b[ok]
                if len(a) < 3:
                    continue
                sid = a.index.get_level_values("sid").to_numpy()
                storms, inv = np.unique(sid, return_inverse=True)
                n = len(storms)
                av, bv = a.to_numpy(np.float64), b.to_numpy(np.float64)
                cnt = np.bincount(inv, minlength=n).astype(np.float64)
                sa, sb = np.bincount(inv, av, n), np.bincount(inv, bv, n)
                if lab == ref:
                    v, se = jackknife(lambda k: sa[k].sum() / cnt[k].sum(), n)
                    out.append(line("cyclones", name, cname, lab, ref, v, v, 0.0, "value " + unit, se, n, "storms", len(av), eff(cnt)))
                else:
                    d, se = jackknife(lambda k: (sa[k].sum() - sb[k].sum()) / cnt[k].sum(), n)
                    out.append(line("cyclones", name, cname, lab, ref, av.mean(), bv.mean(), d, unit, se, n, "storms", len(av), eff(cnt)))
    return out


def imerg_lines(tabs, ref):
    out = []
    frames = {lab: df[df.row == lab] for lab, df in tabs.items()}
    frames["INPUT"] = tabs[ref][tabs[ref].row == "INPUT"]
    frames["CF9"] = tabs[ref][tabs[ref].row == "CF9"]
    key = ["date", "lead", "kind", "key"]
    for kind in ("tc", "rain"):
        for col, evcol in (("fcrps", None), ("twcrps20", "n_obs_gt20"), ("twcrps50", "n_obs_gt50")):
            per = None
            for lab, df in frames.items():
                q = df[df.kind == kind]
                if q.empty or col not in q:
                    continue
                q = q[key + ["n", col] + ([evcol] if evcol else [])].set_index(key)
                q = q.rename(columns={col: f"s_{lab}", "n": f"n_{lab}", **({evcol: f"e_{lab}"} if evcol else {})})
                per = q if per is None else per.join(q, how="inner")
            if per is None or per.empty:
                continue
            nref = per[f"n_{ref}"]
            agg = pd.DataFrame({"N": nref.groupby(level="date").sum()})
            for lab in frames:
                if f"s_{lab}" in per:
                    agg[f"S_{lab}"] = (per[f"s_{lab}"] * nref).groupby(level="date").sum()
            events = per[f"e_{ref}"].groupby(level="date").sum() if evcol else None
            out += ratio_lines("IMERG", f"tp {col} in {'cyclone' if kind == 'tc' else 'heavy-rain'} discs", "all dates", agg,
                               list(frames), ref, "dates", events)
    return out


def hist_lines(hist_dir, rows, ref):
    """Global tail frequencies from the histograms: share of grid points above a threshold, row / reference."""
    out = []
    hist_dir = Path(hist_dir)
    data = {}
    for lab in rows + ["INPUT"]:
        per = {}
        for f in sorted(hist_dir.glob(f"{lab}_*_step*.npz")):
            date = f.stem.split("_")[-2]
            z = np.load(f, allow_pickle=True)
            d = per.setdefault(date, dict(tp=0, ws=0, tpmax=[], wsmax=[]))
            d["tp"] = d["tp"] + z["tp_counts"].sum(1)           # (bands, bins), members pooled
            d["ws"] = d["ws"] + z["ws_counts"].sum(1)
            d["tpmax"].append(z["tp_max_mm"].max(1)); d["wsmax"].append(z["ws_max_ms"].max(1))
            d["tp_edges"], d["ws_edges"], d["bands"] = z["tp_edges_mm"], z["ws_edges_ms"], [str(b) for b in z["bands"]]
        if per:
            data[lab] = per
    if ref not in data:
        return out
    bands = next(iter(data[ref].values()))["bands"]
    groups = {"globe": list(range(len(bands))), "tropics 30S-30N": [i for i, b in enumerate(bands) if b in ("30S-10S", "10S-10N", "10N-30N")]}
    for var, thrs in (("tp", (50.0, 100.0, 200.0)), ("ws", (25.0, 32.5, 50.0))):
        for thr in thrs:
            for gname, gi in groups.items():
                common = sorted(set.intersection(*[set(data[lab]) for lab in data]))
                if len(common) < 3:
                    continue

                def share(lab):
                    num, den = [], []
                    for d in common:
                        h = data[lab][d]
                        c = h[var][gi].sum(0)
                        e = h["tp_edges" if var == "tp" else "ws_edges"]
                        num.append(c[e[:-1] >= thr].sum()); den.append(c.sum())
                    return np.asarray(num, np.float64), np.asarray(den, np.float64)

                nb, db = share(ref)
                for lab in data:
                    na, da = share(lab)
                    name = f"{'12 h rain' if var == 'tp' else '10 m wind'} share of grid points >= {thr:g} {'mm' if var == 'tp' else 'm/s'} [{gname}]"
                    if lab == ref:
                        v, se = jackknife(lambda k: nb[k].sum() / db[k].sum(), len(common))
                        out.append(line("global tails", name, "all dates", lab, ref, v, v, 0.0, "value", se, len(common), "dates", nb.sum(), eff(nb)))
                    else:
                        d, se = jackknife(lambda k: 100.0 * ((na[k].sum() / da[k].sum()) / (nb[k].sum() / db[k].sum()) - 1.0)
                                          if nb[k].sum() > 0 else np.nan, len(common))
                        out.append(line("global tails", name, "all dates", lab, ref, na.sum() / da.sum(), nb.sum() / db.sum(), d, "%", se,
                                        len(common), "dates", nb.sum(), eff(nb)))
    return out


def fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", action="append", required=True, help="LABEL=DIR")
    ap.add_argument("--ref", default="ENS9")
    ap.add_argument("--dates", required=True)
    ap.add_argument("--hist", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = dict(r.split("=", 1) for r in a.row)
    tags = {r["date"]: r["tag"] for r in csv.DictReader(open(a.dates))}
    lines = []
    st = {lab: pd.read_csv(Path(d) / "obs_points" / "scores_per_date.csv", dtype={"date": str}) for lab, d in rows.items()
          if (Path(d) / "obs_points" / "scores_per_date.csv").exists()}
    if a.ref in st:
        lines += station_lines(st, a.ref, tags)
    tc = {lab: pd.read_csv(Path(d) / "tc_cases_scored.csv", dtype={"date": str}) for lab, d in rows.items()
          if (Path(d) / "tc_cases_scored.csv").exists()}
    if a.ref in tc:
        lines += tc_lines(tc, a.ref)
    im = {lab: pd.read_csv(Path(d) / "imerg_discs.csv", dtype={"date": str, "key": str}) for lab, d in rows.items()
          if (Path(d) / "imerg_discs.csv").exists() and (Path(d) / "imerg_discs.csv").stat().st_size > 10}
    if a.ref in im:
        lines += imerg_lines(im, a.ref)
    if a.hist:
        lines += hist_lines(a.hist, list(rows), a.ref)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(lines)
    df.to_csv(out / "xset_table.csv", index=False)
    with open(out / "xset_table.md", "w") as f:
        f.write(f"# XSET-1 table (reference row {a.ref}; rows {', '.join(rows)}; INPUT and CF9 from the reference files)\n\n")
        f.write("diff = row against the reference (%, or the unit shown); SE by jackknife over the units; MDE = detectable effect "
                "(80 % power, 5 % two-sided). Lines with eff under 12 are not usable for a verdict.\n")
        for block in df.block.unique():
            b = df[df.block == block]
            f.write(f"\n## {block}\n\n| score | subset | row | value | reference | diff | SE | MDE | units | events | eff |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in b.itertuples(index=False):
                f.write(f"| {r.score} | {r.subset} | {r.row} | {fmt(float(r.value))} | {fmt(float(r.ref_value))} | "
                        f"{fmt(float(r.diff))} {r.diff_unit if r.row != r.ref else ''} | {fmt(float(r.se))} | {fmt(float(r.mde))} | "
                        f"{r.n_units} {r.unit} | {fmt(float(r.n_events), 6)} | {fmt(float(r.eff_units))} |\n")
    print("TABLE_DONE", out / "xset_table.md", len(df), "lines", flush=True)


if __name__ == "__main__":
    main()
