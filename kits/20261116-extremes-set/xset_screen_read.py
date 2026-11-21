#!/usr/bin/env python3
"""SCREEN-TC read of the extremes set XSET-1 (21 Nov 2026). CPU, seconds. A SCREEN, NEVER A VERDICT.

SCREEN-TC (adopted by the owner 21 Nov 2026 15:13 UTC; docs epics/misc/20261121_information_gain_audit.md s8) = the 22
cyclone dates x members 1-5 x 6 leads, about 26 GH200-h per row instead of 97.5. It flags whether an arm is worth a full
XSET-1 row. It never enters a verdict, a scoreboard or a paper table (rules 46, 48 unchanged).

Cyclone fair CRPS against IBTrACS, members 1-5 of EVERY row (the screen row and the full reference rows are cut to the
same members: members 1-5 of a full row carry the same rule-31 seeds as the screen's), *_all rule (minimum pressure in
the search disc, maximum 10 m wind within 300 km). Same arithmetic as the calibration (calibrate.py of the audit, design
"members 5"): mean of paired case differences, SE delete-one-storm jackknife, MDE = (t.975 + t.80, storms-1 df) x SE.
Call of a score: sign of the difference if |d| > 2.8 SE and |d| > line; line = 0.4 hPa / 0.3 m/s for a training pair
(run wander, audit s3), 0 for a sampler pair (same weights).

    python3 xset_screen_read.py --screen FS242X_SCRTC=A/scores/FS242X_SCRTC --ref GL512_450k=A/scores/GL512_450k \
        [--ref ROW=DIR ...] [--ens A/scores/ENS9] [--kind training|sampler] [--dates lists/dates_xset1.csv] --out DIR

Reads tc_members.csv of each folder (xset_score_events.py). The screen row may also be a full row (it is cut to members
1-5): that is how the read is checked against the calibration. Writes DIR/screen_read.md and DIR/screen_read.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

LEADS = (12, 24, 36, 48, 60, 72)
MEMBERS = (1, 2, 3, 4, 5)
LINES = {"training": {"pmin": 0.4, "wind": 0.3}, "sampler": {"pmin": 0.0, "wind": 0.0}}
VARS = (("pmin", "pmin_hpa", "bt_pres_hpa", "hPa"), ("wind", "wind_ms", "bt_wind_ms", "m/s"))
# audit s8: SCREEN-TC MDE for training pairs about 0.63 hPa / 0.44 m/s, sampler pairs 0.28 / 0.11
S8_MDE = {"training": {"pmin": 0.63, "wind": 0.44}, "sampler": {"pmin": 0.28, "wind": 0.11}}


def tq(n):
    return float(tdist.ppf(0.975, n - 1) + tdist.ppf(0.80, n - 1)) if n > 2 else np.nan


def load(d, row):
    q = pd.read_csv(Path(d) / "tc_members.csv", dtype={"date": str, "sid": str, "row": str},
                    keep_default_na=False, na_values=["", "nan"])
    q = q[(q.row == row) & q.lead.isin(LEADS)]
    if q.empty:
        raise SystemExit(f"FATAL no lines of row {row} in {d}/tc_members.csv")
    q = q.assign(case=q.sid + "|" + q.date + "|" + q.lead.astype(str))
    return q


def case_crps(q):
    """Fair CRPS per case over members 1-5 (a case counts only with all 5 members and a best-track value)."""
    q = q[q.member.isin(MEMBERS)]
    res = {}
    for var, col, bt, _ in VARS:
        p = q.pivot_table(index="case", columns="member", values=col)
        o = q.groupby("case")[bt].first().reindex(p.index)
        ok = (p.notna().sum(axis=1) == len(MEMBERS)) & o.notna() if p.shape[1] == len(MEMBERS) else pd.Series(False, p.index)
        x, y = p[ok].to_numpy(np.float64).T, o[ok].to_numpy(np.float64)
        m = x.shape[0]
        t1 = np.abs(x - y[None]).mean(0)
        s = np.sort(x, 0)
        w = (2 * np.arange(1, m + 1) - m - 1)[:, None]
        res[var] = pd.Series(t1 - (w * s).sum(0) / (m * (m - 1)), index=p.index[ok])
    return res


def storm_jk(dd, sid):
    """Mean of paired case differences; delete-one-storm jackknife SE; number of storms."""
    g = pd.DataFrame({"d": dd.to_numpy(), "s": sid}).groupby("s")["d"].agg(["sum", "count"])
    D, C = g["sum"].to_numpy(), g["count"].to_numpy().astype(float)
    d = D.sum() / C.sum()
    n = len(D)
    if n < 3:
        return d, np.nan, n
    th = (D.sum() - D) / (C.sum() - C)
    return float(d), float(np.sqrt((n - 1) / n * ((th - th.mean()) ** 2).sum())), n


def call(d, se, line):
    if not np.isfinite(se) or abs(d) <= 2.8 * se or abs(d) <= line:
        return "no call"
    return "better" if d < 0 else "worse"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", required=True, help="ROW=DIR of the screen row")
    ap.add_argument("--ref", action="append", default=[], help="ROW=DIR of a full reference row (repeat)")
    ap.add_argument("--ens", default="", help="scores folder of ENS9 (rows ENS9 and INPUT, scored with --with-refs)")
    ap.add_argument("--kind", choices=("training", "sampler"), default="training")
    ap.add_argument("--dates", default="", help="dates_xset1.csv: counts the cyclone dates the screen covers")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    srow, sdir = a.screen.split("=", 1)
    rows = {srow: load(sdir, srow)}
    for r in a.ref:
        k, v = r.split("=", 1)
        rows[k] = load(v, k)
    ctx = {}
    if a.ens:
        for k in ("ENS9", "INPUT"):
            ctx[k] = load(a.ens, k)
    cc = {k: case_crps(q) for k, q in {**rows, **ctx}.items()}
    cinfo = pd.concat(list(rows.values()) + list(ctx.values())).groupby("case")[["sid", "date", "lead"]].first()
    sq = rows[srow]
    nm = sq.groupby("case").member.nunique()
    n_dates_screen = sq.date.nunique()
    n_tc_dates = None
    if a.dates:
        dl = pd.read_csv(a.dates, dtype={"date": str})
        n_tc_dates = int(dl.tag.isin(["TC", "both"]).sum())
    partial = n_tc_dates is not None and n_dates_screen < n_tc_dates
    line = LINES[a.kind]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, md = [], []
    md.append(f"# SCREEN-TC read: {srow}  (A SCREEN, NOT A VERDICT)\n")
    md.append("SCREEN-TC = 22 cyclone dates x members 1-5 x 6 leads (audit note s8, adopted 21 Nov 2026). It flags whether "
              "the arm is worth a full XSET-1 row; it never enters a verdict, scoreboard or paper table (rules 46, 48). "
              "Every row below is cut to members 1-5.\n")
    md.append(f"- screen row {srow}: {sq.case.nunique()} cases, {sq.sid.nunique()} storms, {n_dates_screen} start dates"
              + (f" of {n_tc_dates} cyclone dates" if n_tc_dates else "") + f", members per case {sorted(set(nm))}")
    if partial:
        md.append(f"- PARTIAL SCREEN: {n_dates_screen} of {n_tc_dates} cyclone dates. Not a SCREEN-TC flag; numbers below "
                  "are for checking only.")
    md.append(f"- pair kind {a.kind}: line {line['pmin']} hPa / {line['wind']} m/s; call = |d| > 2.8 SE and > line; "
              f"expected screen MDE (s8) {S8_MDE[a.kind]['pmin']} hPa / {S8_MDE[a.kind]['wind']} m/s\n")

    md.append("## Paired difference, screen row minus reference (negative = screen row better)\n")
    md.append("| reference | score | cases | storms | d | SE | MDE | call |")
    md.append("|---|---|---|---|---|---|---|---|")
    calls = []
    for ref in [r for r in rows if r != srow]:
        for var, _, _, unit in VARS:
            dd = (cc[srow][var] - cc[ref][var]).dropna()
            ci = cinfo.loc[dd.index]
            d, se, n = storm_jk(dd, ci.sid.to_numpy())
            mde = tq(n) * se
            c = call(d, se, line[var])
            calls.append((ref, var, c))
            recs.append(dict(screen=srow, ref=ref, score=f"{var} fair CRPS", unit=unit, lead="all", cases=len(dd), storms=n,
                             d=d, se=se, mde=mde, call=c, kind=a.kind, partial=partial))
            md.append(f"| {ref} | {var} fair CRPS ({unit}) | {len(dd)} | {n} | {d:+.3f} | {se:.3f} | {mde:.3f} | {c} |")
            for L in LEADS:
                m = (ci.lead == L).to_numpy()
                if m.sum() == 0:
                    continue
                dl_, sel_, nl = storm_jk(dd[m], ci.sid.to_numpy()[m])
                recs.append(dict(screen=srow, ref=ref, score=f"{var} fair CRPS", unit=unit, lead=L, cases=int(m.sum()),
                                 storms=nl, d=dl_, se=sel_, mde=tq(nl) * sel_, call="", kind=a.kind, partial=partial))
    md.append("")
    md.append("## Per lead, screen row minus reference (information only, no call)\n")
    md.append("| reference | score | " + " | ".join(f"+{L} h" for L in LEADS) + " |")
    md.append("|---|---|" + "---|" * len(LEADS))
    rd = pd.DataFrame(recs)
    for ref in [r for r in rows if r != srow]:
        for var, _, _, unit in VARS:
            cells = []
            for L in LEADS:
                x = rd[(rd.ref == ref) & (rd.score == f"{var} fair CRPS") & (rd.lead == L)]
                cells.append(f"{x.d.iloc[0]:+.2f} ({x.se.iloc[0]:.2f}, n {x.cases.iloc[0]})" if len(x) else "-")
            md.append(f"| {ref} | {var} ({unit}) | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Levels on the common cases, members 1-5 (fair CRPS against IBTrACS)\n")
    common = None
    for k in cc:
        for var, *_ in VARS:
            ix = set(cc[k][var].index)
            common = ix if common is None else common & ix
    common = sorted(common or [])
    md.append(f"{len(common)} cases present in every row.\n")
    md.append("| row | pmin (hPa) | wind (m/s) |")
    md.append("|---|---|---|")
    for k in cc:
        md.append(f"| {k} | {cc[k]['pmin'].reindex(common).mean():.2f} | {cc[k]['wind'].reindex(common).mean():.2f} |")
    md.append("")

    better = [c for c in calls if c[2] == "better"]
    worse = [c for c in calls if c[2] == "worse"]
    if partial:
        flag = "PARTIAL (no flag)"
    elif better and not worse:
        flag = "WORTH A FULL ROW (screen row better beyond the screen's noise: " + ", ".join(f"{v} vs {r}" for r, v, _ in better) + ")"
    elif worse and not better:
        flag = "NOT WORTH A FULL ROW (screen row worse: " + ", ".join(f"{v} vs {r}" for r, v, _ in worse) + ")"
    elif better and worse:
        flag = "MIXED: worth a full row to settle (better: " + ", ".join(f"{v} vs {r}" for r, v, _ in better) + \
               "; worse: " + ", ".join(f"{v} vs {r}" for r, v, _ in worse) + ")"
    else:
        flag = ("NO CALL: no difference above the screen's detection limit (MDE column). A full row only helps if the "
                "effect that matters is smaller than that MDE.")
    md.insert(4 if not partial else 5, f"- SCREEN FLAG: {flag}\n")
    md.append("Limits (s8): run-to-run wander (about 0.4 hPa) is not reduced by a screen; a training verdict still needs "
              "rungs or replicates. Rain shares and station scores are not read on a screen (not calibrated on TC dates; "
              "5-member station scores gave wrong-sign calls on the 2023 set).")
    (out / "screen_read.md").write_text("\n".join(md) + "\n")
    rd.to_csv(out / "screen_read.csv", index=False)
    print(f"SCREEN_READ {srow} flag: {flag}")
    print(f"wrote {out / 'screen_read.md'}")


if __name__ == "__main__":
    main()
