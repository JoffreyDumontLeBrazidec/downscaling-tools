"""Reader A: station scores at all leads (note s4.2), from obs_points ``scores_per_date.csv`` files.

One file (and row inside it) per role: RUN, REF (GL512 450k), ENS9, INPUT. A role is ``PATH`` or ``PATH::ROW``; the
row defaults to ``model`` (``input`` for INPUT). Region ``all``, radius 0, the variant of ``gates.VARIANT``. Lines are
joined on (lead, date) and weighted with the REF station count n (as ``xset_table.py``). Differences in % = ratio of
sums - 1; SE = delete-one-block jackknife over date blocks.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from eval.verdict import stats as S
from eval.verdict.gates import MARGIN, VARIANT, GateLine

VARS_CRPS = ("10ff", "tp", "2t", "msl")


def parse_role(spec: str, default_row: str) -> tuple[str, str]:
    path, _, row = spec.partition("::")
    return path, row or default_row


def load_scores(spec: str, default_row: str = "model") -> pd.DataFrame:
    path, row = parse_role(spec, default_row)
    df = pd.read_csv(path, dtype={"date": str})
    if "row" in df:
        df = df[df["row"] == row]
    if df.empty:
        raise SystemExit(f"eval.cli verdict: no lines of row {row!r} in {path}")
    return df


def _select(df: pd.DataFrame, var: str) -> pd.DataFrame:
    q = df[(df["var"] == var) & (df["variant"] == VARIANT.get(var, "raw"))]
    if "radius_km" in q:
        q = q[q["radius_km"] == 0]
    if "region" in q:
        q = q[q["region"] == "all"]
    return q


def joined(frames: dict[str, pd.DataFrame], var: str, cols: tuple[str, ...]) -> pd.DataFrame:
    """One line per (lead, date) present in every role: n (REF) and <col>_<role> for each col."""
    out = None
    for role, df in frames.items():
        q = _select(df, var)
        keep = [c for c in ("n",) + cols if c in q]
        q = q[["lead", "date"] + keep].set_index(["lead", "date"])
        if q.index.duplicated().any():
            raise SystemExit(f"eval.cli verdict: duplicate (lead, date) lines for {var} in role {role}")
        q = q.rename(columns={c: f"{c}_{role}" for c in keep})
        out = q if out is None else out.join(q, how="inner")
    return out if out is not None else pd.DataFrame()


def per_block(t: pd.DataFrame, cols: list[str], weight: str, rule: str, square: bool = False) -> pd.DataFrame:
    """Sum over the lines of each date block of weight x col (or weight x col^2)."""
    if t.empty:
        return pd.DataFrame()
    dates = t.index.get_level_values("date")
    blk = S.blocks(dates, rule)
    w = t[weight].to_numpy(np.float64)
    data = {"N": w}
    for c in cols:
        v = t[c].to_numpy(np.float64)
        data[c] = w * (v ** 2 if square else v)
    d = pd.DataFrame(data, index=[blk[x] for x in dates])
    return d.groupby(level=0).sum()


def _diff_line(gate, sub, hard, blk, a, b, margin, mode, n_dates, noise_ok=False, note=""):
    if blk.empty or a not in blk or b not in blk:
        return GateLine(gate, sub, hard, S.MISSING, note=note or "no common lines")
    d, se = S.pct_ratio(blk[a].to_numpy(), blk[b].to_numpy())
    n = len(blk)
    lo, hi = S.interval(d, se, n)
    if mode == "not_worse":
        st = S.state_not_worse(lo, hi, margin)
    elif mode == "better":
        st = S.state_better(lo, hi, noise_ok)
    else:
        st = "INFO"
    N = blk["N"].sum()
    return GateLine(gate, sub, hard, st, value=float(blk[a].sum() / N), ref_value=float(blk[b].sum() / N), diff=d, unit="%",
                    se=se, lo=lo, hi=hi, margin=margin if mode == "not_worse" else math.nan, mde=S.mde(se, n),
                    n_dates=n_dates, n_blocks=n, note=note)


def run(run_spec: str, ref_spec: str, ens9_spec: str, input_spec: str, rule: str = "fixed5",
        members: int = 10) -> list[GateLine]:
    frames = {"RUN": load_scores(run_spec), "REF": load_scores(ref_spec), "ENS9": load_scores(ens9_spec),
              "INPUT": load_scores(input_spec, "input")}
    out: list[GateLine] = []
    bands = [S.band_name(lo, hi) for lo, hi in S.LEAD_BANDS]
    last = bands[-1]
    for var in VARS_CRPS:
        t = joined(frames, var, ("fcrps", "spread_rms", "rmse_mean"))
        if t.empty:
            for g in ("G-A1" if var in ("10ff", "tp") else "G-A2", "G-A3"):
                out.append(GateLine(g, var, True, S.MISSING, note="no common (lead, date) lines"))
            continue
        t = t.rename(columns={"n_REF": "n"})
        leads = t.index.get_level_values("lead").to_numpy()
        tb = t.assign(band=[S.band_of(x) for x in leads])
        fc = [f"fcrps_{r}" for r in frames]
        g_ref = "G-A1" if var in ("10ff", "tp") else "G-A2"
        for band in bands:
            sub = tb[tb.band == band]
            nd = sub.index.get_level_values("date").nunique()
            if sub.empty:
                out.append(GateLine(g_ref, f"{var} {band}", True, S.MISSING, note="no lead in band"))
                if var != "msl":
                    out.append(GateLine("G-A4", f"{var} {band}", True, S.MISSING, note="no lead in band"))
                out.append(GateLine("G-A5", f"{var} {band}", True, S.MISSING, note="no lead in band"))
                continue
            blk = per_block(sub, fc, "n", rule)
            out.append(_diff_line(g_ref, f"{var} {band}", True, blk, "fcrps_RUN", "fcrps_REF", MARGIN[g_ref][var], "not_worse", nd))
            if var != "msl":
                out.append(_diff_line("G-A4", f"{var} {band}", True, blk, "fcrps_RUN", "fcrps_INPUT", math.nan, "better", nd,
                                      noise_ok=band == last))
            out.append(_spread_line(sub, var, band, rule, members, nd))
        blk_all = per_block(tb, fc, "n", rule)
        nd_all = tb.index.get_level_values("date").nunique()
        out.append(_diff_line("G-A3", f"{var} all leads", True, blk_all, "fcrps_RUN", "fcrps_ENS9", MARGIN["G-A3"][var],
                              "not_worse", nd_all))
        # reported, not gated: per lead vs REF, ensemble-mean RMSE vs ENS9 and INPUT per band
        for L in sorted(set(leads)):
            sub = tb[leads == L]
            out.append(_diff_line("A-lead", f"{var} +{int(L)} h fcrps vs REF", False, per_block(sub, fc, "n", rule),
                                  "fcrps_RUN", "fcrps_REF", math.nan, "info", sub.index.get_level_values("date").nunique()))
        rm = [f"rmse_mean_{r}" for r in frames]
        for band in bands:
            sub = tb[tb.band == band]
            if sub.empty:
                continue
            b2 = per_block(sub, rm, "n", rule, square=True)
            nd = sub.index.get_level_values("date").nunique()
            for other in ("ENS9", "INPUT"):
                ln = _diff_line("A-rmse", f"{var} {band} ens-mean MSE vs {other}", False, b2, "rmse_mean_RUN",
                                f"rmse_mean_{other}", math.nan, "info", nd, note="ratio of mean squared errors, %")
                out.append(ln)
    return out


def _spread_line(sub: pd.DataFrame, var: str, band: str, rule: str, members: int, nd: int) -> GateLine:
    """G-A5: SSR = sqrt(sum n spread_rms^2 / sum n rmse^2) x sqrt((M+1)/M) within 0.10 of ENS9's, and the spread
    ratio to ENS9 not further from 1 than GL512 450k's + 2 SE."""
    need = [f"{c}_{r}" for c in ("spread_rms", "rmse_mean") for r in ("RUN", "REF", "ENS9")]
    if any(c not in sub for c in need):
        return GateLine("G-A5", f"{var} {band}", True, S.MISSING, note="spread_rms or rmse_mean absent")
    b = per_block(sub, need, "n", rule, square=True)
    n = len(b)
    f = math.sqrt((members + 1) / members)
    ssr = {r: math.sqrt(b[f"spread_rms_{r}"].sum() / b[f"rmse_mean_{r}"].sum()) * f for r in ("RUN", "REF", "ENS9")}
    vr, ve, vf = (b[f"spread_rms_{r}"].to_numpy() for r in ("RUN", "ENS9", "REF"))
    r_run, se = S.jackknife(lambda k: math.sqrt(vr[k].sum() / ve[k].sum()), n)
    r_ref = math.sqrt(vf.sum() / ve.sum())
    ok_ssr = abs(ssr["RUN"] - ssr["ENS9"]) <= MARGIN["G-A5_ssr_abs"]
    ok_ratio = abs(r_run - 1) <= abs(r_ref - 1) + MARGIN["G-A5_ratio_se"] * (se if math.isfinite(se) else 0.0)
    st = S.PASS if ok_ssr and ok_ratio else S.FAIL
    note = (f"SSR run {ssr['RUN']:.3f} ENS9 {ssr['ENS9']:.3f} REF {ssr['REF']:.3f}; spread ratio to ENS9 run "
            f"{r_run:.3f} REF {r_ref:.3f}" + ("" if ok_ssr else "; SSR off ENS9 by more than 0.10")
            + ("" if ok_ratio else "; spread ratio further from 1 than REF + 2 SE"))
    return GateLine("G-A5", f"{var} {band}", True, st, value=r_run, ref_value=r_ref, unit="spread ratio to ENS9", se=se,
                    margin=MARGIN["G-A5_ssr_abs"], n_dates=nd, n_blocks=n, note=note)
