"""Reader C: extremes and cyclones on XSET-1 (note s4.4), from the kit's score folders.

A role is a row folder of ``kits/20261116-extremes-set`` scoring (``xset_score_stations.py``,
``xset_score_events.py``): ``obs_points/scores_per_date.csv``, ``tc_cases_scored.csv``, ``tc_members.csv``, and
the tail histograms ``<hist>/<ROW>_<date>_step<LLL>.npz``. ``DIR::ROW`` names the row label inside the files
(default: the folder's only label other than INPUT and CF9). Date blocks: the ``chain`` rule of ``xset_table.py``.
Cyclone lines: storms are the jackknife units.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from eval.verdict import stats as S
from eval.verdict.gates import MARGIN, GateLine
from eval.verdict.reader_a import joined, per_block, _diff_line

KEY = ["sid", "date", "lead"]
REFS = {"INPUT", "CF9"}


def _role(spec: str) -> tuple[Path, str]:
    p, _, row = spec.partition("::")
    return Path(p), row


def _read_tc(path: Path) -> pd.DataFrame:
    # keep_default_na=False: the basin code "NA" is not a missing value
    return pd.read_csv(path, dtype={"date": str, "basin": str, "sid": str, "name": str}, keep_default_na=False,
                       na_values=["", "nan"])


def _label(df: pd.DataFrame, row: str, path: Path) -> str:
    if row:
        return row
    labs = sorted(set(df["row"]) - REFS)
    if len(labs) != 1:
        raise SystemExit(f"eval.cli verdict: {path} holds rows {labs}; name one with DIR::ROW")
    return labs[0]


def cyclones(run_spec: str, ref_spec: str, ens9_spec: str) -> list[GateLine]:
    out = []
    frames = {}
    try:
        for role, spec in (("RUN", run_spec), ("REF", ref_spec), ("ENS9", ens9_spec)):
            d, row = _role(spec)
            df = _read_tc(d / "tc_cases_scored.csv")
            frames[role] = df[df.row == _label(df, row, d)].set_index(KEY)
            if role == "ENS9":
                frames["INPUT"] = df[df.row == "INPUT"].set_index(KEY)
    except (FileNotFoundError, SystemExit) as e:
        return [GateLine("G-C1", "cyclones", True, S.MISSING, note=str(e)),
                GateLine("G-C2", "storm position", True, S.MISSING, note=str(e))]
    for col, what in (("pmin_crps_all", "central pressure fair CRPS"), ("wind_crps_all", "max wind fair CRPS")):
        for other, hard in (("REF", True), ("ENS9", False), ("INPUT", False)):
            a, b = frames["RUN"].get(col), frames[other].get(col)
            if a is None or b is None or not len(b):
                out.append(GateLine("G-C1" if hard else "C-info", f"{what} vs {other}", hard, S.MISSING, note="column absent"))
                continue
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
            sid = j.index.get_level_values("sid").to_numpy()
            storms, inv = np.unique(sid, return_inverse=True)
            n = len(storms)
            sa, sb = np.bincount(inv, j.a.to_numpy(), n), np.bincount(inv, j.b.to_numpy(), n)
            d, se = S.pct_ratio(sa, sb)
            lo, hi = S.interval(d, se, n)
            st = S.state_not_worse(lo, hi, MARGIN["G-C1"]) if hard else "INFO"
            out.append(GateLine("G-C1" if hard else "C-info", f"{what} vs {other}", hard, st, value=float(j.a.mean()),
                                ref_value=float(j.b.mean()), diff=d, unit="%", se=se, lo=lo, hi=hi,
                                margin=MARGIN["G-C1"] if hard else math.nan, mde=S.mde(se, n), n_dates=len(j), n_blocks=n,
                                note=f"{len(j)} cases, {n} storms (jackknife units)"))
    out.append(position(run_spec, ref_spec, ens9_spec))
    return out


def _gc_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = (np.radians(np.asarray(x, np.float64)) for x in (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def position(run_spec: str, ref_spec: str, ens9_spec: str) -> GateLine:
    """G-C2: member k's centre against INPUT member k's centre (tc_members.csv, found centres only), mean per case;
    gate on the mean over cases (storm jackknife), median printed. FAIL if RUN - REF > 2 SE."""
    try:
        dist = {}
        d9, _ = _role(ens9_spec)
        m9 = pd.read_csv(d9 / "tc_members.csv", dtype={"date": str, "sid": str})
        inp = m9[m9.row == "INPUT"].set_index(KEY + ["member"])[["clat", "clon"]]
        for role, spec in (("RUN", run_spec), ("REF", ref_spec)):
            d, row = _role(spec)
            m = pd.read_csv(d / "tc_members.csv", dtype={"date": str, "sid": str})
            m = m[m.row == _label(m, row, d)].set_index(KEY + ["member"])[["clat", "clon"]]
            j = m.join(inp, rsuffix="_in", how="inner").dropna()
            j["km"] = _gc_km(j.clat, j.clon, j.clat_in, j.clon_in)
            dist[role] = j.km.groupby(level=KEY).mean()
    except (FileNotFoundError, KeyError, SystemExit) as e:
        return GateLine("G-C2", "storm centre vs INPUT centre", True, S.MISSING, note=f"tc_members.csv: {e}")
    j = pd.concat([dist["RUN"].rename("a"), dist["REF"].rename("b")], axis=1, join="inner").dropna()
    if len(j) < 3:
        return GateLine("G-C2", "storm centre vs INPUT centre", True, S.MISSING, note="fewer than 3 common cases")
    storms, inv = np.unique(j.index.get_level_values("sid"), return_inverse=True)
    n = len(storms)
    cnt = np.bincount(inv, minlength=n).astype(float)
    sa, sb = np.bincount(inv, j.a.to_numpy(), n), np.bincount(inv, j.b.to_numpy(), n)
    d, se = S.jackknife(lambda k: (sa[k].sum() - sb[k].sum()) / cnt[k].sum(), n)
    st = S.FAIL if (math.isfinite(se) and d > MARGIN["G-C2_se"] * se) else S.PASS
    return GateLine("G-C2", "storm centre vs INPUT centre", True, st, value=float(j.a.mean()), ref_value=float(j.b.mean()),
                    diff=d, unit="km", se=se, n_dates=len(j), n_blocks=n,
                    note=f"median run {j.a.median():.0f} km, REF {j.b.median():.0f} km; mean gated (a median has no "
                         f"usable jackknife SE)")


def rain_tails(run_spec: str, ref_spec: str, ens9_spec: str, rule: str = "chain") -> list[GateLine]:
    """G-C3: station twCRPS 20 and 50 mm / 12 h vs REF, leads pooled (100 mm reported only)."""
    from eval.verdict.reader_a import load_scores
    try:
        frames = {r: load_scores(str(_role(s)[0] / "obs_points" / "scores_per_date.csv"))
                  for r, s in (("RUN", run_spec), ("REF", ref_spec), ("ENS9", ens9_spec))}
    except (FileNotFoundError, SystemExit) as e:
        return [GateLine("G-C3", "twCRPS", True, S.MISSING, note=str(e))]
    out = []
    for t in (20, 50, 100):
        col = f"twcrps{t}"
        tb = joined(frames, "tp", (col,))
        if tb.empty or f"{col}_RUN" not in tb:
            out.append(GateLine("G-C3" if t < 100 else "C-info", f"tp twCRPS {t} mm", t < 100, S.MISSING, note="absent"))
            continue
        tb = tb.rename(columns={"n_REF": "n"})
        blk = per_block(tb, [f"{col}_{r}" for r in frames], "n", rule)
        nd = tb.index.get_level_values("date").nunique()
        hard = t < 100
        out.append(_diff_line("G-C3" if hard else "C-info", f"tp twCRPS {t} mm vs REF", hard, blk, f"{col}_RUN", f"{col}_REF",
                              MARGIN["G-C3"], "not_worse" if hard else "info", nd))
        out.append(_diff_line("C-info", f"tp twCRPS {t} mm vs ENS9", False, blk, f"{col}_RUN", f"{col}_ENS9", math.nan, "info", nd))
    return out


def _shares(hist: Path, label: str, var: str, thr: float):
    per = {}
    for f in sorted(Path(hist).glob(f"{label}_*_step*.npz")):
        date = f.stem.split("_")[-2]
        z = np.load(f, allow_pickle=True)
        c = z[f"{var}_counts"].sum(1).sum(0)                      # (bands, members, bins) -> bins
        e = z["tp_edges_mm" if var == "tp" else "ws_edges_ms"]
        n, d = per.get(date, (0.0, 0.0))
        per[date] = (n + float(c[e[:-1] >= thr].sum()), d + float(c.sum()))
    return per


def tails(hist: Path, run_label: str, ref_label: str, ens9_label: str, rule: str = "chain") -> list[GateLine]:
    """G-C4: global share of grid points at or above the threshold (wind 25 m/s; rain 50, 100, 200 mm / 12 h) inside
    [min(REF, ENS9) - 2 SE, max(REF, ENS9) + 2 SE], SE = the run's share jackknifed over date blocks."""
    out = []
    if not hist or not Path(hist).is_dir():
        return [GateLine("G-C4", "tail shares", True, S.MISSING, note="no histogram folder given")]
    for var, thr in (("ws", 25.0), ("tp", 50.0), ("tp", 100.0), ("tp", 200.0)):
        sh = {lab: _shares(hist, lab, var, thr) for lab in (run_label, ref_label, ens9_label)}
        common = sorted(set.intersection(*[set(v) for v in sh.values()]))
        sub = f"{var} >= {thr:g} {'m/s' if var == 'ws' else 'mm'}"
        if len(common) < 3:
            out.append(GateLine("G-C4", sub, True, S.MISSING, note="fewer than 3 common dates"))
            continue
        blk = S.blocks(common, rule)
        def arr(lab):
            g = pd.DataFrame(sh[lab]).T.reindex(common).groupby(lambda d: blk[d]).sum()
            return g[0].to_numpy(), g[1].to_numpy()
        (na, da), (nb, db), (ne, de) = arr(run_label), arr(ref_label), arr(ens9_label)
        s, se = S.jackknife(lambda k: na[k].sum() / da[k].sum(), len(na))
        sr, s9 = nb.sum() / db.sum(), ne.sum() / de.sum()
        w = MARGIN["G-C4_se"] * (se if math.isfinite(se) else 0.0)
        lo, hi = min(sr, s9) - w, max(sr, s9) + w
        st = S.PASS if lo <= s <= hi else S.FAIL
        out.append(GateLine("G-C4", sub, True, st, value=s, ref_value=sr, unit="share of grid points", se=se, lo=lo, hi=hi,
                            n_dates=len(common), n_blocks=len(na), note=f"ENS9 {s9:.3g}, REF {sr:.3g}"))
    return out
