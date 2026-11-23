"""One-page summary of a final-model battery run: headline numbers beside the GL512 450k reference.

``build(out_dir)`` reads the tables the kits wrote under ``out_dir`` (layout in ``eval.final_battery.layout``) and
returns a list of :class:`Line`; ``write(out_dir, ...)`` renders ``summary.md``. Every line names its diagnostic id
(T*/Z*/D* terrain, C* consistency, W* wind), the number for this row, the GL512 450k number of 17 Nov 2026 (row
``GLB450e2_m``, 30 dates of 2023, 10 members) and the pass line of the spec where the spec gives one. A missing input
gives ``n/a`` with the reason, never a crash: the page is written whatever ran.

The numbers here are reductions of the kit tables, computed the way the 17 Nov notes quote them (checked in
``tests/test_summary_reference.py`` against the committed reference tables). The tables stay the record; this page
is the quick look.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.final_battery import layout
from eval.final_battery.references import REF

LEADS = (12, 24, 36, 48, 60, 72)


@dataclass
class Line:
    id: str
    what: str
    value: str
    ref: str = ""
    pass_line: str = ""
    status: str = ""          # "ok", "OUT", "" (no pass line) or "n/a"
    numbers: dict = field(default_factory=dict)


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{x:.{nd}f}"


def _status(values, lo=None, hi=None):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return "n/a"
    if (lo is not None and min(vals) < lo) or (hi is not None and max(vals) > hi):
        return "OUT"
    return "ok"


def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


# ------------------------------------------------------------------------------------------------ terrain
def terrain_lines(out: Path) -> list[Line]:
    import pandas as pd
    p = layout.terrain_paths(out)
    L: list[Line] = []

    def na(i, what, why):
        L.append(Line(i, what, f"n/a ({why})", REF.get(i, ""), status="n/a"))

    t1p = p["tables"] / "T1_bias_rmse_binned.csv"
    if t1p.exists():
        t1 = pd.read_csv(t1p)
        m = t1[(t1.source == "model")]
        fl = m[m.bias_flag.astype(str).str.lower() == "true"]
        by = fl.groupby("var").size().to_dict()
        L.append(Line("T1", "pre-registered bias flags (|bias| > 2 x floor and >= 20/30 dates), model rows",
                      f"{len(fl)} ({', '.join(f'{k} {v}' for k, v in sorted(by.items())) or 'none'})", REF["T1"],
                      numbers={"flagged": int(len(fl)), "by_var": by}))
        r = m[(m["var"] == "sp") & (m.lead == 24) & (m.scheme == "zhr_band") & (m.bin == "2000-3000")]
        if len(r):
            r = r.iloc[0]
            L.append(Line("T1b", "sp bias, land 2000-3000 m, +24 h (Pa, dates same sign)",
                          f"{r.bias_mean:+.1f} ({int(max(r.bias_npos, r.n_dates - r.bias_npos))}/{int(r.n_dates)})", REF["T1b"],
                          numbers={"bias": float(r.bias_mean)}))
        rows = []
        for b in ("land0..10", "inland_water_proxy"):
            q = t1[(t1["var"] == "2t") & (t1.lead == 0) & (t1.scheme == "coast_km") & (t1.bin == b) & (t1.source == "model")]
            if len(q):
                q = q.iloc[0]
                rows.append((b, q.rmse_mean / q.floor_rmse_mean))
        if rows:
            L.append(Line("T1c", "2t RMSE / pf-pair floor, all leads: coast land 0-10 km, inland water",
                          " / ".join(_f(v) for _, v in rows), REF["T1c"], numbers=dict(rows)))
    else:
        na("T1", "bias / RMSE vs floor", "no T1 table")

    t2p = p["tables"] / "T2_lapse_slopes.csv"
    if t2p.exists():
        t2 = pd.read_csv(t2p)
        e = t2[t2["var"] == "2t"].dropna(subset=["slope_model", "slope_truth_ctrl"])
        c = float(np.corrcoef(e.slope_model, e.slope_truth_ctrl)[0, 1]) if len(e) > 2 else float("nan")
        L.append(Line("T2", f"2t effective lapse slope, corr model vs truth control ({len(e)} date-lead-regions)",
                      _f(c, 3), REF["T2"], "> 0.95", _status([c], lo=0.95), {"corr": c, "n": len(e)}))
        inv = t2[(t2["var"] == "2t") & (t2.slope_truth_ctrl > -2)].dropna(subset=["slope_model", "slope_truth_pf_unpaired"])
        if len(inv):
            d = float((inv.slope_model - inv.slope_truth_pf_unpaired).mean())
            L.append(Line("T2b", f"inversions (truth > -2 K/km, {len(inv)} cases, {inv.date.nunique()} dates): model - truth pf (K/km)",
                          f"{d:+.2f}", REF["T2b"], numbers={"diff": d, "n": len(inv)}))
    d2p = p["d2"] / "D2_lapse_2d_skt.csv"
    if d2p.exists():
        d2 = pd.read_csv(d2p)
        cs = {}
        for v in ("2d", "skt"):
            e = d2[d2["var"] == v].dropna(subset=["slope_model", "slope_truth_ctrl"])
            cs[v] = float(np.corrcoef(e.slope_model, e.slope_truth_ctrl)[0, 1]) if len(e) > 2 else float("nan")
        L.append(Line("D2", "2d / skt lapse slope corr, model vs truth control", f"{_f(cs['2d'], 3)} / {_f(cs['skt'], 3)}",
                      REF["D2"], numbers=cs))

    t3p = p["tables"] / "T3_hydrostatic_summary.csv"
    if t3p.exists():
        t3 = pd.read_csv(t3p)
        g = t3.groupby(["source", "metric"])["mean"].mean()
        try:
            r = g[("model", "hp_r_std_mtn")] / g[("truth_pf", "hp_r_std_mtn")]
            msl = (g[("model", "hp_msl_std_mtn")], g[("truth_pf", "hp_msl_std_mtn")])
            L.append(Line("T3", "hydrostatic residual high-pass std over z > 500 m, model / truth pf; msl hp std mountains model vs truth (Pa)",
                          f"{_f(r)}; {msl[0]:.1f} vs {msl[1]:.1f}", REF["T3"], "ratio <= 1.05", _status([r], hi=1.05),
                          {"ratio": float(r), "msl_model": float(msl[0]), "msl_truth": float(msl[1])}))
        except KeyError:
            na("T3", "hydrostatic consistency", "metrics missing")

    d4p = p["d2"] / "D4_2d_vs_2t.csv"
    if d4p.exists():
        d4 = pd.read_csv(d4p)
        q = d4[(d4.source == "model") & (d4.surface == "land") & (d4.metric == "2d_gt_2t+0.1")].sort_values("lead")
        t = d4[(d4.source == "truth_ctrl") & (d4.surface == "land") & (d4.metric == "2d_gt_2t+0.1")]
        vals = {int(r.lead): 100 * r["mean"] for _, r in q.iterrows()}
        tv = 100 * float(t["mean"].max()) if len(t) else float("nan")
        L.append(Line("T4", "2d > 2t + 0.1 K, share of land points (%) by lead; truth control max",
                      ", ".join(f"+{k} h {v:.3f}" for k, v in vals.items()) + f"; truth {tv:.3f}", REF["T4"], "0",
                      _status(list(vals.values()), hi=0.001), {"by_lead": vals}))
    else:
        L.append(Line("T4", "2d > 2t bound (needs 2d, skt, tcw saved)", "n/a (no d2 tables: row lacks 2d/skt/tcw?)",
                      REF["T4"], "0", "n/a"))

    t5p = p["tables"] / "T5_o320_imprint.csv"
    if t5p.exists():
        t5 = pd.read_csv(t5p)
        a = t5[(t5.lead == 0) & (t5.subset == "land")].pivot_table(index="d320_km", columns="quantity", values="mean")
        if {"abs_hp_model", "abs_hp_truth_pf"} <= set(a.columns):
            dd = (a.abs_hp_model - a.abs_hp_truth_pf)
            spread = float(dd.max() - dd.min())
            L.append(Line("T5", "input-grid imprint: range over distance-to-O320 bins of mean|hp 2t| model - truth pf, land (K)",
                          _f(spread, 3), REF["T5"], "< 0.03", _status([spread], hi=0.03), {"range": spread}))

    t6p = p["tables"] / "T6_tails.csv"
    if t6p.exists():
        t6 = pd.read_csv(t6p)
        w = t6[(t6.region == "box_land") & (t6.field == "hp")].pivot_table(index=["date", "lead"], columns="source", values="q99.99")
        if {"model", "truth_pf"} <= set(w.columns):
            d = (w.model - w.truth_pf).dropna()
            L.append(Line("T6", "2t high-pass q99.99 over box land, model - truth pf (K; cases below)",
                          f"{d.mean():+.2f} ({int((d < 0).sum())}/{len(d)})", REF["T6"], numbers={"mean": float(d.mean())}))

    t7p = p["tables"] / "T7_large_scale.csv"
    if t7p.exists():
        t7 = pd.read_csv(t7p)
        q = t7[(t7["var"] == "msl") & (t7.subset == "interior")]
        mm = q[q.source == "model_minus_input"].set_index("lead").lowpass_rms
        tc = q[q.source == "truth_ctrl_minus_input_ctrl"].set_index("lead").lowpass_rms
        v12 = [mm.get(x, np.nan) for x in (12, 36, 60)]
        v00 = [mm.get(x, np.nan) for x in (24, 48, 72)]
        L.append(Line("T7", "msl large-scale move of draw from own input (Pa): 12 UTC-valid / 00 UTC-valid leads; truth control +12 -> +72 h",
                      f"{np.nanmean(v12):.0f} / {np.nanmean(v00):.0f}; truth {tc.get(12, np.nan):.0f} -> {tc.get(72, np.nan):.0f}",
                      REF["T7"], numbers={"m12": float(np.nanmean(v12)), "m00": float(np.nanmean(v00))}))

    z1p = p["z"] / "Z1_spread_by_scale.csv"
    if z1p.exists():
        z1 = pd.read_csv(z1p)
        for i, (v, mask, part, what) in {
            "Z1a": ("msl", "sea", "low", "msl low-pass spread ratio, sea"),
            "Z1b": ("10ff", "sea", "low", "10 m wind low-pass spread ratio, sea"),
            "Z1c": ("2t", "land_lt500", "low", "2t low-pass spread ratio, land < 500 m"),
        }.items():
            q = z1[(z1["var"] == v) & (z1["mask"] == mask)].set_index("lead")[f"ratio_{part}_mean"]
            vals = [q.get(x, np.nan) for x in LEADS]
            L.append(Line(i, f"{what}, model / truth pf, +12..+72 h", " ".join(_f(x) for x in vals), REF[i], "0.9-1.1",
                          _status(vals, 0.9, 1.1), {"by_lead": vals}))
        q = z1[(z1["var"] == "sp") & (z1["mask"] == "land_gt1500")].set_index("lead")
        if len(q):
            vals = [q.ratio_high_mean.get(x, np.nan) for x in LEADS]
            L.append(Line("Z1d", "sp high-pass spread over land > 1500 m: ratio model / truth pf by lead (model std, Pa)",
                          " ".join(_f(x, 1) for x in vals) + f" ({q.std_high_model.mean():.0f} Pa)", REF["Z1d"], "0.9-1.1",
                          _status(vals, 0.9, 1.1), {"by_lead": vals}))
    z4p = p["z"] / "Z4_2t_spread_coast_dz.csv"
    if z4p.exists():
        z4 = pd.read_csv(z4p)
        q = z4[(z4.scheme == "coast_km") & (z4.lead == 12)].set_index("bin").ratio_mean
        a, b = q.get("inland_water_proxy", np.nan), q.get("land0..10", np.nan)
        L.append(Line("Z4", "2t spread ratio model / truth pf at +12 h: inland water / coast land 0-10 km", f"{_f(a)} / {_f(b)}",
                      REF["Z4"], "0.9-1.1", _status([a, b], 0.9, 1.1), {"inland": float(a), "coast": float(b)}))
    z3p = p["z"] / "Z3_sp_diagnosed.csv"
    if z3p.exists():
        z3 = pd.read_csv(z3p)
        q = z3[(z3.lead == 0) & (z3.zhr_band == "2000-3000")].set_index("kind").rmse_or_spread
        if len(q):
            L.append(Line("Z3", "sp RMSE 2000-3000 m, all leads (Pa): direct / diagnosed calibrated / floor",
                          " / ".join(f"{q.get(k, np.nan):.0f}" for k in ("direct", "diag_cal", "floor")), REF["Z3"]))
    z6p = p["z6"] / "Z6_sp_residual_baseline.csv"
    if z6p.exists():
        z6 = pd.read_csv(z6p)
        hp = z6[z6.part == "hp"].set_index(["residual", "zhr_band"])["std"]
        try:
            vals = [hp[("b_model_draws", b)] / hp[("b", b)] for b in ("1500-2000", "2000-3000", "3000+")]
            L.append(Line("Z6", "sp residual on the hydrostatic baseline, high-pass std model / truth (1500-2000, 2000-3000, 3000+ m)",
                          " / ".join(_f(x) for x in vals), REF["Z6"], "<= 1.1", _status(vals, hi=1.1), {"ratios": vals}))
        except KeyError:
            L.append(Line("Z6", "sp residual on the hydrostatic baseline", "see Z6 table (model rows not found by name)", REF["Z6"]))
    dhp = p["d2"] / "DH_fine_amplitude.csv"
    if dhp.exists():
        dh = pd.read_csv(dhp)
        w = dh.groupby(["var", "source"]).mean_abs_hp.mean().unstack("source")
        if {"model", "truth_ctrl"} <= set(w.columns):
            r = (w.model / w.truth_ctrl - 1).abs()
            L.append(Line("DH", "fine-scale amplitude mean|hp| model / truth control, worst of 2t 2d skt (|ratio - 1|)",
                          f"{r.max() * 100:.1f} % ({r.idxmax()})", REF["DH"], "<= 6 %", _status([r.max()], hi=0.06),
                          {"by_var": r.to_dict()}))
    rc = p["root"] / "repeat_check.txt"
    if rc.exists():
        import re
        rs = [abs(float(x)) for x in re.findall(r"parts \d-\d r=([+-][0-9.]+)", rc.read_text())]
        if rs:
            L.append(Line("B11", "sharded-noise repeat check: max |r| of hp(draw - input) msl between point-array parts",
                          _f(max(rs), 4), REF["B11"], "< 0.01", _status([max(rs)], hi=0.01), {"max_r": max(rs)}))
    return L


# ------------------------------------------------------------------------------------------------ consistency
def _ratios(path: Path):
    """ratios.csv with its three colliding 'se' headers renamed by position."""
    import pandas as pd
    t = pd.read_csv(path, header=0)
    t.columns = ["key", "lead", "mt", "mt_se", "it", "it_se", "mi", "mi_se", "n_dates"][: len(t.columns)]
    t["lead"] = t["lead"].astype(str)
    return t.set_index(["key", "lead"])


def consistency_lines(out: Path) -> list[Line]:
    import pandas as pd
    L: list[Line] = []
    c = layout.consistency_paths(out)
    rp = c["edge1"] / "ratios.csv"
    if not rp.exists():
        return [Line("C", "consistency battery", "n/a (no ratios.csv)", status="n/a")]
    R = _ratios(rp)
    S = pd.read_csv(c["edge1"] / "scalars.csv", dtype={"lead": str}).set_index(["key", "lead"])
    lk = [f"{x:03d}" for x in LEADS]

    def r(key, lead="all", col="mt"):
        try:
            return float(R.loc[(key, lead), col])
        except KeyError:
            return float("nan")

    for v in ("tp", "msl", "ws", "2t"):
        vals = [math.sqrt(r(f"S|{v}|large|var", x)) for x in lk]
        L.append(Line(f"C1.{v}", f"large-scale (> 400 km) spread sd ratio model / truth, {v}, +12..+72 h", " ".join(_f(x) for x in vals),
                      REF[f"C1.{v}"], "0.9-1.1", _status(vals, 0.9, 1.1), {"by_lead": vals}))
    e5 = c["edge5"] / "ratios.csv"
    if e5.exists():
        R5 = _ratios(e5)
        vals = [math.sqrt(float(R5.loc[(f"S|tp|large|var", x), "mt"])) if (f"S|tp|large|var", x) in R5.index else float("nan") for x in lk]
        L.append(Line("C1.tp5", "same for tp with a 5-degree edge mask", " ".join(_f(x) for x in vals), REF["C1.tp5"], "0.9-1.1",
                      _status(vals, 0.9, 1.1), {"by_lead": vals}))
    for i, (key, what) in {"C2.tp": ("S|tp|fine|class|land_gt1000|var", "tp, land > 1000 m"),
                           "C2.2t": ("S|2t|fine|class|coast|var", "2t, coast"),
                           "C2.ws": ("S|ws|fine|class|coast|var", "ws, coast"),
                           "C2.sea": ("S|2t|fine|class|sea|var", "2t, sea")}.items():
        a, b = math.sqrt(r(key, "012")), math.sqrt(r(key, "072"))
        L.append(Line(i, f"fine (< 125 km) spread sd ratio, {what}: +12 h -> +72 h", f"{_f(a)} -> {_f(b)}", REF[i], "0.9-1.1",
                      _status([a, b], 0.9, 1.1), {"L012": a, "L072": b}))
    x = r("T|tp|bF|lag_corr")
    L.append(Line("C3", "34-125 km rain lag correlation between 12 h windows, model / truth", _f(x), REF["C3"], "> 0.8",
                  _status([x], lo=0.8), {"ratio": x}))
    vals = [r("P|rain_objects_ge1|count"), r("P|rain_objects_ge1|mean_cells"), r("P|rain_objects_ge10|count"), r("P|rain_objects_ge10|mean_cells")]
    L.append(Line("C4", "rain objects >= 1 mm count / size, >= 10 mm count / size (model / truth)", " / ".join(_f(v, 3) for v in vals),
                  REF["C4"], "0.95-1.05", _status(vals, 0.95, 1.05), {"vals": vals}))
    eb = [math.sqrt(r(f"E|tp|fine_var|bin{i}")) for i in range(8)]
    L.append(Line("C5", "tp fine amplitude model / truth by edge distance (0-0.25 .. > 5 deg); sets the mask", " ".join(_f(v) for v in eb),
                  REF["C5"], numbers={"bins": eb}))
    ap = c["edge1"] / "arrays.npz"
    if not ap.exists():
        ap = c["edge1"] / "arrays_small.npz"
    if ap.exists():
        A = np.load(ap, allow_pickle=True)
        if "model|P|tp_hist" in A.files:
            h = {s: A[f"{s}|P|tp_hist"].astype(float) for s in ("model", "truth")}
            h = {s: v / v.sum() for s, v in h.items()}
            neg = 100 * h["model"][:3].sum()
            L.append(Line("C6", "raw tp: share < -0.01 mm (%); dry |tp| < 0.01 mm model vs truth (%); 0.01-0.03 mm model vs truth (%)",
                          f"{neg:.1f}; {100 * h['model'][3]:.1f} vs {100 * h['truth'][3]:.1f}; {100 * h['model'][4]:.1f} vs {100 * h['truth'][4]:.1f}",
                          REF["C6"], "neg 0 after clip; dry within 2 points",
                          _status([abs(h['model'][3] - h['truth'][3]) * 100], hi=2.0), {"neg_pct": neg}))
    x = r("P|hypso_rms|sea")
    L.append(Line("C7", "msl-sp hypsometric rms over sea, model / truth", _f(x), REF["C7"], numbers={"ratio": x}))
    g = {k: (S.loc[(f"model|P|geo|{k}", "all"), "mean"], S.loc[(f"truth|P|geo|{k}", "all"), "mean"])
         for k in ("median_ws_over_vg", "median_turning_deg", "fine_wind_pressure_corr") if (f"model|P|geo|{k}", "all") in S.index}
    if g:
        L.append(Line("C8", "geostrophic relation, sea 30-65 N: ws/Vg, turning (deg), fine coupling (model vs truth)",
                      "; ".join(f"{a:.3g} vs {b:.3g}" if abs(b) < 1 else f"{a:.2f} vs {b:.2f}" for a, b in g.values()), REF["C8"], "within 3 %",
                      _status([abs(a / b - 1) for a, b in g.values()], hi=0.03)))
    vals = [r("P|div_std|sea"), r("P|vort_std|sea"), r("P|conv_rain_corr_large")]
    L.append(Line("C9", "div std, vort std (sea), large-scale convergence-rain corr: model / truth", " / ".join(_f(v) for v in vals),
                  REF["C9"], "0.95-1.05", _status(vals, 0.95, 1.05)))
    pairs = []
    for k in ("P|conv_share_volume", "P|tcw|rain_volume_share_tcw_lt15"):
        try:
            pairs.append((float(S.loc[(f"model|{k}", "all"), "mean"]), float(S.loc[(f"truth|{k}", "all"), "mean"])))
        except KeyError:
            pairs.append((float("nan"), float("nan")))
    L.append(Line("C10", "convective volume share; rain volume share below 15 kg/m2 tcw (model vs truth, 30-date means)",
                  "; ".join(f"{a:.3f} vs {b:.3f}" for a, b in pairs), REF["C10"], "within 5 %",
                  _status([abs(a / b - 1) for a, b in pairs if b], hi=0.05)))
    rc = [r(f"T|{v}|b{b}|rms_change") for v in ("ws", "2t", "msl") for b in "FML"]
    mm = r("T|msl_min|abs_change")
    L.append(Line("C11", "lead-to-lead rms change model / truth (ws 2t msl x bands F M L): range; box msl_min change",
                  f"{np.nanmin(rc):.2f}-{np.nanmax(rc):.2f}; {_f(mm)}", REF["C11"], "0.9-1.1", _status(rc, 0.9, 1.1)))
    sp = c["edge1"] / "stationary.csv"
    if sp.exists():
        st = pd.read_csv(sp)
        q = st[st.key == "model_truth|ws|grain_mean_corr"]
        if len(q):
            v = float(q.value.min())
            L.append(Line("C12", "stationary grain pattern corr model vs truth, ws (worst class)", _f(v, 3), REF["C12"], "> 0.95",
                          _status([v], lo=0.95)))
    try:
        cc = float(S.loc[("model_truth|S|ws|fine|corr_local_sd", "all"), "mean"])
        cv = r("S|tp|fine|cv_local_sd")
        L.append(Line("C13", "spread-map corr model vs truth (ws fine); CV of local sd model / truth (tp fine)", f"{cc:.2f}; {_f(cv, 3)}", REF["C13"]))
    except KeyError:
        pass
    keys = [k for k in S.index.get_level_values(0).unique() if k.startswith("model|P|") and ("2d" in k or "tcc" in k)]
    L.append(Line("C14", "2d / tcw / tcc bounds (on rows that save them)",
                  "; ".join(f"{k.split('|', 2)[2]} {S.loc[(k, 'all'), 'mean']:.3g}" for k in keys[:4]) or "n/a (row lacks 2d / tcc)",
                  REF["C14"], "0", "" if keys else "n/a"))
    return L


# ------------------------------------------------------------------------------------------------ wind
def _jk(per_date_num, per_date_den):
    """ratio sqrt(sum num / sum den) and its delete-one-date jackknife SE."""
    n, d = np.asarray(per_date_num, float), np.asarray(per_date_den, float)
    r = math.sqrt(n.sum() / d.sum())
    k = len(n)
    if k < 3:
        return r, float("nan")
    loo = np.sqrt((n.sum() - n) / (d.sum() - d))
    return r, float(math.sqrt((k - 1) / k * ((loo - loo.mean()) ** 2).sum()))


def wind_lines(out: Path) -> list[Line]:
    import pandas as pd
    L: list[Line] = []
    w = layout.wind_paths(out)
    for lead, base in sorted(w["leads"].items()):
        sm = base / "small"
        bt = sm / "band_table.csv"
        tag = f"+{lead} h"
        if bt.exists():
            t = pd.read_csv(bt)
            q = t[(t["var"] == "wind speed") & (t.band == "28_45") & (t.src == "model")].set_index("cls").amp_ratio
            cl = ["open sea", "coast / mixed", "flat land", "mountain land"]
            vals = [q.get(k, np.nan) for k in cl]
            spread = float(np.nanmax(vals) - np.nanmin(vals))
            L.append(Line(f"W2.{lead}", f"{tag} ws amplitude ratio 28-45 km: open sea / coast / flat land / mountain (classes must agree)",
                          " / ".join(_f(v, 3) for v in vals), REF.get(f"W2.{lead}", ""), "range < 0.03", _status([spread], hi=0.03),
                          {"by_class": dict(zip(cl, vals))}))
        cd = sm / "class_dates.csv"
        if cd.exists():
            t = pd.read_csv(cd)
            g = t.groupby(["cls", "src"])[["mean_ws", "speckle_px"]].sum()
            cl = ["open sea", "coast / mixed", "flat land", "mountain land"]
            try:
                bias = []
                for k in cl:
                    # mean_ws columns are sums over tiles of per-tile pixel means; normalise by the truth's own sum
                    bias.append(g.loc[(k, "model"), "mean_ws"] / g.loc[(k, "truth"), "mean_ws"] - 1)
                sk = [g.loc[(k, "model"), "speckle_px"] / max(g.loc[(k, "truth"), "speckle_px"], 1e-9) for k in cl]
                L.append(Line(f"W4.{lead}", f"{tag} speckle pixels model / truth by class (sea, coast, flat, mountain)",
                              " / ".join(_f(v) for v in sk), REF.get(f"W4.{lead}", ""), "<= 1.05", _status(sk, hi=1.05),
                              {"by_class": dict(zip(cl, sk))}))
                L.append(Line(f"W3.{lead}", f"{tag} mean wind speed model / truth - 1 by class (%)",
                              " / ".join(f"{100 * v:+.1f}" for v in bias), REF.get(f"W3.{lead}", ""), "within 3 %",
                              _status([abs(v) for v in bias], hi=0.03)))
                dv = {s: t[(t.cls != "edge") & (t.src == s)][["div_pow_28_45", "vort_pow_28_45"]].sum() for s in ("model", "truth")}
                pm = dv["model"].div_pow_28_45 / dv["model"].vort_pow_28_45
                pt = dv["truth"].div_pow_28_45 / dv["truth"].vort_pow_28_45
                L.append(Line(f"W5.{lead}", f"{tag} divergence / vorticity power at 28-45 km, model vs truth", f"{pm:.2f} vs {pt:.2f}",
                              REF.get("W5", ""), "within 5 %", _status([abs(pm / pt - 1)], hi=0.05)))
            except KeyError:
                pass
        sp = next((x for x in (base / "spread_dates.csv", sm / "spread_dates.csv") if x.exists()), None)
        if sp is not None:
            s = pd.read_csv(sp)
            parts = []
            vals = {}
            for v in ("ws", "msl"):
                p = s[s["var"] == v].pivot_table(index="date", columns="src", values="b400_9999")
                rr, se = _jk(p.model, p.truth)
                vals[v] = rr
                parts.append(f"{v} {rr:.2f} ({se:.2f})")
            L.append(Line(f"W7.{lead}", f"{tag} member spread > 400 km, sd ratio model / truth (jackknife SE)", ", ".join(parts),
                          REF.get(f"W7.{lead}", ""), "0.9-1.1", _status(list(vals.values()), 0.9, 1.1), vals))
        cs = next((x for x in (base / "cases.csv", base / "cases.csv.gz") if x.exists()), None)
        if cs is not None:
            c = pd.read_csv(cs)
            g = c.groupby(["case", "date", "kind", "src"]).pmin.std().unstack("src").reset_index()
            parts, vals = [], {}
            for kind, name in (("xt", "extratropical"), ("tc", "tropical")):
                d = g[g.kind == kind].dropna(subset=["model", "truth"])
                d = d[(d.model > 0) & (d.truth > 0)]
                if len(d) < 3:
                    continue
                lr = np.log(d.model / d.truth)
                se = math.sqrt(((lr.groupby(d.date).sum() - lr.groupby(d.date).count() * lr.mean()) ** 2).sum()) / len(lr)
                vals[kind] = math.exp(lr.mean())
                parts.append(f"{name} {vals[kind]:.2f} (log SE {se:.2f}, {len(d)} lows)")
            L.append(Line(f"W8.{lead}", f"{tag} member SD of storm central pressure, model / truth", "; ".join(parts) or "n/a",
                          REF.get(f"W8.{lead}", ""), "0.9-1.1", _status(list(vals.values()), 0.9, 1.1), vals))
    xs = w["xset"]
    if xs.exists():
        L.append(Line("W9", "XSET-1 hurricane structure by best-track class", f"members table {xs.name} written; read with the note s3.2 recipe",
                      REF["W9"]))
    return L


# ------------------------------------------------------------------------------------------------ page
def build(out: Path) -> dict[str, list[Line]]:
    out = Path(out)
    return {"terrain": terrain_lines(out), "consistency": consistency_lines(out), "wind": wind_lines(out)}


def render(sections: dict[str, list[Line]], meta: dict) -> str:
    title = {"terrain": "Terrain and surface (T, Z, D)", "consistency": "Ensemble and physical consistency (C)",
             "wind": "Wind and cyclones (W)"}
    out = [f"# Final-model battery: {meta.get('final_model', '?')} (row {meta.get('row', '?')})", "",
           f"Run {meta.get('when', '')}; predictions {meta.get('pred_root', '')}; {meta.get('n_dates', '?')} dates, leads "
           f"{meta.get('leads', '')}. Reference column = GL512 450k EMA-2000, row GLB450e2_m, 17 Nov 2026 (same 30 dates). "
           "OUT = outside the spec's pass line; blank status = no pass line in the spec (report only). Not a score: a realism audit.",
           ""]
    for k, lines in sections.items():
        if not lines:
            continue
        out += [f"## {title.get(k, k)}", "", "| id | diagnostic | this row | GL512 450k | pass line | status |", "|---|---|---|---|---|---|"]
        for ln in lines:
            out.append(f"| {ln.id} | {ln.what} | {ln.value} | {ln.ref} | {ln.pass_line} | {ln.status} |")
        out.append("")
    nout = sum(ln.status == "OUT" for lines in sections.values() for ln in lines)
    nna = sum(ln.status == "n/a" for lines in sections.values() for ln in lines)
    out += [f"Lines outside the pass line: {nout}. Lines not available: {nna}.", ""]
    return "\n".join(out)


def write(out: Path, meta: dict | None = None) -> Path:
    out = Path(out)
    meta = dict(meta or {})
    mp = out / "battery_meta.json"
    if mp.exists():
        meta = {**json.loads(mp.read_text()), **meta}
    sections = build(out)
    page = out / "summary.md"
    page.write_text(render(sections, meta))
    (out / "summary.json").write_text(json.dumps(
        {k: [ln.__dict__ for ln in v] for k, v in sections.items()}, indent=1, default=float))
    return page
