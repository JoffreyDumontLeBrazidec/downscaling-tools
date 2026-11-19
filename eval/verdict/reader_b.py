"""Reader B: spectra, structure, consistency (note s4.3), from existing outputs only.

* G-B1, G-B2: ``box_spectra`` ``bins.csv`` files of RUN and REF, one per lead, found under a folder as
  ``*_L<LLL>/bins.csv`` (the layout of ``eval.cli spectra``) or ``*L<LLL>*/**/bins.csv``. The ratio is to the file's truth
  (ENS9 member for member-mode rows). Per-date powers = amp^2 x n_draws of the ``scope = <date>`` lines; pooled
  ratio = sqrt(sum power pred / sum power truth); jackknife over date blocks. AMPLITUDE ratios.
* G-B3, G-B5: ``final_battery`` output folders of RUN and REF (``summary.json``, else rebuilt with
  ``eval.final_battery.summary.build``). G-B5 reads line C11 (lead-to-lead rms change model/truth, +12..+72 h only).
* G-B4: ``shape`` ``rows.csv`` and ``texture`` ``texture.json`` of RUN and REF on the same box and dates.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from eval.verdict import stats as S
from eval.verdict.gates import MARGIN, GateLine

SPEC_VARS = ("ws", "tp")
SPEC_LEADS = (24, 72, 120, 240)
LEAD_LINKED = re.compile(r"^(C1|C2|C3|Z1|W7|W8)(\b|\.|$)")


# ------------------------------------------------------------------------------------------------ spectra
def find_bins(root: Path) -> dict[int, Path]:
    out = {}
    for p in sorted(Path(root).rglob("bins.csv")):
        m = re.search(r"_L(\d{2,3})(?:/|$)", str(p.parent) + "/") or re.search(r"L(\d{3})", str(p.parent))
        if m:
            out.setdefault(int(m.group(1)), p)
    return out


def _per_date_powers(df: pd.DataFrame, var: str, kind: str) -> pd.DataFrame:
    """Index (date, bin_centre_km): power of pred and truth x n_draws."""
    a, b = ("amp_resid_pred", "amp_resid_true") if kind == "residual" else ("amp_pred_full", "amp_truth_full")
    q = df[(df.variable == var) & (df.scope != "all")].copy()
    n = q.n_draws.astype(float)
    q["P"] = q[a].astype(float) ** 2 * n
    q["T"] = q[b].astype(float) ** 2 * n
    q["km"] = q.bin_centre_km.astype(float).round(2)
    return q.set_index(["scope", "km"])[["P", "T"]]


def spectra(run_root: Path, ref_root: Path, rule: str = "fixed5", leads=SPEC_LEADS) -> list[GateLine]:
    out: list[GateLine] = []
    rb, fb = find_bins(run_root), find_bins(ref_root)
    lo_km, hi_km = MARGIN["G-B1_km"]
    o_lo, o_hi = MARGIN["G-B2_km"]
    ilo, ihi = MARGIN["G-B1_info_band"]
    for L in leads:
        if L not in rb or L not in fb:
            for v in SPEC_VARS:
                out.append(GateLine("G-B1", f"{v} +{L} h", True, S.MISSING, note=f"bins.csv at +{L} h absent for "
                                    + ("RUN" if L not in rb else "REF")))
                out.append(GateLine("G-B2", f"{v} +{L} h", True, S.MISSING, note="bins.csv absent"))
            continue
        dr, df_ = pd.read_csv(rb[L], dtype={"scope": str}), pd.read_csv(fb[L], dtype={"scope": str})
        for v in SPEC_VARS:
            if v not in set(dr.variable) or v not in set(df_.variable):
                for kind in ("full", "residual"):
                    out.append(GateLine("G-B1", f"{v} {kind} +{L} h", True, S.MISSING, note="variable absent"))
                out.append(GateLine("G-B2", f"{v} +{L} h", True, S.MISSING, note="variable absent"))
                continue
            for kind in ("full", "residual"):
                pr, pf = _per_date_powers(dr, v, kind), _per_date_powers(df_, v, kind)
                common = sorted(set(pr.index.get_level_values(0)) & set(pf.index.get_level_values(0)))
                blk = S.blocks(common, rule)
                kms = sorted({k for k in pr.index.get_level_values(1) if lo_km <= k <= hi_km}, reverse=True)
                fails, info, worst = [], [], ""
                for km in kms:
                    try:
                        a = pr.xs(km, level=1).reindex(common).groupby(lambda d: blk[d]).sum()
                        b = pf.xs(km, level=1).reindex(common).groupby(lambda d: blk[d]).sum()
                    except KeyError:
                        continue
                    Pa, Ta, Pb, Tb = (x.to_numpy(np.float64) for x in (a["P"], a["T"], b["P"], b["T"]))

                    def excess(k):
                        ra = math.sqrt(Pa[k].sum() / Ta[k].sum()) if Ta[k].sum() > 0 else np.nan
                        rf = math.sqrt(Pb[k].sum() / Tb[k].sum()) if Tb[k].sum() > 0 else np.nan
                        return abs(ra - 1) - abs(rf - 1)
                    e, se = S.jackknife(excess, len(a))
                    ra = math.sqrt(Pa.sum() / Ta.sum()) if Ta.sum() > 0 else np.nan
                    bad = math.isfinite(se) and e > MARGIN["G-B1_se"] * se
                    fails.append(bad)
                    if not (ilo <= ra <= ihi):
                        info.append(f"{km:g} km {ra:.2f}")
                    if bad:
                        worst += f"{km:g} km (|r-1| excess {e:.3f}, SE {se:.3f}); "
                run_ = max((sum(1 for _ in g) for k, g in itertools.groupby(fails) if k), default=0)
                st = S.FAIL if run_ >= MARGIN["G-B1_adjacent"] else S.PASS
                if not kms:
                    st = S.MISSING
                out.append(GateLine("G-B1", f"{v} {kind} +{L} h", True, st, value=float(sum(fails)), unit="failing bins",
                                    n_dates=len(common), n_blocks=len(set(blk.values())),
                                    note=(worst + ("outside 0.9-1.1 (info): " + ", ".join(info) if info else "")).strip()))
            # G-B2 overshoot on the pooled full-field ratio
            q = dr[(dr.variable == v) & (dr.scope == "all")]
            q = q[(q.bin_centre_km.astype(float) >= o_lo) & (q.bin_centre_km.astype(float) <= o_hi)]
            r = q.ratio_full.astype(float)
            mx = float(r.max()) if len(r) else math.nan
            qr = df_[(df_.variable == v) & (df_.scope == "all")]
            qr = qr[(qr.bin_centre_km.astype(float) >= o_lo) & (qr.bin_centre_km.astype(float) <= o_hi)]
            st = S.MISSING if not len(r) else (S.FAIL if mx > MARGIN["G-B2_max"] else S.PASS)
            over = ", ".join(f"{k:g} km {x:.2f}" for k, x in zip(q.bin_centre_km.astype(float), r) if x > MARGIN["G-B2_max"])
            out.append(GateLine("G-B2", f"{v} +{L} h", True, st, value=mx,
                                ref_value=float(qr.ratio_full.astype(float).max()) if len(qr) else math.nan,
                                unit="max amplitude ratio", margin=MARGIN["G-B2_max"], note=over))
    return out


# ------------------------------------------------------------------------------------------------ battery
def battery_lines(root: Path) -> list[dict]:
    root = Path(root)
    p = root / "summary.json"
    if p.exists():
        data = json.loads(p.read_text())
    else:
        from eval.final_battery import summary
        data = {k: [ln.__dict__ for ln in v] for k, v in summary.build(root).items()}
    return [ln for lines in data.values() for ln in lines]


def battery(run_root: Path, ref_root: Path) -> list[GateLine]:
    out = []
    try:
        lr, lf = battery_lines(run_root), battery_lines(ref_root)
    except Exception as e:  # missing folder or unreadable tables
        return [GateLine("G-B3", "OUT lines", True, S.MISSING, note=f"battery not readable: {e}"),
                GateLine("G-B5", "C11 +12..+72 h", True, S.MISSING, note="battery not readable"),
                GateLine("G-B5", "+84..+240 h", True, S.MISSING, note="no lead-to-lead diagnostic on MI rows (not built)")]
    out_r = {ln["id"] for ln in lr if ln.get("status") == "OUT"}
    out_f = {ln["id"] for ln in lf if ln.get("status") == "OUT"}
    na_f = {ln["id"] for ln in lf if ln.get("status") == "n/a"}
    # a line that is n/a in the reference too (C14, T4 on GL512 450k, 19 Nov) cannot be compared: listed, not blocking
    na_r = sorted(ln["id"] for ln in lr if ln.get("status") == "n/a" and ln["id"] not in na_f)
    na_both = sorted(ln["id"] for ln in lr if ln.get("status") == "n/a" and ln["id"] in na_f)
    new = sorted(out_r - out_f)
    st = S.FAIL if (len(out_r) > len(out_f) or new) else S.PASS
    if na_r and st == S.PASS:
        st = S.INCONCLUSIVE
    linked = sorted(i for i in out_r if LEAD_LINKED.match(i))
    out.append(GateLine("G-B3", "OUT lines", True, st, value=float(len(out_r)), ref_value=float(len(out_f)), unit="OUT lines",
                        note=f"new: {', '.join(new) or 'none'}; lead-linked: {', '.join(linked) or 'none'}; "
                             f"other: {', '.join(sorted(out_r - set(linked))) or 'none'}"
                             + (f"; n/a (blocks PASS): {', '.join(na_r)}" if na_r else "")
                             + (f"; n/a in both: {', '.join(na_both)}" if na_both else "")))
    c11 = next((ln for ln in lr if ln["id"] == "C11"), None)
    c11f = next((ln for ln in lf if ln["id"] == "C11"), None)
    if c11 is None:
        out.append(GateLine("G-B5", "C11 +12..+72 h", True, S.MISSING, note="no C11 line"))
    else:
        m = re.match(r"\s*([-\d.naninf]+)-([-\d.naninf]+)", str(c11["value"]))
        lo, hi = (float(m.group(1)), float(m.group(2))) if m else (math.nan, math.nan)
        blo, bhi = MARGIN["G-B5_band"]
        st = S.MISSING if not (math.isfinite(lo) and math.isfinite(hi)) else (S.PASS if blo <= lo and hi <= bhi else S.FAIL)
        out.append(GateLine("G-B5", "C11 +12..+72 h", True, st, value=lo, ref_value=hi, unit="range min / max",
                            margin=bhi, note=f"run {c11['value']}; REF {c11f['value'] if c11f else 'n/a'}"))
    out.append(GateLine("G-B5", "+84..+240 h", True, S.MISSING,
                        note="no lead-to-lead diagnostic on MI rows yet (B1 adds no diagnostic); gate cannot PASS"))
    return out


# ------------------------------------------------------------------------------------------------ texture / shape
def _dist_line(gate, sub, a: pd.Series, b: pd.Series, rule, note=""):
    """a, b: per-date distance to truth (RUN, REF). FAIL if RUN is further from truth beyond the date scatter."""
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return GateLine(gate, sub, True, S.MISSING, note="fewer than 3 common dates")
    blk = S.blocks(common, rule)
    ga = a.reindex(common).groupby(lambda d: blk[d]).mean().to_numpy()
    gb = b.reindex(common).groupby(lambda d: blk[d]).mean().to_numpy()
    d, se = S.jackknife(lambda k: ga[k].mean() - gb[k].mean(), len(ga))
    lo, hi = S.interval(d, se, len(ga))
    st = S.FAIL if (math.isfinite(lo) and lo > 0) else S.PASS
    return GateLine(gate, sub, True, st, value=float(ga.mean()), ref_value=float(gb.mean()), diff=d,
                    unit="distance to truth", se=se, lo=lo, hi=hi, n_dates=len(common), n_blocks=len(ga), note=note)


def shape(run_csv: Path, ref_csv: Path, rule: str = "fixed5") -> list[GateLine]:
    out = []
    try:
        r, f = (pd.read_csv(p, dtype={"date": str}) for p in (run_csv, ref_csv))
    except Exception as e:
        return [GateLine("G-B4", "shape elongated fraction", True, S.MISSING, note=str(e))]
    for band in sorted(set(r.band) & set(f.band)):
        def dist(df):
            q = df[df.band == band].groupby(["date", "field"]).elong_frac_gt3.mean().unstack("field")
            return (q["draw"] - q["truth"]).abs()
        out.append(_dist_line("G-B4", f"shape elong_frac_gt3 band {band}", dist(r), dist(f), rule))
    return out or [GateLine("G-B4", "shape elongated fraction", True, S.MISSING, note="no common band")]


def texture(run_json: Path, ref_json: Path, rule: str = "fixed5") -> list[GateLine]:
    try:
        sr, sf = (json.loads(Path(p).read_text())["samples"] for p in (run_json, ref_json))
    except Exception as e:
        return [GateLine("G-B4", "texture grain index", True, S.MISSING, note=str(e))]
    out = []
    stats_ = sorted({k for s in sr for k in s.get("grain_index", {})})
    for st in stats_:
        def per_date(samples):
            d = pd.DataFrame([(s["date"], abs(s["grain_index"].get(st, np.nan))) for s in samples], columns=["date", "g"])
            return d.dropna().groupby("date").g.mean()
        out.append(_dist_line("G-B4", f"texture |grain index| {st}", per_date(sr), per_date(sf), rule,
                              note="0 = truth-like, 1 = white noise"))
    return out or [GateLine("G-B4", "texture grain index", True, S.MISSING, note="no grain index")]
