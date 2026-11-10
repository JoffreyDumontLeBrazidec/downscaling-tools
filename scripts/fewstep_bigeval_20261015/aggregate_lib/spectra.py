"""Part 2: amplitude spectra per bin, never averaged over a band.

Box spectra, per row and view: ``evaluators/box_spectra/bins.csv`` (columns scope, variable,
bin_lo_km, bin_hi_km, bin_centre_km, amp_resid_pred, amp_resid_true, ratio_resid,
amp_pred_full, amp_truth_full, ratio_full, amp_interp, ratio_interp_to_truth, coherence,
aligned, n_draws; scope "all" is pooled over dates and members, scope YYYYMMDD one date) and
``run_meta.json`` (box). Global spectra: ``evaluators/spectra_coherence/curves.csv`` (columns
scope, state, ell, wavelength_km, n_samples, amplitude_ratio, coherence, aligned,
residual_amplitude_ratio, residual_coherence, interp_amplitude_ratio, interp_coherence, P_true,
P_pred, P_interp, P_resid_pred, P_resid_true; scope "all" pooled, scope YYYYMMDD with
``per_date: true``), else ``coherence.json`` (``curves[state]``: ell, P_true, P_pred,
coherence, P_interp; pooled only, no residual powers). The per-degree powers are summed inside
narrow log-spaced degree bins before the ratio is formed (the same operation box_spectra does
inside its wavenumber bins); every bin is reported, none is averaged with another, and the
per-degree values are written in full to ``csv/spectra_global_<view>_per_degree.csv``.

Against the reference (pre-registered rule): per bin, the per-date difference of the row's
amplitude ratio and the reference's, mean over dates, t = mean / SE; a bin is off the reference
when |mean| > 0.03 and |t| >= 2.03. Where the tool gives no per-date values (coherence.json only),
the 0.03 rule alone, and the table says so. A row passes when no bin of 10u, 10v or 2t is off,
full field or residual. Separately every bin outside 0.97-1.03 of the truth is listed.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .common import (Context, SectionResult, fmt, fnum, guarded, mean_se, na, read_csv,
                     read_json, table)

BOX_QUANTS = [("ratio_full", "full field / truth"), ("ratio_resid", "residual / truth residual"),
              ("ratio_interp_to_truth", "input / truth"), ("coherence", "coherence"),
              ("aligned", "aligned part (coherence x full ratio)")]
DIFF_QUANTS = ["ratio_full", "ratio_resid", "aligned"]
GLOBAL_QUANTS = [("ratio_full", "full field / truth"), ("ratio_interp_to_truth", "input / truth"),
                 ("coherence", "coherence"), ("aligned", "aligned part (coherence x full ratio)")]
GATE_VARS = ("10u", "10v", "2t")
EARTH_CIRC_KM = 2 * math.pi * 6371.0


def _bin_key(km: float) -> str:
    return f"{km:.1f}"


def _bin_labels(kms: list[float]) -> list[str]:
    """Column names of the wide tables: whole km, or more decimals when two bins would collide."""
    for nd in (0, 1, 2, 3):
        labels = [f"{km:.{nd}f} km" for km in kms]
        if len(set(labels)) == len(labels):
            return labels
    return [f"{km!r} km" for km in kms]


# --------------------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------------------

def read_box(found) -> dict:
    """{"pooled": {(var, bin): row}, "dated": {(var, bin): {date: row}}, "box": ..., "bins": {bin: km}}"""
    pooled, dated, bins = {}, defaultdict(dict), {}
    for r in read_csv(found.path / "bins.csv"):
        km = fnum(r["bin_centre_km"])
        if not math.isfinite(km):
            continue
        b = _bin_key(km)
        bins[b] = km
        rec = {k: fnum(r.get(k)) for k, _ in BOX_QUANTS}
        rec["bin_lo_km"], rec["bin_hi_km"] = fnum(r.get("bin_lo_km")), fnum(r.get("bin_hi_km"))
        if r["scope"] == "all":
            pooled[(r["variable"], b)] = rec
        else:
            dated[(r["variable"], b)][r["scope"]] = rec
    box = None
    meta = found.path / "run_meta.json"
    if meta.exists():
        try:
            box = read_json(meta).get("box")
        except Exception:  # noqa: BLE001
            box = None
    return {"pooled": pooled, "dated": dict(dated), "box": box, "bins": bins}


def ladder_edges(lmax: int, lmin: int = 20, nbins: int = 30) -> np.ndarray:
    edges = np.unique(np.round(np.geomspace(lmin, lmax + 1, nbins + 1)).astype(int))
    return edges


def _bin_powers(ell: np.ndarray, cols: dict[str, np.ndarray], edges: np.ndarray) -> list[tuple]:
    """[(lo, hi, rec)] per degree bin: powers summed in the bin, then the ratios.

    ``cols``: P_true, P_pred, coherence (per degree), optionally P_interp, P_resid_pred,
    P_resid_true. The cross power is coherence x sqrt(P_pred P_true) per degree."""
    pt, pp = cols["P_true"], cols["P_pred"]
    x = cols["coherence"] * np.sqrt(np.maximum(pp * pt, 0.0))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ell >= lo) & (ell < hi) & np.isfinite(pt) & np.isfinite(pp)
        if not sel.any():
            continue
        spt, spp = float(pt[sel].sum()), float(pp[sel].sum())
        if spt <= 0:
            continue
        r = math.sqrt(spp / spt)
        cc = float(np.nansum(x[sel])) / math.sqrt(spp * spt) if spp > 0 else math.nan
        rec = {"ratio_full": r, "coherence": cc, "aligned": cc * r,
               "ratio_interp_to_truth": math.nan, "ratio_resid": math.nan,
               "ell_lo": int(lo), "ell_hi": int(hi - 1)}
        if "P_interp" in cols:
            rec["ratio_interp_to_truth"] = math.sqrt(float(np.nansum(cols["P_interp"][sel])) / spt)
        if "P_resid_pred" in cols and "P_resid_true" in cols:
            rt = float(np.nansum(cols["P_resid_true"][sel]))
            if rt > 0:
                rec["ratio_resid"] = math.sqrt(float(np.nansum(cols["P_resid_pred"][sel])) / rt)
        out.append((int(lo), int(hi), rec))
    return out


def _wavelength_km(l: float) -> float:
    return EARTH_CIRC_KM / math.sqrt(l * (l + 1.0))


def read_global(found, *, lmin: int = 20, nbins: int = 30) -> dict:
    """{"pooled": {(state, bin): rec}, "dated": {(state, bin): {date: rec}}, "bins": {bin: km},
    "per_degree": [...], "source": ..., "has_resid": bool}.

    Prefers ``curves.csv`` (per-degree powers incl. the residual, scope all and per date);
    falls back on ``coherence.json`` (pooled only, no residual powers)."""
    per_scope: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    curves = found.path / "curves.csv"
    if curves.exists():
        source = "curves.csv"
        for r in read_csv(curves):
            d = per_scope[(r["scope"], r["state"])]
            d["ell"].append(fnum(r["ell"]))
            for k in ("P_true", "P_pred", "coherence", "P_interp", "P_resid_pred", "P_resid_true",
                      "amplitude_ratio", "residual_amplitude_ratio", "interp_amplitude_ratio"):
                d[k].append(fnum(r.get(k)))
    else:
        source = "coherence.json"
        data = read_json(found.path / "coherence.json")
        for state, c in (data.get("curves") or {}).items():
            d = per_scope[("all", state)]
            d["ell"] = [float(v) for v in c["ell"]]
            for k in ("P_true", "P_pred", "coherence", "P_interp", "amplitude_ratio",
                      "interp_amplitude_ratio", "residual_amplitude_ratio"):
                if k in c:
                    d[k] = [fnum(v) for v in c[k]]
    pooled, dated, bins, per_degree = {}, defaultdict(dict), {}, []
    has_resid = False
    for (scope, state), d in per_scope.items():
        ell = np.asarray(d["ell"], dtype=float)
        cols = {k: np.asarray(v, dtype=float) for k, v in d.items()
                if k != "ell" and len(v) == ell.size and np.isfinite(np.asarray(v, dtype=float)).any()}
        if not {"P_true", "P_pred", "coherence"} <= set(cols) or ell.size == 0:
            continue
        has_resid = has_resid or ("P_resid_pred" in cols and "P_resid_true" in cols)
        if scope == "all":
            nan = np.full(ell.size, np.nan)
            for i, l in enumerate(ell):
                if l >= 1:
                    per_degree.append([state, int(l), _wavelength_km(l), cols.get("amplitude_ratio", nan)[i],
                                       cols["coherence"][i], cols.get("residual_amplitude_ratio", nan)[i],
                                       cols.get("interp_amplitude_ratio", nan)[i]])
        edges = ladder_edges(int(np.nanmax(ell)), lmin, nbins)
        for lo, hi, rec in _bin_powers(ell, cols, edges):
            km = _wavelength_km(math.sqrt(lo * max(hi - 1, lo)))
            b = f"l{lo}-{hi - 1}"
            bins[b] = km
            rec["bin_lo_km"], rec["bin_hi_km"] = _wavelength_km(max(hi - 1, 1)), _wavelength_km(lo)
            if scope == "all":
                pooled[(state, b)] = rec
            else:
                dated[(state, b)][scope] = rec
    return {"pooled": pooled, "dated": dict(dated), "bins": bins, "per_degree": per_degree,
            "source": source, "has_resid": has_resid}



# --------------------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------------------

def compare_bin(row: dict, ref: dict, var: str, b: str, q: str, *, tol: float = 0.03,
                t_crit: float = 2.03) -> dict:
    """Row minus reference in one bin: pooled difference, per-date mean, SE, t, off flag."""
    rp, fp = row["pooled"].get((var, b), {}), ref["pooled"].get((var, b), {})
    pooled = fnum(rp.get(q)) - fnum(fp.get(q))
    rd, fd = row["dated"].get((var, b), {}), ref["dated"].get((var, b), {})
    dates = sorted(set(rd) & set(fd))
    diffs = [fnum(rd[d].get(q)) - fnum(fd[d].get(q)) for d in dates]
    m, se, n = mean_se(diffs)
    if n >= 2 and math.isfinite(se):
        t = m / se if se > 0 else (0.0 if m == 0 else math.copysign(math.inf, m))
        off = abs(m) > tol and abs(t) >= t_crit
        rule = "per-date"
    else:
        t, m, se = math.nan, (m if n else math.nan), math.nan
        off = math.isfinite(pooled) and abs(pooled) > tol
        rule = "0.03 rule alone (no per-date values)"
    return {"pooled": pooled, "mean": m, "se": se, "t": t, "n": n, "off": bool(off), "rule": rule}


def _row_label_cells(ctx: Context):
    return [r.label for r in ctx.rows]


def _views(ctx: Context, evaluator: str, required: str) -> dict[str, dict[str, object]]:
    """{view key: {label: Found}} over all rows."""
    out: dict[str, dict] = defaultdict(dict)
    for r in ctx.rows:
        for f in ctx.results(r.label, evaluator, required):
            out[f.key][r.label] = f
    return dict(out)


def _box_text(box) -> str:
    if isinstance(box, (list, tuple)) and len(box) == 4:
        la0, la1, lo0, lo1 = (float(v) for v in box)
        return f"{la0:g} to {la1:g} N, {lo0:g} to {lo1:g} E"
    return str(box) if box else "box unknown"


def _block(ctx: Context, kind: str, view: str, data: dict[str, dict], quants, *,
           has_resid: bool, res: SectionResult) -> str:
    """Tables of one view (all rows) and the summary fragments of each row."""
    ref = ctx.reference
    tol = float(ctx.opt("spectra_tol", 0.03))
    truth_band = float(ctx.opt("truth_band", 0.03))
    t_crit = float(ctx.opt("t_crit", 2.03))
    min_km = float(ctx.opt("min_km", 0.0))
    safe = view.replace("/", "_")
    out = []
    labels = [lab for lab in _row_label_cells(ctx) if lab in data]
    missing = [lab for lab in _row_label_cells(ctx) if lab not in data]
    if missing:
        out.append(na(f"no {kind} output for: {', '.join(missing)}") + "\n")
    bins_all: dict[str, float] = {}
    for lab in labels:
        bins_all.update(data[lab]["bins"])
    bins = sorted(bins_all, key=lambda b: -bins_all[b])
    variables = sorted({v for lab in labels for (v, _) in data[lab]["pooled"]})
    ref_data = data.get(ref)

    # wide tables rows x bins, one per variable and quantity
    for var in variables:
        out.append(f"### {var}\n")
        vbins = [b for b in bins if any((var, b) in data[lab]["pooled"] for lab in labels)]
        for q, qname in quants:
            header = ["row"] + _bin_labels([bins_all[b] for b in vbins])
            rows = []
            for lab in labels:
                rows.append([lab] + [fmt(data[lab]["pooled"].get((var, b), {}).get(q), 3) for b in vbins])
            out.append(f"**{qname}** (amplitude, pooled over dates and members)\n\n")
            out.append(table(ctx, f"spectra_{kind}_{safe}_{var}_{q}", header, rows))
        if not has_resid:
            out.append(f"Residual: {na(f'{kind} gives no residual spectrum')}\n")
        # difference to the reference per bin
        if ref_data is None:
            out.append(na(f"no {kind} output for the reference {ref}; no difference table") + "\n")
            continue
        for q in [d for d in DIFF_QUANTS if any(q2 == d for q2, _ in quants)]:
            header = ["row"] + _bin_labels([bins_all[b] for b in vbins])
            rows, long_rows = [], []
            for lab in labels:
                cells = [lab]
                for b in vbins:
                    c = compare_bin(data[lab], ref_data, var, b, q, tol=tol, t_crit=t_crit)
                    val = c["mean"] if math.isfinite(fnum(c["mean"])) and c["n"] >= 2 else c["pooled"]
                    cells.append(fmt(val, 3, signed=True) + (" *" if c["off"] else ""))
                    long_rows.append([view, lab, var, q, bins_all[b], c["pooled"], c["mean"], c["se"],
                                      c["t"], c["n"], int(c["off"]), c["rule"]])
                    if c["off"] and lab != ref and bins_all[b] >= min_km:
                        ctx.cache.setdefault(("spec_off", kind), defaultdict(list))[lab].append(
                            (view, var, q, bins_all[b], val, c["t"]))
                rows.append(cells)
            out.append(f"**{q}: row minus `{ref}`** (mean over dates of the per-date difference; "
                       f"`*` = off the reference: |diff| > {tol} and |t| >= {t_crit}, or the {tol} rule "
                       "alone where no per-date values exist)\n\n")
            out.append(table(ctx, f"spectra_{kind}_{safe}_{var}_{q}_minus_ref", header, rows))
            from .common import write_csv
            write_csv(ctx.csv_dir / f"spectra_{kind}_{safe}_{var}_{q}_vs_ref_long.csv",
                      ["view", "row", "variable", "quantity", "bin_centre_km", "diff_pooled",
                       "diff_perdate_mean", "se", "t", "n_dates", "off_ref", "rule"], long_rows)

    # bins outside the truth band, every row (reference included)
    lst = []
    for lab in labels:
        for (var, b), rec in sorted(data[lab]["pooled"].items(), key=lambda kv: (kv[0][0], -bins_all[kv[0][1]])):
            if bins_all[b] < min_km:
                continue
            for q in ("ratio_full", "ratio_resid"):
                v = fnum(rec.get(q))
                if math.isfinite(v) and abs(v - 1.0) > truth_band:
                    lst.append([lab, var, q, f"{bins_all[b]:.0f}", fmt(v, 3)])
                    ctx.cache.setdefault(("spec_truth", kind), defaultdict(int))[lab] += 1
    out.append(f"### Bins outside {1 - truth_band:.2f}-{1 + truth_band:.2f} of the truth "
               f"(full field and residual{', bins >= %g km' % min_km if min_km else ''})\n\n")
    out.append(table(ctx, f"spectra_{kind}_{safe}_outside_truth", ["row", "variable", "quantity", "km", "ratio"], lst))

    # bins off the reference, list per row
    offs = ctx.cache.get(("spec_off", kind), {})
    lst2 = []
    for lab in labels:
        for (vw, var, q, km, val, t) in offs.get(lab, []):
            if vw == view:
                lst2.append([lab, var, q, f"{km:.0f}", fmt(val, 3, signed=True), fmt(t, 2)])
    out.append(f"### Bins off the reference (|diff| > {tol})\n\n")
    out.append(table(ctx, f"spectra_{kind}_{safe}_off_reference", ["row", "variable", "quantity", "km", "diff", "t"], lst2))

    # summary fragments for this view
    for lab in labels:
        if lab == ref:
            n_out = sum(1 for r in lst if r[0] == lab)
            res.add(lab, f"{kind} {view}: reference; {n_out} bins outside truth band")
            continue
        mine = [x for x in offs.get(lab, []) if x[0] == view]
        gate = [x for x in mine if x[1] in GATE_VARS and x[2] in ("ratio_full", "ratio_resid")]
        n_out = sum(1 for r in lst if r[0] == lab)
        worst = sorted(gate, key=lambda x: -abs(fnum(x[4])))[:3]
        worst_txt = ", ".join(f"{v} {q.replace('ratio_', '')} {km:.0f} km {fmt(d, 3, signed=True)}"
                              for (_, v, q, km, d, _) in worst)
        status = "pass" if not gate else "fail"
        res.add(lab, f"{kind} {view}: {status} ({len(gate)} gating bins off ref"
                     f"{': ' + worst_txt if worst_txt else ''}; {len(mine)} off in all vars; "
                     f"{n_out} bins outside truth band)")
    return "\n".join(out)


def section(ctx: Context) -> SectionResult:
    res = SectionResult("2. Spectra", "")
    md = ["# 2. Spectra\n",
          "AMPLITUDE ratios (not power), per bin, never averaged over a band. Wide tables: rows x "
          "bins (CSV) / bins as columns (markdown). Pass rule: no bin of 10u, 10v or 2t off the "
          "reference, full field or residual; msl and tp are reported and do not gate.\n"]
    min_km = float(ctx.opt("min_km", 0.0))
    if min_km:
        md.append(f"Bins below {min_km:g} km are tabulated but not listed or gated.\n")

    def box_part() -> str:
        views = _views(ctx, "box_spectra", "bins.csv")
        if not views:
            return "## 2a. Box spectra (box_spectra)\n\n" + na(
                f"no box_spectra/bins.csv under {ctx.eval_root}/<set>/<row>/") + "\n"
        parts = []
        for view, founds in sorted(views.items()):
            data, bad = {}, []
            for lab, f in founds.items():
                try:
                    data[lab] = read_box(f)
                except Exception as exc:  # noqa: BLE001
                    bad.append(f"{lab} ({type(exc).__name__}: {exc})")
                    res.add(lab, f"box {view}: " + na(f"unreadable bins.csv: {exc}"))
            boxes = {_box_text(d["box"]) for d in data.values()}
            parts.append(f"## 2a. Box spectra, view `{view}` ({'; '.join(sorted(boxes))})\n")
            if bad:
                parts.append(na("unreadable input for " + ", ".join(bad)) + "\n")
            if len(boxes) > 1:
                parts.append(na("rows of this view were read on different boxes; not comparable") + "\n")
            parts.append(_block(ctx, "box", view, data, BOX_QUANTS, has_resid=True, res=res))
        return "\n".join(parts)

    def global_part() -> str:
        views = _views(ctx, "spectra_coherence", ("curves.csv", "coherence.json"))
        if not views:
            return "## 2b. Global spectra (spectra_coherence)\n\n" + na(
                f"no spectra_coherence/coherence.json under {ctx.eval_root}/<set>/<row>/") + "\n"
        parts = []
        lmin = int(ctx.opt("global_lmin", 20))
        nb = int(ctx.opt("global_nbins", 30))
        for view, founds in sorted(views.items()):
            data, per_degree = {}, []
            for lab, f in founds.items():
                try:
                    d = read_global(f, lmin=lmin, nbins=nb)
                except Exception as exc:  # noqa: BLE001
                    parts.append(na(f"unreadable coherence.json for {lab}: {type(exc).__name__}: {exc}") + "\n")
                    res.add(lab, f"global {view}: " + na(f"unreadable coherence.json: {exc}"))
                    continue
                data[lab] = d
                per_degree += [[view, lab] + r for r in d["per_degree"]]
            from .common import write_csv
            write_csv(ctx.csv_dir / f"spectra_global_{view.replace('/', '_')}_per_degree.csv",
                      ["view", "row", "state", "ell", "wavelength_km", "amplitude_ratio", "coherence",
                       "residual_amplitude_ratio", "interp_amplitude_ratio"], per_degree)
            has_resid = any(d["has_resid"] for d in data.values())
            per_date = any(d["dated"] for d in data.values())
            sources = sorted({d["source"] for d in data.values()})
            parts.append(f"## 2b. Global spherical-harmonic spectra, view `{view}`\n\n"
                         f"Read from {', '.join(sources)}. Degree bins: {nb} log-spaced bins from l = "
                         f"{lmin} (about {EARTH_CIRC_KM / lmin:.0f} km) to the tool's lmax, named by their "
                         "degree range; powers summed in the bin before the ratio (no bin is averaged "
                         "with another). "
                         + ("Per-date curves exist: the per-date rule applies. " if per_date else
                            "No per-date values: differences to the reference use the 0.03 rule alone. ")
                         + f"Per-degree values: `csv/spectra_global_{view.replace('/', '_')}_per_degree.csv`.\n")
            quants = list(GLOBAL_QUANTS)
            if has_resid:
                quants.insert(1, ("ratio_resid", "residual / truth residual"))
            parts.append(_block(ctx, "global", view, data, quants, has_resid=has_resid, res=res))
        return "\n".join(parts)

    guarded(md, "## 2a. Box spectra", box_part)
    guarded(md, "## 2b. Global spectra", global_part)
    res.md = "\n".join(md)
    for r in ctx.rows:
        if r.label not in res.summary:
            res.add(r.label, na("no spectra output"))
    return res
