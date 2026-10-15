"""Part 1: physical structure (shape, tc_structure, texture, wind_extremes).

Inputs per row (``<EVAL_ROOT>/<set>/<row>/evaluators/<name>/``):

- shape: ``rows.csv`` (columns arm, dirname, date, step, var, member, field in draw/truth/driver,
  band in mid/b12, elong_frac_gt3, n_components, ell_ratio_median, ...), else ``summary.json``
  (``rows``: step, var, window, field, band, statistic, mean, se_date, n_dates). The number of
  elongated features per field is elong_frac_gt3 x n_components of each (date, member) cell.
  Means carry the date-clustered bootstrap SE (eval.shared.date_bootstrap, members of a date
  together); the row-minus-reference difference is also paired cell by cell.
  Rule: differs when |diff| > 2 x combined SE (sqrt(se_row^2 + se_ref^2)); towards the
  truth's value is a gain, away a loss.
- tc_structure: ``cases.csv`` (event, date, lead, field in model/truth/input, member, found,
  pmin_hpa, maxwind300_ms, zeta50_s, zeta100_s, zeta200_s, rmw_km). Member-paired difference to
  the reference on (event, date, lead, member); SE over members (sd of the member means /
  sqrt(n members)). Rule: loss when |diff| > 2 SE, and > 2 hPa on Pmin or > 3 % on peak wind,
  and away from the truth (gain the other way). zeta50 is reported, not judged.
- texture: ``texture.json`` (``aggregate``: state, stratum, truth/model -> stat -> mean, sd, n).
- wind_extremes: ``wind_extremes.json`` (``aggregate``: box, model_minus_truth.retention[R] ->
  mean, sd, n).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .common import (Context, SectionResult, fmt, fnum, guarded, mean_se, na, read_csv,
                     read_json, row_guard, se_of, table)

SHAPE_STATS = ("elong_frac_gt3", "n_elong", "ell_ratio_median", "n_components")
SHAPE_JUDGED = ("elong_frac_gt3", "n_elong")
TCS_STATS = [("pmin_hpa", "Pmin hPa", 1.0, 2.0, "abs"),
             ("maxwind300_ms", "max wind 300 km m/s", 1.0, 3.0, "pct"),
             ("zeta100_s", "vorticity 100 km 1e-5/s", 1e5, None, None),
             ("zeta200_s", "vorticity 200 km 1e-5/s", 1e5, None, None),
             ("rmw_km", "RMW km", 1.0, None, None),
             ("zeta50_s", "vorticity 50 km 1e-5/s (reported, not judged)", 1e5, None, "skip")]
TEXTURE_STATS = ("fine_nn_corr", "fine_var")


def _boot_mean(dates, values):
    try:
        from eval.shared.date_bootstrap import boot_mean
        b = boot_mean(list(dates), np.asarray(values, dtype=float))
        return fnum(b["value"]), fnum(b["se"]), int(b.get("n_dates") or 0)
    except ImportError:  # pragma: no cover - cluster SE by hand
        by = defaultdict(list)
        for d, v in zip(dates, values):
            if math.isfinite(v):
                by[d].append(v)
        means = [sum(v) / len(v) for v in by.values()]
        m, se, n = mean_se(means)
        return m, se, n


# --------------------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------------------

def shape_cells(found, band: str) -> dict:
    """{(step, var, field): {(date, member): {stat: value}}} from rows.csv."""
    out: dict = defaultdict(dict)
    for r in read_csv(found.path / "rows.csv"):
        if r.get("band") != band:
            continue
        rec = {s: fnum(r.get(s)) for s in ("elong_frac_gt3", "ell_ratio_median", "n_components")}
        rec["n_elong"] = rec["elong_frac_gt3"] * rec["n_components"]
        out[(str(r["step"]), r["var"], r["field"])][(r["date"], str(r["member"]))] = rec
    return dict(out)


def shape_summary_only(found, band: str) -> dict:
    """{(step, var, field, stat): (mean, se)} from summary.json (no cells)."""
    data = read_json(found.path / "summary.json")
    out, prio = {}, {}
    rank = {"both": 0, "all": 1}   # pooled windows first; else the first window seen
    for r in data.get("rows", []):
        if r.get("band") != band:
            continue
        k = (str(r["step"]), r["var"], r["field"], r["statistic"])
        p = rank.get(r.get("window"), 2)
        if k not in out or p < prio[k]:
            out[k] = (fnum(r["mean"]), fnum(r.get("se_date")))
            prio[k] = p
    return out


def shape_section(ctx: Context, res: SectionResult) -> str:
    band = ctx.opt("shape_band", "mid")
    ref = ctx.reference
    founds = {r.label: ctx.results_by_key(r.label, "shape", ("rows.csv", "summary.json")) for r in ctx.rows}
    keys = sorted({k for d in founds.values() for k in d})
    head = (f"## 1a. Shape (band `{band}`; 40-109 km for `mid` at kernel_scale 1)\n\n"
            "Elongated fraction (> 3:1) and number of elongated features per field "
            "(elong_frac_gt3 x n_components per date and member), axis ratio; row (draw), "
            f"reference `{ref}` (draw) and truth (truth field of the reference's file), date-clustered "
            "SE. 'differs' = |diff| > 2 x combined SE; 'gain' = towards the truth.\n\n")
    for k in keys:   # the band's wavelength span and kernel scale, as the run recorded them
        f0 = founds[ref].get(k) or next((d[k] for d in founds.values() if k in d), None)
        try:
            meta = read_json(f0.path / "run_meta.json")
            head += (f"View `{k}`: box {meta.get('box')} {meta.get('box_bounds')}, kernel_scale "
                     f"{meta.get('kernel_scale', 1)}, band {band} = {meta.get('band_km', {}).get(band, '?')} km, "
                     f"regrid {meta.get('regrid_deg')} deg.\n\n")
        except Exception:  # noqa: BLE001
            pass
    if not keys:
        for r in ctx.rows:
            res.add(r.label, "shape " + na("no shape output"))
        return head + na(f"no shape output under {ctx.where('<row>', 'shape')}") + "\n"
    header = ["view", "row", "var", "step", "statistic", "row value", "row SE", "ref value", "ref SE",
              "truth", "truth SE", "diff row-ref", "combined SE", "paired SE", "differs (2 SE)", "direction"]
    rows = []
    for key in keys:
        rf = founds[ref].get(key)
        ref_cells = shape_cells(rf, band) if rf and (rf.path / "rows.csv").exists() else None
        ref_sum = shape_summary_only(rf, band) if rf and ref_cells is None and (rf.path / "summary.json").exists() else None
        for row in ctx.rows:
            f = founds[row.label].get(key)
            if f is None:
                rows.append([key, row.label] + [na("no shape output")] * 14)
                res.add(row.label, f"shape {key}: " + na("no output"))
                continue
            if rf is None:
                rows.append([key, row.label] + [na(f"no shape output for the reference {ref}")] * 14)
                continue
            with row_guard(rows, [key, row.label], len(header) - 2, res, row.label, f"shape {key}"):
                cells = shape_cells(f, band) if (f.path / "rows.csv").exists() else None
                frags = []
                if cells is not None and ref_cells is not None:
                    for (step, var, field) in sorted(k for k in cells if k[2] == "draw"):
                        rc = cells[(step, var, "draw")]
                        fc = ref_cells.get((step, var, "draw"), {})
                        tc = ref_cells.get((step, var, "truth"), cells.get((step, var, "truth"), {}))
                        for st in SHAPE_STATS:
                            def stat_of(c):
                                ks = sorted(c)
                                return _boot_mean([k[0] for k in ks], [c[k][st] for k in ks])
                            rv, rse, _ = stat_of(rc)
                            fv, fse, _ = stat_of(fc) if fc else (math.nan, math.nan, 0)
                            tv, tse, _ = stat_of(tc) if tc else (math.nan, math.nan, 0)
                            common = sorted(set(rc) & set(fc))
                            pdiff, pse, _ = _boot_mean([k[0] for k in common],
                                                       [rc[k][st] - fc[k][st] for k in common]) if common else (math.nan, math.nan, 0)
                            diff = rv - fv
                            comb = math.sqrt(rse ** 2 + fse ** 2) if math.isfinite(rse) and math.isfinite(fse) else math.nan
                            differs = math.isfinite(comb) and abs(diff) > 2 * comb
                            direction = ""
                            if math.isfinite(tv) and math.isfinite(diff):
                                direction = "towards truth" if abs(rv - tv) < abs(fv - tv) else "away from truth"
                            verdict = ("" if row.label == ref else
                                       ("gain" if differs and direction == "towards truth" else
                                        "loss" if differs else "no difference") if st in SHAPE_JUDGED else "reported")
                            rows.append([key, row.label, var, step, st, fmt(rv, 4), fmt(rse, 4), fmt(fv, 4),
                                         fmt(fse, 4), fmt(tv, 4), fmt(tse, 4), fmt(diff, 4, signed=True),
                                         fmt(comb, 4), fmt(pse, 4), "yes" if differs else "no",
                                         f"{direction}; {verdict}" if verdict else direction])
                            if st in SHAPE_JUDGED and row.label != ref:
                                frags.append(f"{var} {st} {fmt(rv, 3)} vs {fmt(fv, 3)} (truth {fmt(tv, 3)}, "
                                             f"{fmt(diff, 3, signed=True)}±{fmt(comb, 3)}) {verdict}")
                else:
                    s_row = shape_summary_only(f, band)
                    s_ref = ref_sum if ref_sum is not None else shape_summary_only(rf, band)
                    for (step, var, field, st) in sorted(k for k in s_row if k[2] == "draw"):
                        if st not in SHAPE_STATS:
                            continue
                        rv, rse = s_row[(step, var, "draw", st)]
                        fv, fse = s_ref.get((step, var, "draw", st), (math.nan, math.nan))
                        tv, tse = s_ref.get((step, var, "truth", st), (math.nan, math.nan))
                        diff = rv - fv
                        comb = math.sqrt(rse ** 2 + fse ** 2) if math.isfinite(rse) and math.isfinite(fse) else math.nan
                        differs = math.isfinite(comb) and abs(diff) > 2 * comb
                        direction = ("towards truth" if abs(rv - tv) < abs(fv - tv) else "away from truth") if math.isfinite(tv) else ""
                        rows.append([key, row.label, var, step, st, fmt(rv, 4), fmt(rse, 4), fmt(fv, 4), fmt(fse, 4),
                                     fmt(tv, 4), fmt(tse, 4), fmt(diff, 4, signed=True), fmt(comb, 4),
                                     na("no rows.csv"), "yes" if differs else "no", direction])
                    rows.append([key, row.label, "", "", "n_elong"] + [na("rows.csv missing (summary.json only)")] * 11)
                    if row.label != ref:
                        frags.append("n_elong " + na("rows.csv missing"))
                if row.label != ref and frags:
                    res.add(row.label, f"shape {key}: " + "; ".join(frags))
                elif row.label == ref:
                    res.add(row.label, f"shape {key}: reference")
    return head + table(ctx, "structure_shape", header, rows)


# --------------------------------------------------------------------------------------
# tc_structure
# --------------------------------------------------------------------------------------

def tcs_cases(found) -> list[dict]:
    return read_csv(found.path / "cases.csv")


def _found_rows(cases, field):
    return [r for r in cases if r["field"] == field and str(r.get("found")) in ("1", "1.0", "True")]


def tc_structure_section(ctx: Context, res: SectionResult) -> str:
    ref = ctx.reference
    head = ("## 1b. Storm structure (tc_structure)\n\n"
            "Per event and lead (and all leads): row (model), reference (model), truth (truth field "
            "of the reference's files); member-paired difference row - reference on (event, date, "
            "lead, member) with SE over members. 'loss' = |diff| > 2 SE and above the threshold "
            "(2 hPa Pmin, 3 % peak wind) and away from the truth; 'gain' = the same towards it. "
            "Vorticity in 1e-5 s^-1.\n\n")
    founds = {r.label: ctx.results_by_key(r.label, "tc_structure", "cases.csv") for r in ctx.rows}
    keys = sorted({k for d in founds.values() for k in d})
    if not keys:
        for r in ctx.rows:
            res.add(r.label, "tc_structure " + na("no output"))
        return head + na(f"no tc_structure/cases.csv under {ctx.eval_root}/<set>/<row>/") + "\n"
    header = ["view", "row", "event", "lead", "quantity", "row value", "ref value", "truth value", "diff row-ref",
              "SE (members)", "n members", "n pairs", "status"]
    rows = []
    for key in keys:
        rf = founds[ref].get(key)
        if rf is None:
            rows.append([key, "*", "", "", ""] + [na(f"no tc_structure output for the reference {ref}")] * 8)
            continue
        ref_cases = tcs_cases(rf)
        ref_model = _found_rows(ref_cases, "model")
        ref_truth = _found_rows(ref_cases, "truth")
        events = sorted({r["event"] for r in ref_model})
        for row in ctx.rows:
            f = founds[row.label].get(key)
            if f is None:
                rows.append([key, row.label, "", "", ""] + [na("no tc_structure output")] * 8)
                res.add(row.label, f"tc_structure {key}: " + na("no output"))
                continue
            with row_guard(rows, [key, row.label], len(header) - 2, res, row.label, f"tc_structure {key}"):
                model = _found_rows(tcs_cases(f), "model")
                counts = defaultdict(int)
                losses, gains = [], []
                for ev in events:
                    leads = sorted({int(fnum(r["lead"])) for r in ref_model if r["event"] == ev})
                    for lead in leads + ["all"]:
                        sel = lambda rs: [r for r in rs if r["event"] == ev and (lead == "all" or int(fnum(r["lead"])) == lead)]  # noqa: E731
                        a, b, t = sel(model), sel(ref_model), sel(ref_truth)
                        ia = {(r["date"], str(r["lead"]), str(r["member"])): r for r in a}
                        ib = {(r["date"], str(r["lead"]), str(r["member"])): r for r in b}
                        pairs = sorted(set(ia) & set(ib))
                        for col, name, scale, thr, kind in TCS_STATS:
                            rv = mean_se([fnum(r[col]) for r in a])[0]
                            fv = mean_se([fnum(r[col]) for r in b])[0]
                            tv = mean_se([fnum(r[col]) for r in t])[0]
                            diffs = defaultdict(list)
                            for k in pairs:
                                d = fnum(ia[k][col]) - fnum(ib[k][col])
                                if math.isfinite(d):
                                    diffs[k[2]].append(d)
                            all_d = [x for v in diffs.values() for x in v]
                            dmean = sum(all_d) / len(all_d) if all_d else math.nan
                            _, se_m, n_m = mean_se([sum(v) / len(v) for v in diffs.values() if v])
                            status = ""
                            if row.label != ref and kind != "skip" and math.isfinite(dmean):
                                big = math.isfinite(se_m) and abs(dmean) > 2 * se_m
                                if kind == "abs":
                                    big = big and abs(dmean) > thr
                                elif kind == "pct":
                                    big = big and math.isfinite(fv) and fv != 0 and abs(dmean) / abs(fv) * 100 > thr
                                towards = math.isfinite(tv) and abs(rv - tv) < abs(fv - tv)
                                status = ("gain" if towards else "loss") if big else "no difference"
                                counts[status] += 1
                                if status == "loss":
                                    losses.append(f"{ev} {lead} {col} {fmt(dmean * scale, 2, signed=True)}")
                                elif status == "gain":
                                    gains.append(f"{ev} {lead} {col} {fmt(dmean * scale, 2, signed=True)}")
                            elif kind == "skip":
                                status = "reported"
                            rows.append([key, row.label, ev, lead, name, fmt(rv * scale, 3), fmt(fv * scale, 3),
                                         fmt(tv * scale, 3), fmt(dmean * scale, 3, signed=True),
                                         fmt(se_m * scale, 3), n_m, len(pairs), status])
                if row.label == ref:
                    res.add(row.label, f"tc_structure {key}: reference")
                else:
                    res.add(row.label, f"tc_structure {key}: {counts['loss']} loss, {counts['gain']} gain, "
                                       f"{counts['no difference']} no difference"
                                       + (f"; losses: {', '.join(losses[:4])}" if losses else "")
                                       + (f"; gains: {', '.join(gains[:3])}" if gains else ""))
    return head + table(ctx, "structure_tc_structure", header, rows)


# --------------------------------------------------------------------------------------
# texture and wind_extremes
# --------------------------------------------------------------------------------------

def texture_section(ctx: Context, res: SectionResult) -> str:
    ref = ctx.reference
    head = ("## 1c. Texture (texture)\n\nNearest-neighbour correlation of the fine part and fine "
            "variance per stratum (region strata first); row, reference, truth (truth of the "
            "reference's files), mean over samples with SE = sd / sqrt(n).\n\n")
    founds = {r.label: ctx.results_by_key(r.label, "texture", "texture.json") for r in ctx.rows}
    keys = sorted({k for d in founds.values() for k in d})
    if not keys:
        for r in ctx.rows:
            res.add(r.label, "texture " + na("no output"))
        return head + na(f"no texture/texture.json under {ctx.eval_root}/<set>/<row>/") + "\n"
    header = ["view", "row", "state", "stratum", "kind", "statistic", "row value", "row SE", "ref value",
              "ref SE", "truth value", "row-ref", "row/truth"]
    rows = []
    for key in keys:
        rf = founds[ref].get(key)
        if rf is None:
            rows.append([key, "*"] + [na(f"no texture output for the reference {ref}")] * 11)
            continue
        rdata = read_json(rf.path / "texture.json")
        kinds = {k: v.get("kind", "") for k, v in (rdata.get("strata") or {}).items()}
        ragg = {(a["state"], a["stratum"]): a for a in rdata.get("aggregate", [])}
        for row in ctx.rows:
            f = founds[row.label].get(key)
            if f is None:
                rows.append([key, row.label] + [na("no texture output")] * 11)
                res.add(row.label, f"texture {key}: " + na("no output"))
                continue
            with row_guard(rows, [key, row.label], len(header) - 2, res, row.label, f"texture {key}"):
                agg = {(a["state"], a["stratum"]): a for a in read_json(f.path / "texture.json").get("aggregate", [])}
                frags = []
                order = sorted(agg, key=lambda k: (k[0], kinds.get(k[1], "") != "region", k[1]))
                for (state, stratum) in order:
                    a, b = agg[(state, stratum)], ragg.get((state, stratum))
                    for st in TEXTURE_STATS:
                        m = a["model"].get(st, {})
                        rv, rse = fnum(m.get("mean")), se_of(m.get("sd"), m.get("n"))
                        fm = b["model"].get(st, {}) if b else {}
                        fv, fse = fnum(fm.get("mean")), se_of(fm.get("sd"), fm.get("n"))
                        tv = fnum((b or a)["truth"].get(st, {}).get("mean"))
                        rows.append([key, row.label, state, stratum, kinds.get(stratum, ""), st, fmt(rv, 4),
                                     fmt(rse, 4), fmt(fv, 4), fmt(fse, 4), fmt(tv, 4),
                                     fmt(rv - fv, 4, signed=True), fmt(rv / tv if tv else math.nan, 3)])
                        if (st == "fine_nn_corr" and kinds.get(stratum) == "region" and row.label != ref):
                            frags.append(f"{state} {stratum} nn {fmt(rv, 3)} vs {fmt(fv, 3)} (truth {fmt(tv, 3)})")
                if row.label != ref:
                    res.add(row.label, f"texture {key}: " + ("; ".join(frags[:6]) if frags else na("no region strata")))
    return head + table(ctx, "structure_texture", header, rows)


def wind_extremes_section(ctx: Context, res: SectionResult) -> str:
    ref = ctx.reference
    head = ("## 1d. Wind maximum coherence (wind_extremes)\n\nRetention ratio (maximum of the R-km "
            "disk average over the raw maximum), model minus truth, per box and radius; mean over "
            "samples, SE = sd / sqrt(n).\n\n")
    founds = {r.label: ctx.results_by_key(r.label, "wind_extremes", "wind_extremes.json") for r in ctx.rows}
    keys = sorted({k for d in founds.values() for k in d})
    if not keys:
        for r in ctx.rows:
            res.add(r.label, "wind_extremes " + na("no output"))
        return head + na(f"no wind_extremes/wind_extremes.json under {ctx.eval_root}/<set>/<row>/") + "\n"
    header = ["view", "row", "box", "radius km", "row (model-truth)", "row SE", "ref (model-truth)",
              "ref SE", "row - ref", "n"]
    rows = []
    for key in keys:
        rf = founds[ref].get(key)
        ragg = {a["box"]: a for a in read_json(rf.path / "wind_extremes.json").get("aggregate", [])} if rf else {}
        for row in ctx.rows:
            f = founds[row.label].get(key)
            if f is None:
                rows.append([key, row.label] + [na("no wind_extremes output")] * 8)
                res.add(row.label, f"wind_extremes {key}: " + na("no output"))
                continue
            with row_guard(rows, [key, row.label], len(header) - 2, res, row.label, f"wind_extremes {key}"):
                frags = []
                for a in read_json(f.path / "wind_extremes.json").get("aggregate", []):
                    b = ragg.get(a["box"])
                    for rk, v in sorted(a.get("model_minus_truth", {}).get("retention", {}).items(),
                                        key=lambda kv: fnum(kv[0])):
                        rv, rse = fnum(v.get("mean")), se_of(v.get("sd"), v.get("n"))
                        w = (b or {}).get("model_minus_truth", {}).get("retention", {}).get(rk, {})
                        fv, fse = fnum(w.get("mean")), se_of(w.get("sd"), w.get("n"))
                        rows.append([key, row.label, a["box"], rk, fmt(rv, 4, signed=True), fmt(rse, 4),
                                     fmt(fv, 4, signed=True) if b else na(f"no reference output for box {a['box']}"),
                                     fmt(fse, 4), fmt(rv - fv, 4, signed=True), v.get("n", "")])
                        if row.label != ref and abs(fnum(rk) - 30.0) < 1e-6:
                            frags.append(f"{a['box']} R30 {fmt(rv, 3, signed=True)} vs ref {fmt(fv, 3, signed=True)}")
                if row.label != ref:
                    res.add(row.label, f"wind_extremes {key}: " + ("; ".join(frags) if frags else "see table"))
    return head + table(ctx, "structure_wind_extremes", header, rows)


def section(ctx: Context) -> SectionResult:
    res = SectionResult("1. Structure", "")
    md = ["# 1. Physical structure\n",
          "Maps (zoom_maps) are read by eye and are not aggregated here.\n"]
    guarded(md, "## 1a. Shape", lambda: shape_section(ctx, res))
    guarded(md, "## 1b. Storm structure", lambda: tc_structure_section(ctx, res))
    guarded(md, "## 1c. Texture", lambda: texture_section(ctx, res))
    guarded(md, "## 1d. Wind extremes", lambda: wind_extremes_section(ctx, res))
    res.md = "\n".join(md)
    return res
