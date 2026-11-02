"""Summaries of the per-date station scores: means over dates with their standard errors, paired
differences between rows of one run, and between the model rows of two runs (``eval.cli obs compare``).

Every statistic is over dates: one value per date (the station mean of that date), then the mean, the
standard error of the mean (ddof 1) and the number of dates on which the first member of a pair is better
("wins"). Lower is better for every score except the correlation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KEYS = ["var", "variant", "radius_km", "region", "lead"]
HEADLINE = ("fcrps", "spread", "spread_rms", "ssr_corrected", "seeps")
SECONDARY = ("bias", "rmse_mean", "mae_mean", "mae_member", "corr_mean")
TAIL_PREFIXES = ("twcrps", "brier", "n_obs_gt")
# Rows of one run compared date by date: (a, b) gives a - b.
DEFAULT_PAIRS = (("model", "input"), ("model", "truth"), ("truth", "input"),
                 ("model", "hres"), ("hres", "truth"), ("hres", "input"))
VAR_UNITS = {"2t": "K", "10ff": "m/s", "msl": "Pa", "tp": "mm / 12 h", "tp24": "mm / 24 h"}


def tail_name(col: str) -> str:
    """'twcrps5' -> 'fair twCRPS > 5 mm', 'brier10' -> 'fair Brier > 10 mm'."""
    for prefix, name in (("twcrps", "fair twCRPS"), ("brier", "fair Brier"), ("n_obs_gt", "observed count")):
        if col.startswith(prefix):
            return f"{name} > {col[len(prefix):]} mm"
    return col


def tail_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith(TAIL_PREFIXES)]


def oriented_metrics(df) -> list[str]:
    """Metrics with a direction (lower or higher is better), headline first."""
    return ["fcrps"] + [c for c in df.columns if c.startswith(("twcrps", "brier"))] + \
           (["seeps"] if "seeps" in df.columns else []) + \
           [c for c in ("mae_mean", "rmse_mean", "mae_member", "corr_mean") if c in df.columns]


def summarise(df):
    """Mean over dates per (var, variant, radius, region, lead, row): every score, its SE, mean n, dates."""
    cols = [c for c in HEADLINE + SECONDARY if c in df.columns] + tail_columns(df)
    g = df.groupby(KEYS + ["row", "kind"])
    s = g[cols].mean().join(g[cols].sem().add_suffix("_se")).join(g["n"].mean().rename("mean_n"))
    s = s.join(g["members"].first()).join(g.size().rename("dates"))
    if {"spread_rms", "rmse_mean"} <= set(df.columns):
        # spread/skill from the variance and MSE pooled over dates, next to the mean of the per-date ratios
        def pooled_ssr(q):
            m, mse = q.members.iloc[0], (q.rmse_mean ** 2).mean()
            return np.sqrt((q.spread_rms ** 2).mean() / mse * (m + 1) / m) if m > 1 and mse > 0 else np.nan

        pool = g[["spread_rms", "rmse_mean", "members"]].apply(pooled_ssr)
        s = s.join(pool.rename("ssr_pooled"))
    return s.reset_index()


def _paired_table(wide_a, wide_b, metric_list, keys):
    import pandas as pd

    out = []
    for met in metric_list:
        if met not in wide_a or met not in wide_b:
            continue
        d = (wide_a[met] - wide_b[met]).dropna()
        if d.empty:
            continue
        rel = (d / wide_b[met].reindex(d.index)).replace([np.inf, -np.inf], np.nan).dropna()
        better = (d > 0) if met == "corr_mean" else (d < 0)
        gg = d.groupby(level=keys)
        res = pd.DataFrame({"mean": gg.mean(), "se": gg.sem(), "dates": gg.size(),
                            "wins": better.groupby(level=keys).sum().astype(int),
                            "rel_mean": rel.groupby(level=keys).mean()})
        res["metric"] = met
        out.append(res.reset_index())
    return out


def paired(df, pairs=DEFAULT_PAIRS):
    """Paired differences a - b of two rows of the same run, per date, then over dates."""
    import pandas as pd

    mets = oriented_metrics(df)
    wide = df.pivot_table(index=KEYS + ["date"], columns="row", values=mets)
    out = []
    for a, b in pairs:
        if a not in set(df.row) or b not in set(df.row):
            continue
        wa = {m: wide[(m, a)] for m in mets if (m, a) in wide.columns}
        wb = {m: wide[(m, b)] for m in mets if (m, b) in wide.columns}
        for t in _paired_table(wa, wb, mets, KEYS):
            t.insert(len(KEYS), "pair", f"{a}-{b}")
            out.append(t)
    if not out:
        return pd.DataFrame(columns=KEYS + ["pair", "mean", "se", "dates", "wins", "rel_mean", "metric"])
    return pd.concat(out, ignore_index=True)


def compare_runs(scores_a, scores_b, *, row="model", row_b=None):
    """Paired differences of one row of run A minus one row of run B, matched on (var, variant, radius,
    region, lead, date). Returns (table, checks): checks reports dates present in one run only and
    station-count mismatches, which mean the two samples are not the same stations."""
    import pandas as pd

    row_b = row_b or row
    a = scores_a[scores_a.row == row].set_index(KEYS + ["date"])
    b = scores_b[scores_b.row == row_b].set_index(KEYS + ["date"])
    common = a.index.intersection(b.index)
    checks = {"only_in_a": int(len(a.index.difference(b.index))), "only_in_b": int(len(b.index.difference(a.index))),
              "n_mismatch": int((a.loc[common, "n"] != b.loc[common, "n"]).sum())}
    # Truth rows are the same field in both runs when both read the same dates: a difference flags a mix-up.
    ta, tb = scores_a[scores_a.row == "truth"], scores_b[scores_b.row == "truth"]
    if len(ta) and len(tb):
        t = ta.set_index(KEYS + ["date"])["fcrps"].sub(tb.set_index(KEYS + ["date"])["fcrps"]).dropna()
        checks["truth_fcrps_max_abs_diff"] = float(t.abs().max()) if len(t) else None
    mets = [m for m in oriented_metrics(scores_a) if m in scores_b.columns]
    a, b = a.loc[common], b.loc[common]
    tabs = _paired_table({m: a[m] for m in mets}, {m: b[m] for m in mets}, mets, KEYS)
    tab = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    if len(tab):
        mean_n = a["n"].groupby(level=KEYS).mean().rename("mean_n").reset_index()
        tab = tab.merge(mean_n, on=KEYS, how="left")
    return tab, checks


def _fmt(x, se=None, digits=3):
    if x is None or not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}" + (f" ({se:.{digits}f})" if se is not None and np.isfinite(se) else "")


def write_summary_md(summ, pr, meta, path, *, region="all"):
    """Headline tables (ensemble scores first) for one region, one block per variable and variant."""
    label = meta.get("run_label") or "model"
    unit_name = meta.get("sample_unit", "stations")
    leads = sorted(set(summ.lead))
    lines = [f"# {meta.get('evaluator', 'obs_points')}: {label} against {meta.get('reference', 'station observations')}", "",
             f"Region `{region}`. Mean over dates of the per-date mean over {unit_name}; SE over dates in brackets. "
             f"`model` is the {max(meta.get('members') or [0])}-member ensemble; "
             + ", ".join(f"`{r}`" for r in ("input", "truth", "hres") if r in set(summ.row))
             + " are single fields (their fair CRPS is the mean absolute error and their spread is zero). Deterministic "
             "columns (bias, RMSE, MAE and correlation of the ensemble mean) are in summary.csv, not here.",
             f"Dates: {len(meta.get('dates', []))}. Reference: `{meta.get('cache_dir') or meta.get('imerg_file')}`. "
             "Weighting: "
             f"{meta.get('config', {}).get('weighting', 'none')}.", ""]
    s = summ[summ.region == region]
    for (v, var_, r), q in s.groupby(["var", "variant", "radius_km"], sort=False):
        unit = VAR_UNITS.get(v, "")
        head = f"## {v} {var_}" + (f", disc radius {r:g} km" if r else "") + f" ({unit})"
        lines += [head, "", "| row | score | " + " | ".join(f"+{l} h" for l in leads) + " |",
                  "|---|---|" + "---|" * len(leads)]
        mets = [("fcrps", "fair CRPS"), ("ssr_corrected", "spread/skill (corrected)"), ("spread", "spread")]
        if "seeps" in q.columns and q["seeps"].notna().any():
            mets.append(("seeps", "SEEPS (mean over members)"))
        mets += [(c, tail_name(c)) for c in q.columns
                 if c.startswith(("twcrps", "brier")) and not c.endswith("_se") and q[c].notna().any()]
        for row in ("model", "input", "truth", "hres"):
            qq = q[q.row == row].set_index("lead")
            if qq.empty:
                continue
            for key, name in mets:
                if row != "model" and key in ("ssr_corrected", "spread"):
                    continue
                cells = [_fmt(qq.loc[l, key], qq.loc[l, f"{key}_se"]) if l in qq.index else "" for l in leads]
                tag = row if row == "model" else f"{row} (single field)"
                label = "SEEPS" if key == "seeps" and row != "model" else name
                lines.append(f"| {tag} | {label} | " + " | ".join(cells) + " |")
        nn = q[q.row == "model"].set_index("lead")
        lines.append(f"| {unit_name} per date | | " + " | ".join(f"{nn.loc[l, 'mean_n']:.0f}" if l in nn.index else ""
                                                        for l in leads) + " |")
        for c in sorted([c for c in q.columns if c.startswith("n_obs_gt") and not c.endswith("_se")
                             and q[c].notna().any()],
                        key=lambda c: float(c[8:])):
            lines.append(f"| observed > {c[8:]} mm per date | | " +
                         " | ".join(f"{nn.loc[l, c]:.1f}" if l in nn.index else "" for l in leads) + " |")
        p = pr[(pr.region == region) & (pr["var"] == v) & (pr.variant == var_) & (pr.radius_km == r)
               & (pr.metric == "fcrps")]
        for pair in ("model-input", "model-truth", "truth-input"):
            pp = p[p.pair == pair].set_index("lead")
            if pp.empty:
                continue
            cells = [f"{pp.loc[l, 'mean']:+.3f} ({pp.loc[l, 'se']:.3f}) w{int(pp.loc[l, 'wins'])}/{int(pp.loc[l, 'dates'])}"
                     if l in pp.index else "" for l in leads]
            lines.append(f"| {pair} | fair CRPS, paired | " + " | ".join(cells) + " |")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")


def default_region(regions) -> str:
    """'all' when the run has it (obs_points), else its first region (obs_imerg)."""
    regions = list(dict.fromkeys(regions))
    return "all" if "all" in regions or not regions else regions[0]


def write_compare_md(tab, meta, path, *, region=None):
    """Paired run comparison, headline metrics only, for one region."""
    region = region or (default_region(tab.region) if len(tab) else "all")
    la, lb, row = meta["label_a"], meta["label_b"], meta["row"]
    unit_name = meta.get("sample_unit", "stations")
    lines = [f"# {la} minus {lb} ({row} rows), paired by date", "",
             f"Region `{region}`. Per date the mean score over {unit_name} of {la} minus that of {lb}; mean over dates, "
             f"SE over dates in brackets, and wNN/MM = dates on which {la} is better. Negative is better for {la} on every "
             f"score shown. These are paired differences on the same {unit_name} and dates, not a verdict on their own.",
             f"Checks: {json.dumps(meta.get('checks', {}))}", ""]
    if tab.empty:
        Path(path).write_text("\n".join(lines + ["No common rows."]) + "\n")
        return
    t = tab[(tab.region == region) & (tab.metric.str.startswith(("fcrps", "twcrps", "brier", "seeps")))]
    leads = sorted(set(tab.lead))
    for (v, var_, r), q in t.groupby(["var", "variant", "radius_km"], sort=False):
        lines += [f"## {v} {var_}" + (f", disc radius {r:g} km" if r else "") + f" ({VAR_UNITS.get(v, '')})", "",
                  "| score | " + " | ".join(f"+{l} h" for l in leads) + " |", "|---|" + "---|" * len(leads)]
        for met, qq in q.groupby("metric", sort=False):
            qq = qq.set_index("lead")
            name = {"fcrps": "fair CRPS", "seeps": "SEEPS"}.get(met) or tail_name(met)
            lines.append(f"| {name} | " + " | ".join(
                f"{qq.loc[l, 'mean']:+.4f} ({qq.loc[l, 'se']:.4f}) w{int(qq.loc[l, 'wins'])}/{int(qq.loc[l, 'dates'])}"
                if l in qq.index else "" for l in leads) + " |")
        nn = q.drop_duplicates("lead").set_index("lead")
        if "mean_n" in nn:
            lines.append(f"| {unit_name} per date | " + " | ".join(f"{nn.loc[l, 'mean_n']:.0f}" if l in nn.index else ""
                                                             for l in leads) + " |")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")
