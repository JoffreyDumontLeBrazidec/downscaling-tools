"""obs_points plot(): every figure from the files run() wrote (summary.csv, paired.csv, rank_hist.csv,
run_meta.json). Ensemble scores are the headline: fair CRPS, corrected spread/skill, paired differences,
rank histograms and, for tp, the tail scores. Colours by role: model red, truth black, input blue dashed,
HRES grey. Every title carries the number of dates and the mean number of stations per date."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from eval.evaluators.obs_points.core.summary import VAR_UNITS, tail_name

LOG = logging.getLogger(__name__)

ROW_ROLE = {"model": "model", "truth": "truth", "input": "input", "hres": "baseline"}
ROW_OFFSET = {"model": 0.0, "truth": -1.5, "input": 1.5, "hres": 3.0}
VAR_TITLE = {"2t": "2 m temperature", "10ff": "10 m wind speed", "msl": "Mean sea-level pressure",
             "tp": "12 h precipitation"}
VARIANT_TITLE = {"raw": "nearest grid point", "lapse": "nearest grid point, lapse-rate height correction",
                 "imerg": "IMERG Final V07B on the same O1280 points"}


def _words(meta):
    """(where, unit) wording of the reference: stations by default; obs_imerg sets its own in run_meta.json."""
    return meta.get("where", "at stations"), meta.get("sample_unit", "stations")


def _row_label(row, run_label):
    return {"model": f"{run_label} (ensemble)", "truth": "Truth (single field)",
            "input": "Input, interpolated (single field)", "hres": "HRES (single field)"}[row]


def _footer(fig, text):
    fig.text(0.01, 0.0, text, ha="left", va="top", fontsize=8.5, color="#52514e")


def _lead_axis(ax, leads):
    ax.set_xticks(leads)
    ax.set_xticklabels([f"+{l}" for l in leads])
    ax.set_xlim(min(leads) - 8, max(leads) + 8)


def _series(ax, q, key, row, run_label, leads):
    from eval.plotting import role_style

    qq = q[q.row == row].set_index("lead").reindex(leads)
    if qq[key].isna().all():
        return
    m, se = qq[key].to_numpy(float), qq[f"{key}_se"].to_numpy(float)
    x = np.array(leads, float)
    st = role_style(ROW_ROLE[row])
    if row == "model":
        ax.plot(x, m, marker="o", label=_row_label(row, run_label), **st)
        ax.fill_between(x, m - se, m + se, color=st["color"], alpha=0.2, lw=0)
    else:
        st.update(linestyle="none")
        ax.errorbar(x + ROW_OFFSET[row], m, yerr=se, marker="s", markersize=5, capsize=3,
                    label=_row_label(row, run_label), **st)


def plot_scores(results_dir, summ, pr, meta, var, variant, radius, regions):
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, save_figure

    run_label = meta.get("run_label") or "model"
    where, unit_name = _words(meta)
    s = summ[(summ["var"] == var) & (summ.variant == variant) & (summ.radius_km == radius)]
    regions = [r for r in regions if r in set(s.region)]
    if not regions:
        return []
    leads = sorted(set(s.lead))
    unit = VAR_UNITS.get(var, "")
    panels = [("fcrps", f"Fair CRPS ({unit})"), ("ssr_corrected", "Spread / skill, corrected"),
              ("paired", f"Paired fair CRPS difference ({unit})")]
    with eval_style():
        fig, axes = plt.subplots(len(panels), len(regions), figsize=(5.4 * len(regions), 3.0 * len(panels)),
                                 squeeze=False, sharex=True)
        for j, reg in enumerate(regions):
            q = s[s.region == reg]
            nd = int(q[q.row == "model"].dates.max())
            ns = float(q[q.row == "model"].mean_n.mean())
            for i, (key, ylabel) in enumerate(panels):
                ax = axes[i, j]
                if key == "fcrps":
                    for row in ("model", "truth", "input", "hres"):
                        if row in set(q.row):
                            _series(ax, q, key, row, run_label, leads)
                    ax.set_title(f"{reg}: {nd} dates, ~{ns:,.0f} {unit_name} per date", fontsize=10.5)
                elif key == "ssr_corrected":
                    _series(ax, q, key, "model", run_label, leads)
                    ax.axhline(1.0, color="#0b0b0b", lw=1, ls="--")
                else:
                    p = pr[(pr["var"] == var) & (pr.variant == variant) & (pr.radius_km == radius)
                           & (pr.region == reg) & (pr.metric == "fcrps")]
                    for k, (pair, colour) in enumerate((("model-input", "#1f77b4"), ("model-truth", "#000000"),
                                                        ("model-hres", "#4d4d4d"))):
                        pp = p[p.pair == pair].set_index("lead").reindex(leads)
                        if pp["mean"].isna().all():
                            continue
                        x = np.array(leads, float) + 1.5 * (k - 1)
                        ax.errorbar(x, pp["mean"], yerr=pp["se"], color=colour, marker="o", ls="none", capsize=3,
                                    label=f"{run_label} minus {pair.split('-')[1]}")
                        for xx, yy, w, d in zip(x, pp["mean"], pp["wins"], pp["dates"]):
                            if np.isfinite(yy):
                                ax.annotate(f"{int(w)}/{int(d)}", (xx, yy), textcoords="offset points",
                                            xytext=(4, 4), fontsize=7, color=colour)
                    ax.axhline(0.0, color="#0b0b0b", lw=1)
                    ax.legend(fontsize=8, loc="best")
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == len(panels) - 1:
                    _lead_axis(ax, leads)
                    ax.set_xlabel("Lead time (h)")
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95), fontsize=9.5)
        rad = f", mean over a {radius:g} km disc" if radius else ""
        fig.suptitle(f"{VAR_TITLE.get(var, var)} {where} ({VARIANT_TITLE.get(variant, variant)}{rad}): "
                     f"{run_label}", fontweight="bold")
        _footer(fig, f"Mean over dates of the per-date mean over {unit_name}; bands and bars are one standard error over dates. "
                "A single field's fair CRPS is its mean absolute error, so the ensemble is lower partly by "
                "construction.\nSpread/skill = sqrt(mean member variance) / RMSE of the ensemble mean x sqrt((M+1)/M); "
                "1 is a calibrated ensemble. Numbers by the paired differences: dates on which the model is better.")
        fig.tight_layout(rect=(0, 0.03, 1, 0.92))
        tag = f"_r{radius:g}km" if radius else ""
        return save_figure(fig, Path(results_dir) / f"obs_{var}_{variant}{tag}_scores", close=True)


def plot_tail(results_dir, summ, meta, radius, region="all", stem=None):
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, save_figure

    run_label = meta.get("run_label") or "model"
    where, unit_name = _words(meta)
    s = summ[(summ["var"] == "tp") & (summ.radius_km == radius) & (summ.region == region)]
    thr = sorted({c[8:] for c in s.columns if c.startswith("n_obs_gt") and not c.endswith("_se")
                  and s[c].notna().any()}, key=float)
    if s.empty or not thr:
        return []
    leads = sorted(set(s.lead))
    with eval_style():
        fig, axes = plt.subplots(2, len(thr), figsize=(4.8 * len(thr), 6.2), squeeze=False, sharex=True)
        for j, t in enumerate(thr):
            nobs = s[s.row == "model"].set_index("lead").reindex(leads)[f"n_obs_gt{t}"]
            for i, key in enumerate((f"twcrps{t}", f"brier{t}")):
                ax = axes[i, j]
                for row in ("model", "truth", "input", "hres"):
                    if row in set(s.row):
                        _series(ax, s, key, row, run_label, leads)
                ax.set_ylabel(tail_name(key) + (" (mm)" if key.startswith("twcrps") else ""))
                if i == 0:
                    ax.set_title(f"> {t} mm\nobserved exceedances per date, by lead: "
                                 + ", ".join(f"{n:.0f}" for n in nobs.to_numpy(float)), fontsize=9.5)
                else:
                    _lead_axis(ax, leads)
                    ax.set_xlabel("Lead time (h)")
        nd = int(s[s.row == "model"].dates.max())
        ns = float(s[s.row == "model"].mean_n.mean())
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.94), fontsize=9.5)
        rad = f", {radius:g} km disc" if radius else ""
        fig.suptitle(f"12 h precipitation tail {where} ({region}{rad}): {run_label}, {nd} dates, "
                     f"~{ns:,.0f} {unit_name} per date", fontweight="bold")
        _footer(fig, "Fair threshold-weighted CRPS above t = fair CRPS of max(x, t); fair Brier score of exceeding t. "
                "Lower is better. Few observed exceedances make the tail scores noisy: read them with the counts.")
        fig.tight_layout(rect=(0, 0.03, 1, 0.88))
        tag = f"_r{radius:g}km" if radius else ""
        return save_figure(fig, Path(results_dir) / (stem or f"obs_tp_tail{tag}"), close=True)


def plot_rank_hist(results_dir, ranks, meta, var, variant, region="all", stem=None):
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, save_figure

    run_label = meta.get("run_label") or "model"
    where, unit_name = _words(meta)
    q = ranks[(ranks["var"] == var) & (ranks.variant == variant) & (ranks.radius_km == 0) & (ranks.region == region)
              & (ranks.row == "model")]
    if q.empty:
        return []
    m = int(q.members.max())
    cols = [f"r{i}" for i in range(m + 1)]
    leads = sorted(set(q.lead))
    with eval_style():
        fig, axes = plt.subplots(1, len(leads), figsize=(2.9 * len(leads), 3.3), squeeze=False, sharey=True)
        for j, lead in enumerate(leads):
            qq = q[q.lead == lead]
            c = qq[cols].sum().to_numpy(float)
            ax = axes[0, j]
            ax.bar(np.arange(m + 1), c / c.sum(), color="#d62728", alpha=0.8, width=0.85)
            ax.axhline(1.0 / (m + 1), color="#0b0b0b", lw=1, ls="--")
            ax.set_title(f"+{lead} h: {qq.date.nunique()} dates,\n{int(c.sum()):,} values", fontsize=9.5)
            ax.set_xlabel("Rank of the observation")
            if j == 0:
                ax.set_ylabel("Frequency")
        fig.suptitle(f"Rank histograms {where}, {VAR_TITLE.get(var, var)} ({VARIANT_TITLE.get(variant, variant)}, "
                     f"{region}): {run_label}, {m} members, ~{q.n.mean():,.0f} {unit_name} per date", fontweight="bold",
                     fontsize=11)
        _footer(fig, "Dashed line: the flat histogram of a calibrated ensemble. U shape = too little spread or a bias "
                f"{where} (observation error and representativeness also widen the outer bins). Ties broken at random.")
        fig.tight_layout(rect=(0, 0.04, 1, 0.88))
        return save_figure(fig, Path(results_dir) / (stem or f"obs_rank_hist_{var}_{variant}"), close=True)


def plot(results_dir, lane_config: dict, eval_config: dict, *, output_dir=None, **kwargs):
    import pandas as pd

    results_dir = Path(results_dir)
    meta = json.loads((results_dir / "run_meta.json").read_text())
    if eval_config.get("run_label"):
        meta["run_label"] = str(eval_config["run_label"])
    summ = pd.read_csv(results_dir / "summary.csv")
    pr = pd.read_csv(results_dir / "paired.csv")
    rk_path = results_dir / "rank_hist.csv"
    ranks = pd.read_csv(rk_path) if rk_path.exists() and rk_path.stat().st_size > 1 else pd.DataFrame()
    regions = [str(r) for r in eval_config.get("plot_regions", ("all", "europe"))]
    written = []
    for (var, variant, radius), _ in summ.groupby(["var", "variant", "radius_km"], sort=False):
        written += plot_scores(results_dir, summ, pr, meta, var, variant, float(radius), regions)
        if var == "tp":
            written += plot_tail(results_dir, summ, meta, float(radius))
        if radius == 0 and len(ranks):
            written += plot_rank_hist(results_dir, ranks, meta, var, variant)
    LOG.info("obs_points: %d figure files written in %s", len(written), results_dir)
    return written


def plot_compare(out_dir, tab, meta, region="all"):
    """Paired run comparison by lead: one panel per (variable, variant) for fair CRPS, plus the tp tail
    scores; points are the mean difference A - B over dates, bars one SE, labels the date wins of A."""
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, save_figure

    if tab.empty:
        return []
    la, lb = meta["label_a"], meta["label_b"]
    where, unit_name = _words(meta)
    t = tab[(tab.region == region) & (tab.radius_km == 0)]
    panels = [(v, var_, "fcrps") for (v, var_), _ in t.groupby(["var", "variant"], sort=False)]
    panels += [("tp", "raw", m) for m in sorted(set(t[t["var"] == "tp"].metric))
               if m.startswith(("twcrps", "brier"))]
    leads = sorted(set(t.lead))
    ncol = min(4, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    with eval_style():
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False)
        for k, (v, var_, met) in enumerate(panels):
            ax = axes.flat[k]
            q = t[(t["var"] == v) & (t.variant == var_) & (t.metric == met)].set_index("lead").reindex(leads)
            ax.errorbar(leads, q["mean"], yerr=q["se"], color="#d62728", marker="o", ls="none", capsize=3)
            for x, y, w, d in zip(leads, q["mean"], q["wins"], q["dates"]):
                if np.isfinite(y):
                    ax.annotate(f"{int(w)}/{int(d)}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax.axhline(0.0, color="#0b0b0b", lw=1)
            name = "fair CRPS" if met == "fcrps" else tail_name(met)
            ns = q["mean_n"].mean() if "mean_n" in q else np.nan
            ax.set_title(f"{v} {var_}: {name}\n{int(q['dates'].max())} dates, ~{ns:,.0f} {unit_name}", fontsize=9.5)
            _lead_axis(ax, leads)
            ax.set_xlabel("Lead time (h)")
            unit = "" if met.startswith("brier") else f" ({VAR_UNITS.get(v, '')})"
            ax.set_ylabel(f"{la} - {lb}{unit}")
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(f"{la} minus {lb} {where} ({meta['row']} rows, {region}), paired by date", fontweight="bold")
        _footer(fig, f"Below zero = {la} better. Bars: one SE over dates. Labels: dates on which {la} is better. "
                "Ensemble scores only; deterministic differences are in paired_runs.csv.")
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        return save_figure(fig, Path(out_dir) / "paired_runs", close=True)
