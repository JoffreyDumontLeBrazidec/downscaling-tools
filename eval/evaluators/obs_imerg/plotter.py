"""obs_imerg plot(): the obs_points score figures (fair CRPS, spread/skill, paired differences, tail scores,
rank histograms) with IMERG as the reference, plus the tp PDFs of IMERG, truth, input and model. Reads the
saved tables alone."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.evaluators.obs_points.plotter import plot_rank_hist, plot_scores, plot_tail


def plot_pdfs(results_dir, meta, regions):
    import matplotlib.pyplot as plt
    from eval.plotting import eval_style, reference_style, role_style, save_figure

    z = np.load(Path(results_dir) / "hist.npz")
    edges, leads, regs, rows, counts = z["edges"], z["leads"], z["regions"], z["rows"], z["counts"].astype(float)
    regions = [r for r in regions if r in set(regs)]
    ul = sorted(set(leads.tolist()))
    if not regions:
        return []
    centres = np.sqrt(edges[1:-1] * edges[2:])  # bin 0 = [0, 0.1) mm is the dry fraction, drawn as text
    widths = np.diff(edges)[1:]
    styles = {"imerg": reference_style(0, linewidth=2.6), "truth": role_style("truth"), "input": role_style("input"),
              "model": role_style("model")}
    names = {"imerg": "IMERG Final V07B", "truth": "Truth", "input": "Input (interpolated)",
             "model": f"{meta.get('run_label', 'model')} (all members)"}
    with eval_style():
        fig, axes = plt.subplots(len(regions), len(ul), figsize=(3.9 * len(ul), 3.0 * len(regions)),
                                 squeeze=False, sharex=True, sharey=True)
        for i, reg in enumerate(regions):
            for j, lead in enumerate(ul):
                ax = axes[i, j]
                dry = []
                for row in ("imerg", "truth", "input", "model"):
                    k = np.flatnonzero((leads == lead) & (regs == reg) & (rows == row))
                    if not k.size:
                        continue
                    c = counts[k[0]]
                    pdf = c[1:] / c.sum() / widths
                    ax.loglog(centres, np.where(pdf > 0, pdf, np.nan), label=names[row], **styles[row])
                    dry.append(f"{row} {c[0] / c.sum():.0%}")
                ax.set_title(f"{reg}, +{lead} h", fontsize=10)
                ax.text(0.03, 0.03, "below 0.1 mm: " + ", ".join(dry), transform=ax.transAxes, fontsize=6.5)
                if j == 0:
                    ax.set_ylabel("PDF (per mm)")
                if i == len(regions) - 1:
                    ax.set_xlabel("12 h precipitation (mm)")
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(f"12 h precipitation PDFs against IMERG: {meta.get('run_label')}, {len(meta.get('dates', []))} dates",
                     fontweight="bold", y=0.995)
        fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955), fontsize=9)
        fig.text(0.01, 0.0, "IMERG has less drizzle and a heavier tail than the IFS and is weak poleward of 60 N and over "
                 "snow; read the PDFs with that in mind.", ha="left", va="top", fontsize=8.5, color="#52514e")
        fig.tight_layout(rect=(0, 0.02, 1, 0.9))
        return save_figure(fig, Path(results_dir) / "obs_imerg_tp_pdfs", close=True)


def plot(results_dir, lane_config: dict, eval_config: dict, *, output_dir=None, **kwargs):
    import pandas as pd

    results_dir = Path(results_dir)
    meta = json.loads((results_dir / "run_meta.json").read_text())
    if eval_config.get("run_label"):
        meta["run_label"] = str(eval_config["run_label"])
    summ = pd.read_csv(results_dir / "summary.csv")
    pr = pd.read_csv(results_dir / "paired.csv")
    rk = results_dir / "rank_hist.csv"
    ranks = pd.read_csv(rk) if rk.exists() and rk.stat().st_size > 1 else pd.DataFrame()
    regions = [str(r) for r in eval_config.get("plot_regions", list(meta.get("regions", {}))[:2])]
    written = plot_scores(results_dir, summ, pr, meta, "tp", "imerg", 0.0, regions)
    for reg in regions:
        written += plot_tail(results_dir, summ, meta, 0.0, region=reg, stem=f"obs_imerg_tail_{reg}")
        if len(ranks):
            written += plot_rank_hist(results_dir, ranks, meta, "tp", "imerg", region=reg,
                                      stem=f"obs_imerg_rank_hist_{reg}")
    written += plot_pdfs(results_dir, meta, regions)
    return written
