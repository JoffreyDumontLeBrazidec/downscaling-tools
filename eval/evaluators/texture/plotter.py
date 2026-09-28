"""Figures for the texture evaluator: one PNG per weather state.

Six panels per stratum: grouped bars (truth in black, model in red) for
fine_lag1_zonal, fine_nn_corr, top5_share and kurtosis; the fine-variance
ratio model/truth; and the grain index (model - truth) / (noise - truth) for
the two correlations. Error bars are the standard deviation over the (file,
member) samples. Grey ticks mark the white-noise reference of each stratum
(Gaussian noise pushed through the same fine-part operator, which is where
pure grain sits); dotted lines mark the ratio 1 (model equals truth) and the
grain-index values 0 (truth-like) and 1 (white noise).

Each figure is written as PNG (150 dpi) and PDF in the house style of ``eval.plotting``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.plotting import SEQUENCE, eval_style, reference_style, role_style, save_figure, shorten_run_label, variable_spec

LOG = logging.getLogger(__name__)

PANELS = [
    ("fine_lag1_zonal", "Lag-1 zonal correlation\nof the fine part", "pair"),
    ("fine_nn_corr", "Correlation of the fine part with\nthe mean of its 6 neighbours", "pair"),
    ("fine_var", "Fine-part variance ratio\n(model / truth)", "ratio"),
    ("top5_share", "Share of fine-part energy\nin the top 5 % of points", "pair"),
    ("kurtosis", "Excess kurtosis\nof the fine part", "pair"),
    ("grain_index", "Grain index\n(0 = truth-like, 1 = white noise)", "grain"),
]
TRUTH = role_style("truth")["color"]
MODEL = role_style("model")["color"]
NOISE = reference_style(0)["color"]          # white-noise reference: a neutral anchor
GRAIN = (SEQUENCE[0], SEQUENCE[2])          # two estimates of one metric: neutral colours


def _series(rows: dict, strata: list[str], side: str, stat: str):
    def _get(s, key):
        entry = (rows[s].get(side) or {}).get(stat)
        if not entry or entry.get(key) is None:
            return np.nan if key == "mean" else 0.0
        return entry[key]

    mean = np.array([_get(s, "mean") for s in strata], dtype=np.float64)
    sd = np.array([_get(s, "sd") for s in strata], dtype=np.float64)
    return mean, sd


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / "texture.json"
    if not path.exists():
        LOG.warning("texture: nothing to plot, %s missing", path)
        return output_dir
    payload = json.loads(path.read_text())
    aggregate = payload.get("aggregate", [])
    if not aggregate:
        return output_dir

    label = payload.get("run_label", "")
    strata_all = payload.get("strata_order", [])

    run = shorten_run_label(str(label or ""))
    run = "" if run == "predictions" else run   # the generic directory name says nothing
    for state in payload.get("states", []):
        rows = {r["stratum"]: r for r in aggregate if r["state"] == state}
        strata = [s for s in strata_all if s in rows]
        if not strata:
            continue
        x = np.arange(len(strata))
        w = 0.38
        n = max(r["n_samples"] for r in rows.values())
        with eval_style():
            fig, axes = plt.subplots(1, len(PANELS), figsize=(4.1 * len(PANELS), 5.6), layout="constrained")
            for ax, (stat, title, kind) in zip(axes, PANELS):
                if kind == "pair":
                    t_mean, t_sd = _series(rows, strata, "truth", stat)
                    m_mean, m_sd = _series(rows, strata, "model", stat)
                    n_mean, _ = _series(rows, strata, "noise", stat)
                    ax.bar(x - w / 2, t_mean, w, yerr=t_sd, color=TRUTH, capsize=2,
                           error_kw={"ecolor": "0.5"}, label=f"truth (n = {n})")
                    ax.bar(x + w / 2, m_mean, w, yerr=m_sd, color=MODEL, capsize=2,
                           label=f"model (n = {n})")
                    if np.any(np.isfinite(n_mean)):
                        ax.plot(x, n_mean, ls="none", marker="_", ms=22, mew=2.4, color=NOISE,
                                label="white noise, same filter")
                    ax.axhline(0.0, color="0.4", lw=0.6, ls=":")
                    if stat in ("fine_lag1_zonal", "fine_nn_corr"):
                        lo = float(np.nanmin(np.r_[t_mean, m_mean, n_mean, 0.0]))
                        ax.set_ylim(lo - 0.08, 1.0)
                elif kind == "ratio":
                    r_mean, r_sd = _series(rows, strata, "ratio", stat)
                    ax.bar(x, r_mean, 0.6, yerr=r_sd, color=MODEL, capsize=2,
                           label=f"model / truth (n = {n})")
                    ax.axhline(1.0, color="0.2", lw=0.9, ls=":", label="equal (1)")
                else:
                    g1_mean, g1_sd = _series(rows, strata, "grain_index", "fine_lag1_zonal")
                    g2_mean, g2_sd = _series(rows, strata, "grain_index", "fine_nn_corr")
                    ax.bar(x - w / 2, g1_mean, w, yerr=g1_sd, color=GRAIN[0], capsize=2,
                           label="from the lag-1 correlation")
                    ax.bar(x + w / 2, g2_mean, w, yerr=g2_sd, color=GRAIN[1], capsize=2,
                           label="from the neighbour correlation")
                    ax.axhline(0.0, color="0.2", lw=0.9, ls=":", label="truth-like (0)")
                    ax.axhline(1.0, color="0.2", lw=0.9, ls="--", label="white noise (1)")
                    ax.set_ylim(min(-0.1, float(np.nanmin(np.r_[g1_mean, g2_mean, 0.0])) - 0.05),
                                max(1.15, float(np.nanmax(np.r_[g1_mean, g2_mean, 1.0])) + 0.1))
                ax.set_xticks(x)
                ax.set_xticklabels([s.replace("_", " ") for s in strata], rotation=45, ha="right",
                                   fontsize=8)
                ax.set_title(title, fontsize=10)
                ax.grid(axis="x", visible=False)
                ax.legend(fontsize=7, loc="best")
            fig.suptitle(
                f"Texture of {variable_spec(state).name} on the native O1280 grid{', ' + run if run else ''}\n"
                f"error bars: one standard deviation over {n} samples (file x member)",
                fontsize=12,
            )
            out = output_dir / f"texture_{state}.png"
            save_figure(fig, out, close=True)
        LOG.info("texture: wrote %s (and .pdf)", out)
    return output_dir
