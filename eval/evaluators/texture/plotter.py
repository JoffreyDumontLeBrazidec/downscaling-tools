"""Figures for the texture evaluator: one PNG per weather state.

Six panels per stratum: grouped bars (truth in grey, model in red) for
fine_lag1_zonal, fine_nn_corr, top5_share and kurtosis; the fine-variance
ratio model/truth; and the grain index (model - truth) / (noise - truth) for
the two correlations. Error bars are the standard deviation over the (file,
member) samples. Black ticks mark the white-noise reference of each stratum
(Gaussian noise pushed through the same fine-part operator, which is where
pure grain sits); dotted lines mark the ratio 1 (model equals truth) and the
grain-index values 0 (truth-like) and 1 (white noise).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = logging.getLogger(__name__)

PANELS = [
    ("fine_lag1_zonal", "lag-1 zonal correlation of the fine part", "pair"),
    ("fine_nn_corr", "corr(fine part, mean of 6 neighbours)", "pair"),
    ("fine_var", "fine variance, model / truth", "ratio"),
    ("top5_share", "share of fine energy in the top 5% of points", "pair"),
    ("kurtosis", "excess kurtosis of the fine part", "pair"),
    ("grain_index", "grain index: 0 = truth-like, 1 = white noise", "grain"),
]


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

    for state in payload.get("states", []):
        rows = {r["stratum"]: r for r in aggregate if r["state"] == state}
        strata = [s for s in strata_all if s in rows]
        if not strata:
            continue
        x = np.arange(len(strata))
        w = 0.38
        fig, axes = plt.subplots(1, len(PANELS), figsize=(4.1 * len(PANELS), 4.8))
        for ax, (stat, title, kind) in zip(axes, PANELS):
            if kind == "pair":
                t_mean, t_sd = _series(rows, strata, "truth", stat)
                m_mean, m_sd = _series(rows, strata, "model", stat)
                n_mean, _ = _series(rows, strata, "noise", stat)
                ax.bar(x - w / 2, t_mean, w, yerr=t_sd, color="0.55", capsize=2, label="truth")
                ax.bar(x + w / 2, m_mean, w, yerr=m_sd, color="tab:red", capsize=2, label="model")
                if np.any(np.isfinite(n_mean)):
                    ax.plot(x, n_mean, ls="none", marker="_", ms=22, mew=2.0, color="k",
                            label="white noise, same filter")
                ax.axhline(0.0, color="0.4", lw=0.6, ls=":")
                if stat in ("fine_lag1_zonal", "fine_nn_corr"):
                    lo = float(np.nanmin(np.r_[t_mean, m_mean, n_mean, 0.0]))
                    ax.set_ylim(lo - 0.08, 1.0)
            elif kind == "ratio":
                r_mean, r_sd = _series(rows, strata, "ratio", stat)
                ax.bar(x, r_mean, 0.6, yerr=r_sd, color="tab:red", capsize=2, label="model / truth")
                ax.axhline(1.0, color="k", lw=0.8, ls=":", label="equal (1)")
            else:
                g1_mean, g1_sd = _series(rows, strata, "grain_index", "fine_lag1_zonal")
                g2_mean, g2_sd = _series(rows, strata, "grain_index", "fine_nn_corr")
                ax.bar(x - w / 2, g1_mean, w, yerr=g1_sd, color="tab:orange", capsize=2, label="from lag-1")
                ax.bar(x + w / 2, g2_mean, w, yerr=g2_sd, color="tab:purple", capsize=2, label="from nn_corr")
                ax.axhline(0.0, color="k", lw=0.8, ls=":", label="truth-like (0)")
                ax.axhline(1.0, color="k", lw=0.8, ls="--", label="white noise (1)")
                ax.set_ylim(min(-0.1, float(np.nanmin(np.r_[g1_mean, g2_mean, 0.0])) - 0.05),
                            max(1.15, float(np.nanmax(np.r_[g1_mean, g2_mean, 1.0])) + 0.1))
            ax.set_xticks(x)
            ax.set_xticklabels(strata, rotation=45, ha="right", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=7, loc="best")
        n = max(r["n_samples"] for r in rows.values())
        fig.suptitle(
            f"Texture on the native O1280 grid -- {state} -- {label} "
            f"(error bars: sd over {n} file x member samples)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = output_dir / f"texture_{state}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        LOG.info("texture: wrote %s", out)
    return output_dir
