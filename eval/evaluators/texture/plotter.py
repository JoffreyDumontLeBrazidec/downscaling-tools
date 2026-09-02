"""Figures for the texture evaluator: one PNG per weather state.

Five panels of grouped bars (truth in grey, model in red) per stratum:
fine_lag1_zonal, fine_nn_corr, fine_var as the model/truth ratio, top5_share
and kurtosis. Error bars are the standard deviation over the (file, member)
samples. Dotted reference lines mark what a Gaussian white-noise fine field
would give (0 correlation, 0 excess kurtosis, top-5% share of about 0.28) and
the ratio 1 where model equals truth.
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
]


def _gaussian_top_share(frac: float) -> float:
    """Share of sum(z^2) carried by the top `frac` of |z| for standard normal z."""
    from scipy.stats import norm
    zq = norm.ppf(1.0 - frac / 2.0)
    return float(2.0 * (zq * norm.pdf(zq) + (1.0 - norm.cdf(zq))))


def _series(rows: dict, strata: list[str], side: str, stat: str):
    mean = np.array(
        [np.nan if rows[s][side][stat]["mean"] is None else rows[s][side][stat]["mean"] for s in strata],
        dtype=np.float64,
    )
    sd = np.array(
        [0.0 if rows[s][side][stat]["sd"] is None else rows[s][side][stat]["sd"] for s in strata],
        dtype=np.float64,
    )
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
    frac = float(payload.get("config", {}).get("top_fraction", 0.05))
    gauss_share = _gaussian_top_share(frac)

    for state in payload.get("states", []):
        rows = {r["stratum"]: r for r in aggregate if r["state"] == state}
        strata = [s for s in strata_all if s in rows]
        if not strata:
            continue
        x = np.arange(len(strata))
        w = 0.38
        fig, axes = plt.subplots(1, len(PANELS), figsize=(4.3 * len(PANELS), 4.8))
        for ax, (stat, title, kind) in zip(axes, PANELS):
            if kind == "pair":
                t_mean, t_sd = _series(rows, strata, "truth", stat)
                m_mean, m_sd = _series(rows, strata, "model", stat)
                ax.bar(x - w / 2, t_mean, w, yerr=t_sd, color="0.55", capsize=2, label="truth")
                ax.bar(x + w / 2, m_mean, w, yerr=m_sd, color="tab:red", capsize=2, label="model")
                if stat == "top5_share":
                    ax.axhline(gauss_share, color="k", lw=0.8, ls=":", label=f"Gaussian noise ({gauss_share:.2f})")
                else:
                    ax.axhline(0.0, color="k", lw=0.8, ls=":", label="Gaussian noise (0)")
                if stat in ("fine_lag1_zonal", "fine_nn_corr"):
                    ax.set_ylim(min(-0.1, float(np.nanmin(np.r_[t_mean, m_mean])) - 0.05), 1.0)
            else:
                r_mean, r_sd = _series(rows, strata, "ratio", stat)
                ax.bar(x, r_mean, 0.6, yerr=r_sd, color="tab:red", capsize=2, label="model / truth")
                ax.axhline(1.0, color="k", lw=0.8, ls=":", label="equal (1)")
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
