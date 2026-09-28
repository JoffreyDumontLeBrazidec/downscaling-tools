"""Leadtime evaluator — multi-panel diagnostic plots.

Produces ``leadtime_plots.pdf`` containing:

  Page 1  — Surface nMSE vs leadtime, all variables, global
             Twin-axis: nMSE full-field (left) + skill vs interp (right).

  Page 2-5 — Same per region (tropics / NH extra / SH extra), one page each.

  Page 6  — Skill-vs-interp heatmap: variable × leadtime, all regions overlaid.

  Page 7  — Full-field power spectra per variable, one curve per leadtime +
             truth (black dashed) and input O96 (brown dotted).

  Page 8  — Residual (correction) spectra: y_pred−x_interp vs y−x_interp,
             one curve per leadtime.

  Page 9  — Spectral relative-L2 vs truth by wavenumber band (large/meso/fine),
             grouped bar per variable, coloured by leadtime.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

# Surface variable display order and labels
_SFC_VARS = ["10u", "10v", "2t", "2d", "msl", "sp", "skt", "tcw"]
_SFC_LABELS = {
    "10u": "10u [m/s]", "10v": "10v [m/s]", "2t": "2t [K]", "2d": "2d [K]",
    "msl": "MSLP [Pa]", "sp": "SP [Pa]", "skt": "SKT [K]", "tcw": "TCW [kg/m²]",
}
_SPEC_VARS = ["10u", "10v", "2t", "msl", "t_850", "z_500"]
_SPEC_LABELS = {
    "10u": "10u", "10v": "10v", "2t": "2t", "msl": "MSLP",
    "t_850": "T850", "z_500": "Z500",
}
_REGION_LABELS = {
    "global": "Global", "tropics": "Tropics (±20°)",
    "nh_extra": "NH Extratropics (>20N)", "sh_extra": "SH Extratropics (<20S)",
}
_STEP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
_BANDS = [(3, 30, "large scale ℓ<30"), (30, 100, "meso ℓ30-100"), (100, None, "fine ℓ>100")]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _get_step_region_var(data: dict, step: int, region: str, var: str, key: str) -> float | None:
    v = (
        data.get("by_step", {})
        .get(str(step), {})
        .get(region, {})
        .get(var, {})
        .get(key)
    )
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _spectra_mean_cl(data: dict, step: int, var: str, curve: str) -> np.ndarray | None:
    raw = data.get("spectra", {}).get(str(step), {}).get(var, {}).get(curve)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    return arr if arr.size > 0 else None


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------
def _page_surface_nMSE(
    fig,
    axes,
    data: dict,
    steps: list[int],
    region: str,
    run_label: str,
) -> None:
    """Fill axes grid: nMSE_full + skill_vs_interp vs leadtime, one axis per var."""
    vars_present = [v for v in _SFC_VARS if any(
        _get_step_region_var(data, s, region, v, "nmse_full") is not None for s in steps
    )]
    step_labels = [f"{s}h" for s in steps]

    for ax_i, var in enumerate(vars_present):
        ax = axes[ax_i // 4][ax_i % 4]
        nmse_full = [_get_step_region_var(data, s, region, var, "nmse_full") for s in steps]
        nmse_res = [_get_step_region_var(data, s, region, var, "nmse_residual") for s in steps]
        skill = [_get_step_region_var(data, s, region, var, "skill_vs_interp") for s in steps]

        ax2 = ax.twinx()
        ax.plot(step_labels, nmse_full, "o-", color="#1f77b4", lw=2.2, ms=5, label="nMSE full")
        ax.plot(step_labels, nmse_res, "s--", color="#ff7f0e", lw=1.6, ms=4, label="nMSE residual")
        ax2.plot(step_labels, skill, "^:", color="#2ca02c", lw=1.6, ms=4, label="skill vs interp")
        ax2.axhline(0, color="0.55", lw=0.8, ls=":")

        ax.set_title(_SFC_LABELS.get(var, var), fontsize=8, pad=3)
        ax.set_ylabel("nMSE", fontsize=7)
        ax2.set_ylabel("skill", fontsize=7, color="#2ca02c")
        ax2.tick_params(axis="y", colors="#2ca02c", labelsize=6)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="0.85", lw=0.5)

    # Hide unused axes
    for ax_i in range(len(vars_present), axes.size):
        axes.flat[ax_i].set_visible(False)

    rlabel = _REGION_LABELS.get(region, region)
    fig.suptitle(f"Surface nMSE vs leadtime — {rlabel}\n{run_label}", fontsize=10, fontweight="bold")

    # Shared legend below the grid
    h = [
        __import__("matplotlib.lines", fromlist=["Line2D"]).Line2D([0], [0], color="#1f77b4", lw=2, label="nMSE full field"),
        __import__("matplotlib.lines", fromlist=["Line2D"]).Line2D([0], [0], color="#ff7f0e", lw=1.6, ls="--", label="nMSE residual"),
        __import__("matplotlib.lines", fromlist=["Line2D"]).Line2D([0], [0], color="#2ca02c", lw=1.6, ls=":", marker="^", ms=4, label="skill vs interp"),
    ]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))


def _page_skill_heatmap(
    fig, ax, data: dict, steps: list[int], run_label: str,
) -> None:
    """Heatmap: variable × leadtime, annotated skill_vs_interp values, all regions as subplots."""
    import matplotlib.pyplot as plt
    regions = [r for r in ["global", "tropics", "nh_extra", "sh_extra"]]
    n_regions = len(regions)
    step_labels = [f"{s}h" for s in steps]

    for ri, region in enumerate(regions):
        cur_ax = ax if n_regions == 1 else ax[ri]
        vars_present = [v for v in _SFC_VARS if any(
            _get_step_region_var(data, s, region, v, "skill_vs_interp") is not None for s in steps
        )]
        if not vars_present:
            cur_ax.set_visible(False)
            continue
        mat = np.array([
            [_get_step_region_var(data, s, region, v, "skill_vs_interp") or float("nan") for s in steps]
            for v in vars_present
        ])
        vmax = max(np.nanmax(np.abs(mat)), 0.01)
        im = cur_ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        cur_ax.set_xticks(range(len(steps)))
        cur_ax.set_xticklabels(step_labels, fontsize=8)
        cur_ax.set_yticks(range(len(vars_present)))
        cur_ax.set_yticklabels(vars_present, fontsize=8)
        cur_ax.set_title(_REGION_LABELS.get(region, region), fontsize=9, fontweight="bold")
        for i in range(len(vars_present)):
            for j in range(len(steps)):
                v = mat[i, j]
                if not np.isnan(v):
                    cur_ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(v) > 0.6 * vmax else "black")
        plt.colorbar(im, ax=cur_ax, shrink=0.85, label="skill vs interp")

    fig.suptitle(f"Skill vs interpolated-input baseline\n{run_label}", fontsize=10, fontweight="bold")


def _page_spectra(
    fig, axes, data: dict, steps: list[int], run_label: str,
    pred_key: str, truth_key: str, input_key: str | None,
    page_title: str,
    lmax: int,
) -> None:
    """Power spectra page: one panel per var, one curve per leadtime."""
    ell = np.arange(lmax + 1, dtype=np.float64)
    spectra_vars = [v for v in _SPEC_VARS if any(
        _spectra_mean_cl(data, s, v, pred_key) is not None for s in steps
    )]

    for ax_i, var in enumerate(spectra_vars):
        ax = axes.flat[ax_i]
        for si, step in enumerate(steps):
            color = _STEP_COLORS[si % len(_STEP_COLORS)]
            pred_cl = _spectra_mean_cl(data, step, var, pred_key)
            if pred_cl is None:
                continue
            n = min(len(pred_cl), len(ell))
            keep = (ell[:n] >= 3) & np.isfinite(pred_cl[:n]) & (pred_cl[:n] > 0)

            if si == 0:
                truth_cl = _spectra_mean_cl(data, step, var, truth_key)
                if truth_cl is not None:
                    nt = min(len(truth_cl), len(ell))
                    kt = (ell[:nt] >= 3) & np.isfinite(truth_cl[:nt]) & (truth_cl[:nt] > 0)
                    ax.plot(ell[:nt][kt], truth_cl[:nt][kt], color="k", lw=2.2, ls="--", label="truth", zorder=5)
                if input_key is not None:
                    inp_cl = _spectra_mean_cl(data, step, var, input_key)
                    if inp_cl is not None:
                        ni = min(len(inp_cl), len(ell))
                        ki = (ell[:ni] >= 3) & np.isfinite(inp_cl[:ni]) & (inp_cl[:ni] > 0)
                        ax.plot(ell[:ni][ki], inp_cl[:ni][ki], color="#a6611a", lw=1.5, ls=":", label="input O96", zorder=4)

            ax.plot(ell[:n][keep], pred_cl[:n][keep], color=color, lw=1.8, label=f"{step}h", alpha=0.85)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavenumber ℓ", fontsize=8)
        ax.set_ylabel("Mean power Cℓ", fontsize=8)
        ax.set_title(_SPEC_LABELS.get(var, var), fontsize=9)
        ax.legend(fontsize=7, loc="lower left", frameon=False)
        ax.grid(color="0.85", ls="--", lw=0.3)
        ax.tick_params(labelsize=7)

    for ax_i in range(len(spectra_vars), axes.size):
        axes.flat[ax_i].set_visible(False)

    fig.suptitle(f"{page_title}\n{run_label}", fontsize=10, fontweight="bold")


def _page_spectra_band_barplot(
    fig, axes, data: dict, steps: list[int], run_label: str, lmax: int,
) -> None:
    """Grouped bar: relative-L2 vs truth by wavenumber band, one panel per var."""
    ell = np.arange(lmax + 1, dtype=np.float64)
    spectra_vars = [v for v in _SPEC_VARS if any(
        _spectra_mean_cl(data, s, v, "pred_cl") is not None for s in steps
    )]
    step_labels = [f"{s}h" for s in steps]
    band_colors = ["#4e79a7", "#f28e2b", "#e15759"]

    for ax_i, var in enumerate(spectra_vars):
        ax = axes.flat[ax_i]
        x = np.arange(len(steps), dtype=float)
        bar_w = 0.25

        for bi, (lo, hi, blabel) in enumerate(_BANDS):
            hi_eff = hi if hi is not None else lmax
            rl2_vals = []
            for step in steps:
                pred_cl = _spectra_mean_cl(data, step, var, "pred_cl")
                truth_cl = _spectra_mean_cl(data, step, var, "truth_cl")
                if pred_cl is None or truth_cl is None:
                    rl2_vals.append(float("nan"))
                    continue
                n = min(len(pred_cl), len(truth_cl), len(ell))
                m = ((ell[:n] > lo) & (ell[:n] <= hi_eff)
                     & np.isfinite(pred_cl[:n]) & np.isfinite(truth_cl[:n]) & (truth_cl[:n] > 0))
                if not m.any():
                    rl2_vals.append(float("nan"))
                    continue
                p, t = pred_cl[:n][m], truth_cl[:n][m]
                rl2_vals.append(float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-12)))

            ax.bar(x + bi * bar_w, rl2_vals, bar_w, label=blabel,
                   color=band_colors[bi], alpha=0.82)

        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(step_labels, fontsize=8)
        ax.set_ylabel("Rel-L2 vs truth", fontsize=8)
        ax.set_title(_SPEC_LABELS.get(var, var), fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="0.85", lw=0.5)

    for ax_i in range(len(spectra_vars), axes.size):
        axes.flat[ax_i].set_visible(False)

    fig.suptitle(f"Spectral relative-L2 vs truth by wavenumber band\n{run_label}",
                 fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    run_label: str = "",
    **kwargs,
) -> Path:
    """Generate leadtime_plots.pdf from leadtime_scores.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D  # noqa: F401  (used via __import__ trick above)

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "leadtime_scores.json"
    if not json_path.exists():
        LOG.warning("leadtime_scores.json not found, skipping plot: %s", json_path)
        return output_dir

    data = _load_json(json_path)
    steps: list[int] = data.get("steps", [])
    lmax: int = int(data.get("lmax", 319))
    has_spectra = bool(data.get("spectra_vars"))
    label = run_label or lane_config.get("name", results_dir.parent.name)

    pdf_path = output_dir / "leadtime_plots.pdf"
    LOG.info("Writing leadtime plots: %s", pdf_path)

    with PdfPages(pdf_path) as pdf:
        # Pages 1-4: nMSE per region
        for region in ["global", "tropics", "nh_extra", "sh_extra"]:
            fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
            _page_surface_nMSE(fig, axes, data, steps, region, label)
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 5: skill heatmaps (2×2 grid, one per region)
        fig, ax_grid = plt.subplots(2, 2, figsize=(14, 9))
        _page_skill_heatmap(fig, ax_grid.reshape(-1), data, steps, label)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if has_spectra:
            # Page 6: full-field spectra
            ncols = 3
            nrows = math.ceil(len(_SPEC_VARS) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
            _page_spectra(
                fig, axes, data, steps, label,
                pred_key="pred_cl", truth_key="truth_cl", input_key=None,
                page_title="Full-field power spectra (prediction vs truth)",
                lmax=lmax,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Page 7: residual spectra
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
            _page_spectra(
                fig, axes, data, steps, label,
                pred_key="residual_pred_cl", truth_key="residual_truth_cl", input_key=None,
                page_title="Residual (correction) spectra: y_pred − x_interp  vs  y − x_interp",
                lmax=lmax,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Page 8: spectral band bar plots
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows))
            _page_spectra_band_barplot(fig, axes, data, steps, label, lmax=lmax)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    LOG.info("Leadtime plots written to %s", pdf_path)
    return output_dir
