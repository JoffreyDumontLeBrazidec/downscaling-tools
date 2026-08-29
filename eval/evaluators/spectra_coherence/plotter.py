"""Figures for the spectra-coherence evaluator.

Two rows per weather state:
  top    -- amplitude ratio R(l) and coherence C(l) against spherical-harmonic degree
  bottom -- normalised per-degree error E(l) and the phase-only floor 1 - C(l)^2

The floor is the point of the whole figure: it is the smallest error reachable at
that scale by any rescaling of the prediction, so wherever it sits near 1 the
model is producing incoherent texture and no sharpness knob can rescue it.
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


def _smooth(y, ell, width=8):
    """Log-spaced running mean, so the high-degree tail is readable."""
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        lo = max(1, int(i / (1.0 + 1.0 / width)))
        hi = min(len(y), int(i * (1.0 + 1.0 / width)) + 1)
        if hi > lo:
            seg = y[lo:hi]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                out[i] = float(np.mean(seg))
    return out


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

    path = results_dir / "coherence.json"
    if not path.exists():
        LOG.warning("spectra_coherence: nothing to plot, %s missing", path)
        return output_dir
    payload = json.loads(path.read_text())
    curves = payload.get("curves", {})
    if not curves:
        return output_dir

    label = payload.get("run_label", "")
    states = [s for s in payload.get("states", []) if s in curves]
    n = len(states)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.2), squeeze=False)

    for j, state in enumerate(states):
        c = curves[state]
        ell = np.asarray(c["ell"], dtype=np.float64)
        keep = ell >= 1
        R = _smooth(np.asarray(c["amplitude_ratio"]), ell)
        C = _smooth(np.asarray(c["coherence"]), ell)

        ax = axes[0][j]
        ax.semilogx(ell[keep], R[keep], color="tab:blue", lw=1.6, label="amplitude ratio R")
        ax.semilogx(ell[keep], C[keep], color="tab:red", lw=1.6, label="coherence C")
        if "interp_coherence" in c:
            Ci = _smooth(np.asarray(c["interp_coherence"]), ell)
            ax.semilogx(ell[keep], Ci[keep], color="tab:red", lw=1.2, ls="--",
                        label="coherence C (interp baseline)")
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
        ax.set_ylim(-0.15, 1.6)
        ax.set_title(state)
        ax.set_xlabel("spherical-harmonic degree l")
        if j == 0:
            ax.set_ylabel("ratio / correlation")
            ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25, which="both")

        ax = axes[1][j]
        E = _smooth(np.asarray(c["normalised_error"]), ell)
        F = _smooth(np.asarray(c["error_floor_phase_only"]), ell)
        ax.semilogx(ell[keep], E[keep], color="k", lw=1.6, label="error E = 1 + R^2 - 2RC")
        ax.semilogx(ell[keep], F[keep], color="tab:orange", lw=1.6,
                    label="phase-only floor 1 - C^2")
        if "interp_normalised_error" in c:
            Ei = _smooth(np.asarray(c["interp_normalised_error"]), ell)
            ax.semilogx(ell[keep], Ei[keep], color="tab:green", lw=1.2, ls="--",
                        label="error of interp baseline")
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.set_ylim(0.0, 2.2)
        ax.set_xlabel("spherical-harmonic degree l")
        if j == 0:
            ax.set_ylabel("normalised error")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25, which="both")

    fig.suptitle("Per-scale amplitude vs phase  --  " + str(label), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = output_dir / "spectra_coherence.pdf"
    fig.savefig(out, dpi=150)
    fig.savefig(output_dir / "spectra_coherence.png", dpi=140)
    plt.close(fig)
    LOG.info("spectra_coherence: wrote %s", out)
    return output_dir
