"""Figures for the spectra-coherence evaluator.

Two rows per weather state:
  top    -- amplitude ratio R(l) and coherence C(l) against spherical-harmonic degree
  bottom -- normalised per-degree error E(l) and the phase-only floor 1 - C(l)^2

The floor is the point of the whole figure: it is the smallest error reachable at
that scale by any rescaling of the prediction, so wherever it sits near 1 the
model is producing incoherent texture and no sharpness knob can rescue it.

Colour encodes the QUANTITY (amplitude ratio, coherence, error, floor) with neutral colours
from ``eval.plotting.SEQUENCE``; the line style encodes the SOURCE (solid = model, dashed =
the interpolated input used as a baseline). Red and blue are deliberately not used, because
in every other figure of the framework they mean "model" and "input".
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.plotting import AXIS, SEQUENCE, eval_style, save_figure, shorten_run_label, variable_spec

LOG = logging.getLogger(__name__)

# one neutral colour per quantity; the source is told apart by the line style
Q_COLOUR = {"R": SEQUENCE[0], "C": SEQUENCE[1], "E": SEQUENCE[5], "F": SEQUENCE[4]}
MODEL_LS = "-"
INTERP_LS = (0, (5, 2))


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

    label = shorten_run_label(str(payload.get("run_label") or ""))
    label = "" if label == "predictions" else label   # the generic directory name says nothing
    states = [s for s in payload.get("states", []) if s in curves]
    steps = payload.get("steps") or []
    n = len(states)
    with eval_style():
        fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 8.8), squeeze=False, layout="constrained")
        n_samples = []
        for j, state in enumerate(states):
            c = curves[state]
            ell = np.asarray(c["ell"], dtype=np.float64)
            keep = ell >= 1
            R = _smooth(np.asarray(c["amplitude_ratio"]), ell)
            C = _smooth(np.asarray(c["coherence"]), ell)
            if c.get("n_samples") is not None:
                n_samples.append(int(c["n_samples"]))

            ax = axes[0][j]
            ax.semilogx(ell[keep], R[keep], color=Q_COLOUR["R"], ls=MODEL_LS, lw=2.0,
                        label="amplitude ratio R (model / truth)")
            ax.semilogx(ell[keep], C[keep], color=Q_COLOUR["C"], ls=MODEL_LS, lw=2.0,
                        label="coherence C (model with truth)")
            if "interp_coherence" in c:
                Ci = _smooth(np.asarray(c["interp_coherence"]), ell)
                ax.semilogx(ell[keep], Ci[keep], color=Q_COLOUR["C"], lw=1.6, ls=INTERP_LS,
                            label="coherence C (interpolated input with truth)")
            ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
            ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
            ax.set_ylim(-0.15, 1.6)
            ax.set_title(variable_spec(state).name)
            ax.set_xlabel(AXIS["wavenumber"])
            if j == 0:
                ax.set_ylabel("Amplitude ratio R or coherence C")
                ax.legend(fontsize=8, loc="lower left")
            ax.grid(alpha=0.25, which="both")

            ax = axes[1][j]
            E = _smooth(np.asarray(c["normalised_error"]), ell)
            F = _smooth(np.asarray(c["error_floor_phase_only"]), ell)
            ax.semilogx(ell[keep], E[keep], color=Q_COLOUR["E"], ls=MODEL_LS, lw=2.0,
                        label="model error E = 1 + R² − 2RC")
            ax.semilogx(ell[keep], F[keep], color=Q_COLOUR["F"], ls=MODEL_LS, lw=2.0,
                        label="phase-only floor 1 − C² (model)")
            if "interp_normalised_error" in c:
                Ei = _smooth(np.asarray(c["interp_normalised_error"]), ell)
                ax.semilogx(ell[keep], Ei[keep], color=Q_COLOUR["E"], lw=1.6, ls=INTERP_LS,
                            label="error E of the interpolated input")
            ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
            ax.set_ylim(0.0, 2.2)
            ax.set_xlabel(AXIS["wavenumber"])
            if j == 0:
                ax.set_ylabel("Error normalised by the truth power")
                ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.25, which="both")

        what = []
        if label:
            what.append(label)
        if steps:
            what.append("lead time " + ", ".join(f"{int(s)} h" for s in steps))
        if n_samples:
            what.append(f"n = {max(n_samples)} fields per variable")
        fig.suptitle("Per-scale amplitude and phase of the model against the truth"
                     + ("\n" + "; ".join(what) if what else ""), fontsize=12)
        out = output_dir / "spectra_coherence.pdf"
        save_figure(fig, out, close=True)
    LOG.info("spectra_coherence: wrote %s (and .png)", out)
    return output_dir
