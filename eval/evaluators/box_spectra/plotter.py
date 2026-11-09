"""The wavelength-ladder figure of standing rule 14, from bins.csv (pooled rows).

One panel per variable, large scales on the left: predicted residual over true residual, full
field over truth, interpolated input over truth, and the aligned part (coherence x full-field
ratio), with the six marker lines of the o320->o1280 ladder. Amplitude, never power.
"""
from __future__ import annotations

import math
from pathlib import Path

from eval.plotting.roles import INPUT_COLOR, MODEL_COLOR, TRUTH_COLOR
from eval.plotting.style import eval_style, save_figure

from .runner import read_bins

A_KM = 6371.229
CIRC = 2 * math.pi * A_KM
AREA = 4 * math.pi * A_KM ** 2
_SPACING = lambda n: math.sqrt(AREA / n)  # noqa: E731
# the o320->o1280 ladder (plot_wavelength_ladder.py of 2026-10-08)
LADDER = [
    (CIRC / 319, "input ends (T319)"),
    (4 * _SPACING(655_362), "4 hidden spacings"),
    (2 * _SPACING(542_080), "o320 Nyquist"),
    (2 * _SPACING(655_362), "2 hidden spacings"),
    (CIRC / 1279, "truth ends (T1279)"),
    (2 * _SPACING(6_599_680), "o1280 Nyquist"),
]
CURVES = [("ratio_resid", MODEL_COLOR, "-", "predicted residual / true residual"),
          ("ratio_full", MODEL_COLOR, "--", "prediction / truth (full field)"),
          ("ratio_interp_to_truth", INPUT_COLOR, "-", "interpolated input / truth"),
          ("aligned", TRUTH_COLOR, ":", "aligned part (coherence x full-field ratio)")]


def grid_panels(rows: list[dict]) -> dict:
    """Curves of the standard spectra figure from the pooled bins.csv rows.

    Amplitude per bin (display unit, as stored: sqrt of the power averaged over draws), placed at
    the equivalent total wavenumber
    40030 km / bin centre so that the figure reads like the spherical-harmonic one. Truth, the
    interpolated input (the input ensemble when the predictions hold one draw per input member)
    and the model.
    """
    import numpy as np

    from eval.plotting.spec_helpers import amplitude_in_display
    from eval.plotting.spectra_grid import Curve, wavelength_to_wavenumber

    panels = {}
    for var in dict.fromkeys(r["variable"] for r in rows):
        sub = sorted((r for r in rows if r["variable"] == var), key=lambda r: -float(r["bin_centre_km"]))
        wvn = wavelength_to_wavenumber([float(r["bin_centre_km"]) for r in sub])

        def amp(col):
            vals = np.array([float(r[col]) if r[col] not in ("", "nan") else np.nan for r in sub])
            return amplitude_in_display(var, vals)

        panels[var] = [Curve("Truth", wvn, amp("amp_truth_full"), role="truth"),
                       Curve("Input (interpolated)", wvn, amp("amp_interp"), role="input"),
                       Curve(f"Model ({sub[0]['n_draws']} draws)", wvn, amp("amp_pred_full"), role="model")]
    return panels


def plot_grid(results_dir, *, output_dir=None, title=None):
    """The standard spectra figure (eval.plotting.spectra_grid) from bins.csv: box_spectra_grid.png/.pdf."""
    from eval.plotting.spectra_grid import plot_spectra_grid

    path = Path(results_dir) / "bins.csv"
    if not path.exists():
        return []
    rows = [r for r in read_bins(path) if r["scope"] == "all"]
    if not rows:
        return []
    foot = ("Box FFT amplitude per wavelength bin (display unit), placed at total wavenumber "
            "40030 km / wavelength; ratio = amplitude / truth amplitude (rule 14 full-field ratio).")
    return plot_spectra_grid(grid_panels(rows), Path(output_dir or results_dir) / "box_spectra_grid",
                             title=title, footnote=foot)


def plot(results_dir, lane_config: dict, eval_config: dict, *, output_dir=None, **kwargs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    path = results_dir / "bins.csv"
    if not path.exists():
        return None
    rows = [r for r in read_bins(path) if r["scope"] == "all"]
    variables = sorted({r["variable"] for r in rows}, key=lambda v: ["10u", "10v", "ws", "2t", "msl"].index(v)
                       if v in ("10u", "10v", "ws", "2t", "msl") else 99)
    if not variables:
        return None
    with eval_style():
        fig, axes = plt.subplots(len(variables), 1, figsize=(10, 2.9 * len(variables)), sharex=True,
                                 squeeze=False)
        for ax, var in zip(axes[:, 0], variables):
            sub = sorted((r for r in rows if r["variable"] == var), key=lambda r: -float(r["bin_centre_km"]))
            x = [float(r["bin_centre_km"]) for r in sub]
            for col, color, ls, label in CURVES:
                y = [min(float(r[col]), 1.6) if r[col] not in ("", "nan") else float("nan") for r in sub]
                ax.plot(x, y, color=color, ls=ls, lw=1.8, marker="o", ms=3, label=label)
            for lam, _ in LADDER:
                ax.axvline(lam, color="0.55", lw=0.7, ls=(0, (4, 3)))
            ax.axhline(1.0, color="0.75", lw=1.0)
            ax.set_xscale("log")
            ax.set_xlim(max(x + [600.0]), 15)
            ax.set_ylim(0, 1.6)
            ax.set_ylabel(f"{var}\namplitude ratio per bin")
            ax.text(0.01, 0.95, f"draws: {sub[0]['n_draws']}", transform=ax.transAxes, va="top", fontsize=8)
        top = axes[0, 0]
        for i, (lam, lab) in enumerate(LADDER):
            top.text(lam, 1.62 + 0.12 * (i % 2), f"{lab}\n{lam:.0f} km", ha="center", va="bottom", fontsize=7)
        top.legend(loc="lower left", fontsize=8, frameon=False)
        ticks = [500, 300, 200, 125, 80, 56, 40, 31, 25, 18]
        axes[-1, 0].set_xticks(ticks)
        axes[-1, 0].set_xticklabels([str(t) for t in ticks])
        axes[-1, 0].minorticks_off()
        axes[-1, 0].set_xlabel("wavelength [km], large scales on the left")
        written = save_figure(fig, Path(output_dir or results_dir) / "box_spectra_ladder", close=True)
    written += plot_grid(results_dir, output_dir=output_dir, title=(eval_config or {}).get("title"))
    return written
