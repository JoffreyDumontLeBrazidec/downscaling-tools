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
    return written
