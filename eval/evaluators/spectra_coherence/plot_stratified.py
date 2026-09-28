"""Figure for the surface-stratified amplitude/phase result.

One column per weather state. Top row is phase agreement by surface class, with
the classes ordered by how rough the terrain is; bottom row is the amplitude
ratio on the same axes. The point of the pairing is that the bottom row is flat
at one everywhere while the top row is not: the model puts the same amount of
fine-scale energy into every surface type and only gets it in the right PLACE
where the orography it is given tells it where to put it.

Colour encodes the wavenumber BAND (neutral colours from ``eval.plotting.SEQUENCE``); the
line style encodes the source (solid with dots = model, dotted with crosses = the
interpolated input). Red and blue keep their framework meaning and are not used for bands.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.plotting import SEQUENCE, eval_style, save_figure, shorten_run_label, variable_spec

LOG = logging.getLogger(__name__)

CLASS_ORDER = ["ocean", "flat_land", "coast", "rolling_land", "complex_land"]
CLASS_LABEL = {
    "ocean": "ocean",
    "flat_land": "flat land",
    "coast": "coast",
    "rolling_land": "rolling land",
    "complex_land": "complex terrain",
}
BAND_STYLE = {
    "synoptic": (SEQUENCE[0], "-"),
    "meso": (SEQUENCE[1], "-"),
    "fine": (SEQUENCE[2], "-"),
    "very_fine": (SEQUENCE[3], "-"),
    "near_grid": (SEQUENCE[4], "-"),
}


def _band_label(name: str, bands) -> str:
    """"fine (ℓ 300–500)" from the band table stored in the payload."""
    for b in bands or []:
        if isinstance(b, dict) and b.get("name") == name:
            lo, hi = b.get("lo"), b.get("hi")
            if lo is not None and hi is not None:
                rng = f"ℓ ≥ {lo:g}" if hi >= 10000 else f"ℓ {lo:g}–{hi:g}"
                return f"{name.replace('_', ' ')} ({rng})"
    return name.replace("_", " ")


def plot_stratified(results_dir, *, output_dir=None):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    path = results_dir / "coherence_by_surface.json"
    if not path.exists():
        LOG.warning("stratified plot: %s missing", path)
        return output_dir
    d = json.loads(path.read_text())

    rows = d["rows"]
    states, bands = [], []
    for r in rows:
        if r["state"] not in states:
            states.append(r["state"])
        if r["band"] not in bands:
            bands.append(r["band"])
    idx = {(r["state"], r["band"], r["surface_class"]): r for r in rows}
    classes = [c for c in CLASS_ORDER if any(k[2] == c for k in idx)]
    xs = np.arange(len(classes))
    med = d.get("surface_classes", {})
    ticks = [
        "%s\n%s m" % (
            CLASS_LABEL.get(c, c),
            ("%.0f" % med[c]["median_orog_std_m"]) if med.get(c, {}).get("median_orog_std_m") is not None else "?",
        )
        for c in classes
    ]

    with eval_style():
        fig, axes = plt.subplots(2, len(states), figsize=(4.6 * len(states), 8.6), squeeze=False,
                                 layout="constrained")
        for j, st in enumerate(states):
            for k, bd in enumerate(bands):
                col, ls = BAND_STYLE.get(bd, (SEQUENCE[k % len(SEQUENCE)], "-"))
                c_model = [idx[(st, bd, c)]["correlation"] if (st, bd, c) in idx else np.nan for c in classes]
                c_interp = [idx[(st, bd, c)].get("interp_correlation", np.nan) if (st, bd, c) in idx else np.nan for c in classes]
                r_model = [idx[(st, bd, c)]["amplitude_ratio"] if (st, bd, c) in idx else np.nan for c in classes]
                axes[0][j].plot(xs, c_model, color=col, ls=ls, marker="o", ms=5, lw=2.0,
                                label=_band_label(bd, d.get("bands")))
                axes[0][j].plot(xs, c_interp, color=col, ls=":", marker="x", ms=5, lw=1.4, alpha=0.85)
                axes[1][j].plot(xs, r_model, color=col, ls=ls, marker="o", ms=5, lw=2.0,
                                label=_band_label(bd, d.get("bands")))

            for row, ylab, lo, hi in ((0, "Phase agreement C (correlation with truth)", -0.05, 1.05),
                                      (1, "Amplitude ratio R (model / truth)", 0.5, 1.5)):
                ax = axes[row][j]
                ax.set_xticks(xs)
                ax.set_xticklabels(ticks, fontsize=7.5)
                ax.set_ylim(lo, hi)
                ax.grid(alpha=0.25)
                if row == 1:
                    ax.axhline(1.0, color="0.4", lw=0.9, ls=":")
                if j == 0:
                    ax.set_ylabel(ylab)
            axes[0][j].set_title(variable_spec(st).name)
            if j == 0:
                # the amplitude row is flat near 1, so its upper half is free for the legend
                handles, labels = axes[0][j].get_legend_handles_labels()
                axes[1][j].legend(handles, labels, fontsize=7.5, loc="upper left", ncol=2,
                                  title="wavenumber band (top row: solid = model,\n"
                                        "dotted = interpolated input)",
                                  title_fontsize=7.5)

        raw = str(d.get("run_label") or "")
        run = raw if len(raw) <= 60 and not raw.startswith(("manual_", "anemoi_")) else shorten_run_label(raw)
        n_files = d.get("n_member_files")
        fig.suptitle(
            "Phase agreement and amplitude by surface type" + (f", {run}" if run and run != "predictions" else "")
            + (f" (n = {n_files} member files)" if n_files else "")
            + "\nsurface classes ordered by the median standard deviation of the orography (m, under each name)",
            fontsize=12,
        )
        out = output_dir / "coherence_by_surface.pdf"
        save_figure(fig, out, close=True)
    LOG.info("stratified plot: wrote %s (and .png)", out)
    return output_dir
