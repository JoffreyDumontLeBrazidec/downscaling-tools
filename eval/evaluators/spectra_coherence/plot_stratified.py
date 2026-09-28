"""Figure for the surface-stratified amplitude/phase result.

One column per weather state. Top row is phase agreement by surface class, with
the classes ordered by how rough the terrain is; bottom row is the amplitude
ratio on the same axes. The point of the pairing is that the bottom row is flat
at one everywhere while the top row is not: the model puts the same amount of
fine-scale energy into every surface type and only gets it in the right PLACE
where the orography it is given tells it where to put it.
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

CLASS_ORDER = ["ocean", "flat_land", "coast", "rolling_land", "complex_land"]
CLASS_LABEL = {
    "ocean": "ocean",
    "flat_land": "flat land",
    "coast": "coast",
    "rolling_land": "rolling land",
    "complex_land": "complex terrain",
}
BAND_STYLE = {
    "synoptic": ("tab:blue", "-"),
    "meso": ("tab:green", "-"),
    "fine": ("tab:red", "-"),
    "very_fine": ("tab:purple", "-"),
    "near_grid": ("tab:brown", "-"),
}


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

    fig, axes = plt.subplots(2, len(states), figsize=(4.4 * len(states), 8.0), squeeze=False)
    for j, st in enumerate(states):
        for bd in bands:
            col, ls = BAND_STYLE.get(bd, ("k", "-"))
            c_model = [idx[(st, bd, c)]["correlation"] if (st, bd, c) in idx else np.nan for c in classes]
            c_interp = [idx[(st, bd, c)].get("interp_correlation", np.nan) if (st, bd, c) in idx else np.nan for c in classes]
            r_model = [idx[(st, bd, c)]["amplitude_ratio"] if (st, bd, c) in idx else np.nan for c in classes]
            axes[0][j].plot(xs, c_model, color=col, ls=ls, marker="o", ms=4, lw=1.7, label=bd)
            axes[0][j].plot(xs, c_interp, color=col, ls=":", marker="x", ms=4, lw=1.1, alpha=0.75)
            axes[1][j].plot(xs, r_model, color=col, ls=ls, marker="o", ms=4, lw=1.7, label=bd)

        for row, ylab, lo, hi in ((0, "phase agreement C", -0.05, 1.05), (1, "amplitude ratio R", 0.5, 1.5)):
            ax = axes[row][j]
            ax.set_xticks(xs)
            ax.set_xticklabels(ticks, fontsize=7)
            ax.set_ylim(lo, hi)
            ax.grid(alpha=0.25)
            if row == 1:
                ax.axhline(1.0, color="0.4", lw=0.9, ls=":")
            if j == 0:
                ax.set_ylabel(ylab)
        axes[0][j].set_title(st)
        if j == 0:
            axes[0][j].legend(fontsize=7, loc="upper left", title="band (dotted = interp)", title_fontsize=7)

    fig.suptitle(
        "Phase agreement by surface type  --  %s\nx-axis ordered by median orographic standard deviation"
        % d.get("run_label", ""), fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = output_dir / "coherence_by_surface.pdf"
    fig.savefig(out, dpi=150)
    fig.savefig(output_dir / "coherence_by_surface.png", dpi=140)
    plt.close(fig)
    LOG.info("stratified plot: wrote %s", out)
    return output_dir
