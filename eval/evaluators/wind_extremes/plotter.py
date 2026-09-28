"""Figures for the wind-extreme evaluator: one PNG per geographical box.

Three panels. The left panel is the retention curve: the maximum wind that
survives a disk average, divided by the raw maximum, as a function of the
averaging radius, for the model, the truth and the interpolated driver. Curves
that fall steeply belong to maxima carried by a few points; curves that stay
high belong to maxima carried by a coherent structure. The middle panel shows
the raw maximum and the size of the connected patch above 90 percent of it. The
right panel shows how far the wind maximum sits from the truth's and from the
driver's, in kilometres. Error bars and dots are the spread over the (file,
member) samples.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .runner import SOURCES

LOG = logging.getLogger(__name__)

COLOURS = {"model": "#c0392b", "truth": "#4d4d4d", "input": "#2980b9"}
LABELS = {"model": "model", "truth": "truth", "input": "driver, interpolated"}


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    **kwargs,
) -> list[Path]:
    results_dir = Path(results_dir)
    path = results_dir / "wind_extremes.json"
    if not path.exists():
        LOG.warning("wind_extremes: nothing to plot, %s missing", path)
        return []
    payload = json.loads(path.read_text())
    out_dir = Path(output_dir) if output_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    radii = [float(r) for r in payload["config"]["radii_km"]]
    keys = [f"{r:g}" for r in radii]
    written: list[Path] = []

    for row in payload.get("aggregate", []):
        box = row["box"]
        samples = [s for s in payload["samples"] if s["box"] == box]
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

        ax = axes[0]
        for source in SOURCES:
            mean = [row[source]["retention"][k]["mean"] for k in keys]
            sd = [row[source]["retention"][k]["sd"] or 0.0 for k in keys]
            ax.errorbar(radii, mean, yerr=sd, marker="o", capsize=3,
                        color=COLOURS[source], label=LABELS[source])
        ax.set_xlabel("averaging radius (km)")
        ax.set_ylabel("maximum after averaging / raw maximum")
        ax.set_title("how much of the maximum survives averaging", fontsize=11)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        width = 0.35
        pos = np.arange(len(SOURCES))
        peaks = [row[s]["peak"]["mean"] for s in SOURCES]
        peak_sd = [row[s]["peak"]["sd"] or 0.0 for s in SOURCES]
        ax.bar(pos - width / 2, peaks, width, yerr=peak_sd, capsize=3,
               color=[COLOURS[s] for s in SOURCES], label="peak wind (m/s)")
        ax.set_xticks(pos)
        ax.set_xticklabels([LABELS[s] for s in SOURCES], fontsize=8)
        ax.set_ylabel("peak 10 m wind speed (m/s)")
        ax2 = ax.twinx()
        patch = [row[s]["patch_points_90pct"]["mean"] for s in SOURCES]
        ax2.bar(pos + width / 2, patch, width, color="none", edgecolor="black", hatch="//",
                label="patch above 90% (points)")
        ax2.set_ylabel("patch above 90% of the peak (grid points)", fontsize=9)
        ax.set_title("peak amplitude and the size of its patch", fontsize=11)

        ax = axes[2]
        pairs = ["model_vs_truth", "model_vs_input", "truth_vs_input"]
        nice = {"model_vs_truth": "model vs truth", "model_vs_input": "model vs driver",
                "truth_vs_input": "truth vs driver"}
        for i, pair in enumerate(pairs):
            vals = [s["peak_displacement_km"][pair] for s in samples
                    if s["peak_displacement_km"].get(pair) is not None]
            if vals:
                ax.scatter(np.full(len(vals), i) + np.random.uniform(-0.12, 0.12, len(vals)),
                           vals, s=14, alpha=0.55, color="#34495e")
                ax.hlines(float(np.median(vals)), i - 0.25, i + 0.25, color="#c0392b", lw=2)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([nice[p] for p in pairs], fontsize=8)
        ax.set_ylabel("distance between wind maxima (km)")
        ax.set_title("where the maximum sits", fontsize=11)
        ax.grid(alpha=0.3, axis="y")

        fig.suptitle(f"Wind extremes — {payload['run_label']} — {box} "
                     f"({row['n_samples']} file-member samples)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=3.0)
        target = out_dir / f"wind_extremes_{box}.png"
        fig.savefig(target, dpi=140)
        plt.close(fig)
        written.append(target)
        LOG.info("wind_extremes: wrote %s", target)
    return written
