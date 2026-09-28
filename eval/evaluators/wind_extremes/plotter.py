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

Each figure is written as PNG (150 dpi) and PDF in the house style of ``eval.plotting``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.plotting import eval_style, role_style, save_figure, shorten_run_label, variable_spec

from .runner import SOURCES

LOG = logging.getLogger(__name__)

ROLE = {"model": "model", "truth": "truth", "input": "input"}
LABELS = {"model": "model", "truth": "truth", "input": "input (interpolated to the target grid)"}
SHORT = {"model": "model", "truth": "truth", "input": "input,\ninterpolated"}
COLOURS = {s: role_style(r)["color"] for s, r in ROLE.items()}
POINTS_COLOUR = "0.45"   # sample dots in the displacement panel: neutral, not a role colour
MEDIAN_COLOUR = "0.05"


def _box_text(box: str, bounds=None) -> str:
    name = box.replace("_", " ")
    if bounds and len(bounds) == 4:
        s, n, w, e = bounds
        return f"{name} ({s:g} to {n:g}°N, {w:g} to {e:g}°E)"
    return name


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
    boxes = payload["config"].get("boxes", {})
    unit = variable_spec("10ff").unit
    run = shorten_run_label(str(payload.get("run_label") or ""))
    run = "" if run == "predictions" else run
    rng = np.random.default_rng(0)   # jitter of the dots only; fixed so reruns look identical
    written: list[Path] = []

    for row in payload.get("aggregate", []):
        box = row["box"]
        samples = [s for s in payload["samples"] if s["box"] == box]
        n = row["n_samples"]
        with eval_style():
            fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.0), layout="constrained")

            ax = axes[0]
            for source in SOURCES:
                mean = [row[source]["retention"][k]["mean"] for k in keys]
                sd = [row[source]["retention"][k]["sd"] or 0.0 for k in keys]
                st = role_style(ROLE[source])
                ax.errorbar(radii, mean, yerr=sd, marker="o", capsize=3,
                            label=f"{LABELS[source]} (n = {n})", **st)
            ax.set_xlabel("Averaging radius (km)")
            ax.set_ylabel("Peak after disk averaging / raw peak")
            ax.set_title("How much of the peak survives averaging")
            ax.set_ylim(0.0, 1.02)
            ax.legend(loc="lower left")

            ax = axes[1]
            width = 0.35
            pos = np.arange(len(SOURCES))
            peaks = [row[s]["peak"]["mean"] for s in SOURCES]
            peak_sd = [row[s]["peak"]["sd"] or 0.0 for s in SOURCES]
            ax.bar(pos - width / 2, peaks, width, yerr=peak_sd, capsize=3,
                   color=[COLOURS[s] for s in SOURCES], label=f"peak 10 m wind speed ({unit})")
            ax.set_xticks(pos)
            ax.set_xticklabels([SHORT[s] for s in SOURCES])
            ax.set_ylabel(f"Peak 10 m wind speed ({unit})")
            ax2 = ax.twinx()
            ax2.spines["right"].set_visible(True)
            ax2.grid(False)
            patch = [row[s]["patch_points_90pct"]["mean"] for s in SOURCES]
            ax2.bar(pos + width / 2, patch, width, color="none", edgecolor="0.2", hatch="//",
                    label="connected patch above 90 % of the peak (grid points)")
            ax2.set_ylabel("Patch above 90 % of the peak (grid points)")
            ax.set_title("Peak and the size of its patch")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, ["filled: " + l1[0], "hatched: " + l2[0]], loc="upper center",
                      bbox_to_anchor=(0.5, -0.14), fontsize=8)

            ax = axes[2]
            pairs = ["model_vs_truth", "model_vs_input", "truth_vs_input"]
            nice = {"model_vs_truth": "model vs\ntruth", "model_vs_input": "model vs\ninput",
                    "truth_vs_input": "truth vs\ninput"}
            counts = []
            for i, pair in enumerate(pairs):
                vals = [s["peak_displacement_km"][pair] for s in samples
                        if s["peak_displacement_km"].get(pair) is not None]
                counts.append(len(vals))
                if vals:
                    ax.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)),
                               vals, s=14, alpha=0.5, color=POINTS_COLOUR, linewidths=0,
                               label="one sample" if i == 0 else None)
                    ax.hlines(float(np.median(vals)), i - 0.25, i + 0.25, color=MEDIAN_COLOUR,
                              lw=2.6, label="median" if i == 0 else None)
            ax.set_xticks(range(len(pairs)))
            ax.set_xticklabels([f"{nice[p]}\n(n = {c})" for p, c in zip(pairs, counts)])
            ax.set_ylabel("Distance between the wind peaks (km)")
            ax.set_title("Where the peak sits")
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")

            fig.suptitle(f"10 m wind extremes{', ' + run if run else ''}: {_box_text(box, boxes.get(box))}\n"
                         f"{n} samples (file x member); error bars: one standard deviation over samples",
                         fontsize=12)
            target = out_dir / f"wind_extremes_{box}.png"
            save_figure(fig, target, close=True)
        written.append(target)
        LOG.info("wind_extremes: wrote %s (and .pdf)", target)
    return written
