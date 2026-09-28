"""Figures for the displacement evaluator: one PNG per box and field.

The left panel is a scatter of the offset that best aligns each sample, eastward
against northward, in kilometres, with the origin marked. The offset is where the
second field's feature sits relative to the first field's, so a cloud centred on
the origin means the model leaves features where the driver put them, and a cloud
away from it means it moves them, by the amount shown. The right
panel shows how much correlation the shift buys: the correlation without any
shift against the correlation at the best shift. Points close to the diagonal
mean the alignment was already as good as it gets, which is the reassuring case.

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

from eval.plotting import SEQUENCE, eval_style, save_figure, shorten_run_label, variable_spec

LOG = logging.getLogger(__name__)

# The three comparisons are pairs of fields, not roles, so they take neutral colours from
# SEQUENCE plus a distinct marker each (red and blue stay reserved for model and input).
PAIR_STYLE = {
    "model_vs_input": (SEQUENCE[0], "o", "model relative to input"),
    "model_vs_truth": (SEQUENCE[1], "s", "model relative to truth"),
    "truth_vs_input": (SEQUENCE[2], "^", "truth relative to input"),
}


def _run_text(label) -> str:
    """Readable run name; the generic directory name "predictions" says nothing and is dropped."""
    text = shorten_run_label(str(label or ""))
    return "" if text in ("", "predictions") else text


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
    path = results_dir / "displacement.json"
    if not path.exists():
        LOG.warning("displacement: nothing to plot, %s missing", path)
        return []
    payload = json.loads(path.read_text())
    out_dir = Path(output_dir) if output_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes = (payload.get("config") or {}).get("boxes", {})
    run = _run_text(payload.get("run_label"))
    written: list[Path] = []

    for row in payload.get("aggregate", []):
        box, field = row["box"], row["field"]
        sel = [s for s in payload["samples"] if s["box"] == box and s["field"] == field]
        if not sel:
            continue
        field_name = variable_spec(field).name
        with eval_style():
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), layout="constrained")

            ax = axes[0]
            limit = 1.0
            for pair, (colour, marker, label) in PAIR_STYLE.items():
                east = [s["shift"][pair]["east_km"] for s in sel if pair in s["shift"]]
                north = [s["shift"][pair]["north_km"] for s in sel if pair in s["shift"]]
                if not east:
                    continue
                ax.scatter(east, north, s=18, alpha=0.5, color=colour, marker=marker,
                           linewidths=0, label=f"{label} (n = {len(east)})")
                ax.scatter([np.median(east)], [np.median(north)], s=170, marker="+",
                           color=colour, linewidths=2.8, zorder=5)
                limit = max(limit, np.percentile(np.abs(east + north), 98))
            ax.axhline(0.0, color="0.2", lw=0.8)
            ax.axvline(0.0, color="0.2", lw=0.8)
            ax.set_xlim(-limit * 1.15, limit * 1.15)
            ax.set_ylim(-limit * 1.15, limit * 1.15)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Eastward offset of the second field's feature (km)")
            ax.set_ylabel("Northward offset of the second field's feature (km)")
            ax.set_title("Offset of the best alignment (+ = median)")
            ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")

            ax = axes[1]
            for pair, (colour, marker, label) in PAIR_STYLE.items():
                zero = [s["shift"][pair]["corr_zero"] for s in sel if pair in s["shift"]]
                best = [s["shift"][pair]["corr_best"] for s in sel if pair in s["shift"]]
                if not zero:
                    continue
                ax.scatter(zero, best, s=18, alpha=0.5, color=colour, marker=marker,
                           linewidths=0, label=f"{label} (n = {len(zero)})")
            lo = 0.0
            ax.plot([lo, 1.0], [lo, 1.0], color="0.2", lw=0.9, ls="--", label="no gain (diagonal)")
            ax.set_xlabel("Correlation without a shift")
            ax.set_ylabel("Correlation at the best shift")
            ax.set_title("What the shift buys")
            ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")

            fig.suptitle(f"Feature displacement, {field_name}{', ' + run if run else ''}: {_box_text(box, boxes.get(box))}\n"
                         f"{row['n_samples']} samples (file x member)", fontsize=12)
            target = out_dir / f"displacement_{box}_{field}.png"
            save_figure(fig, target, close=True)
        written.append(target)
        LOG.info("displacement: wrote %s (and .pdf)", target)
    return written
