"""Figures for the displacement evaluator: one PNG per box and field.

The left panel is a scatter of the offset that best aligns each sample, eastward
against northward, in kilometres, with the origin marked. The offset is where the
second field's feature sits relative to the first field's, so a cloud centred on
the origin means the model leaves features where the driver put them, and a cloud
away from it means it moves them, by the amount shown. The right
panel shows how much correlation the shift buys: the correlation without any
shift against the correlation at the best shift. Points close to the diagonal
mean the alignment was already as good as it gets, which is the reassuring case.
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

PAIR_STYLE = {
    "model_vs_input": ("#c0392b", "model against driver"),
    "model_vs_truth": ("#7f8c8d", "model against truth"),
    "truth_vs_input": ("#2980b9", "truth against driver"),
}


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
    written: list[Path] = []

    for row in payload.get("aggregate", []):
        box, field = row["box"], row["field"]
        sel = [s for s in payload["samples"] if s["box"] == box and s["field"] == field]
        if not sel:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

        ax = axes[0]
        limit = 1.0
        for pair, (colour, label) in PAIR_STYLE.items():
            east = [s["shift"][pair]["east_km"] for s in sel if pair in s["shift"]]
            north = [s["shift"][pair]["north_km"] for s in sel if pair in s["shift"]]
            if not east:
                continue
            ax.scatter(east, north, s=18, alpha=0.5, color=colour, label=label)
            ax.scatter([np.median(east)], [np.median(north)], s=150, marker="+",
                       color=colour, linewidths=2.5)
            limit = max(limit, np.percentile(np.abs(east + north), 98))
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_xlim(-limit * 1.15, limit * 1.15)
        ax.set_ylim(-limit * 1.15, limit * 1.15)
        ax.set_xlabel("how far east the second field's feature sits (km)")
        ax.set_ylabel("how far north the second field's feature sits (km)")
        ax.set_title(f"{box} · {field}: where the best alignment sits")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1]
        for pair, (colour, label) in PAIR_STYLE.items():
            zero = [s["shift"][pair]["corr_zero"] for s in sel if pair in s["shift"]]
            best = [s["shift"][pair]["corr_best"] for s in sel if pair in s["shift"]]
            if not zero:
                continue
            ax.scatter(zero, best, s=18, alpha=0.5, color=colour, label=label)
        lo = 0.0
        ax.plot([lo, 1.0], [lo, 1.0], color="black", lw=0.8, ls="--")
        ax.set_xlabel("correlation with no shift")
        ax.set_ylabel("correlation at the best shift")
        ax.set_title(f"{box} · {field}: what the shift buys")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        fig.suptitle(f"Feature displacement — {payload['run_label']} — {box} · {field} "
                     f"({row['n_samples']} file-member samples)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        target = out_dir / f"displacement_{box}_{field}.png"
        fig.savefig(target, dpi=140)
        plt.close(fig)
        written.append(target)
        LOG.info("displacement: wrote %s", target)
    return written
