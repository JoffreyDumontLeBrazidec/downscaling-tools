"""sigma_loss evaluator — plotter (View A).

View A: per-sigma F-space loss curve (total + per-variable) on log-x, log-y.
Writes under <results_dir>/plots/sigma_loss/.

M0+ STUB: View B (sigma x variable heatmap) — see TODO at the bottom.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = logging.getLogger(__name__)

DATA_SUBDIR = ("data", "sigma_loss")
PLOTS_SUBDIR = ("plots", "sigma_loss")


def _read_rows(csv_path: Path) -> dict[str, list[tuple[float, float]]]:
    by_var: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                by_var[row["variable"]].append((float(row["sigma"]), float(row["fspace_loss"])))
            except (TypeError, ValueError):
                continue
    for v in by_var.values():
        v.sort(key=lambda p: p[0])
    return by_var


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> list[Path]:
    results_dir = Path(results_dir)
    out_base = Path(output_dir) if output_dir else results_dir
    data_dir = results_dir.joinpath(*DATA_SUBDIR)
    plots_dir = out_base.joinpath(*PLOTS_SUBDIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "per_sigma.csv"
    if not csv_path.exists():
        LOG.warning("sigma_loss plotter: no per_sigma.csv at %s", csv_path)
        return []

    meta = {}
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
    sigma_data = float(meta.get("sigma_data", 1.0))
    run_id = meta.get("run_id", "")
    ckpt_step = meta.get("ckpt_step", "")

    by_var = _read_rows(csv_path)
    if "__total__" not in by_var:
        LOG.warning("sigma_loss plotter: no __total__ curve")
        return []

    outputs: list[Path] = []
    fig, ax = plt.subplots(figsize=(9, 6))

    # per-variable curves (thin, background)
    for var, pts in sorted(by_var.items()):
        if var == "__total__":
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, lw=0.8, alpha=0.45)

    # total curve (bold, foreground)
    tot = by_var["__total__"]
    ax.plot([p[0] for p in tot], [p[1] for p in tot],
            lw=2.6, color="black", marker="o", ms=4, label="total (F-space)")

    ax.axvline(sigma_data, color="crimson", ls="--", lw=1.2,
               label=f"sigma_data={sigma_data:g}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma")
    ax.set_ylabel("F-space loss  (weighted MSE, weights=1/c_out^2)")
    title = "Per-sigma F-space loss"
    if run_id:
        title += f"  [{run_id[:8]} step {ckpt_step}]"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    view_a = plots_dir / "view_a_per_sigma_loss.png"
    fig.savefig(view_a, dpi=130)
    plt.close(fig)
    outputs.append(view_a)
    LOG.info("sigma_loss plotter: wrote %s", view_a)

    # ---- View B STUB (M0+): sigma x variable heatmap ----
    # TODO: render a log-loss heatmap (rows=variables, cols=sigma) to expose
    # which variables drive the extreme-band trade-off. Not implemented in M0.

    return outputs
