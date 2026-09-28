"""sigma_loss evaluator — plotter (View A).

View A: per-sigma F-space loss curve (total + per-variable) on log-x, log-y.
Writes under <results_dir>/plots/sigma_loss/ (PNG at 150 dpi plus a PDF of the same name),
in the house style of ``eval.plotting``.

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

from eval.plotting import AXIS, eval_style, save_figure, sequence_style, variable_spec  # noqa: E402

LOG = logging.getLogger(__name__)

DATA_SUBDIR = ("data", "sigma_loss")
PLOTS_SUBDIR = ("plots", "sigma_loss")
# The total is an aggregate, not a role: a near-black grey keeps it apart from the truth black.
TOTAL_COLOR = "#333333"
_VAR_ORDER = ("10u", "10v", "10ff", "2t", "2d", "skt", "sp", "msl", "tcw", "tp", "cp")


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


def _var_sort_key(var: str) -> tuple:
    """Surface variables in a fixed order first, then pressure-level fields by name and level."""
    key = variable_spec(var).key
    if key in _VAR_ORDER:
        return (0, _VAR_ORDER.index(key), 0)
    fam, _, lev = key.partition("_")
    return (1, fam, int(lev) if lev.isdigit() else 0)


def _total_batches(csv_path: Path) -> int | None:
    """Number of validation batches behind the total curve (largest n_batches), if recorded."""
    best = None
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("variable") != "__total__":
                continue
            try:
                n = int(float(row.get("n_batches") or ""))
            except ValueError:
                continue
            best = n if best is None else max(best, n)
    return best


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
    n_batches = _total_batches(csv_path)
    view_a = plots_dir / "view_a_per_sigma_loss.png"
    with eval_style():
        fig, ax = plt.subplots(figsize=(10, 6))

        # per-variable curves (thin, background); colour and dash from the neutral sequence,
        # because the variables are quantities, not roles
        for i, (var, pts) in enumerate(sorted(by_var.items(), key=lambda kv: _var_sort_key(kv[0]))):
            if var == "__total__":
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, label=variable_spec(var).name,
                    **sequence_style(i, linewidth=1.1, alpha=0.8, zorder=2))

        # total curve (bold, foreground)
        tot = by_var["__total__"]
        total_label = "Total over all variables"
        if n_batches:
            total_label += f" (n = {n_batches} validation batches)"
        ax.plot([p[0] for p in tot], [p[1] for p in tot], color=TOTAL_COLOR, linewidth=3.0,
                marker="o", markersize=4.5, zorder=5, label=total_label)

        ax.axvline(sigma_data, color="0.45", ls=(0, (5, 2)), lw=1.3, zorder=1,
                   label=f"σ_data = {sigma_data:g}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(AXIS["sigma"])
        ax.set_ylabel("Loss in F-space (weighted MSE, weight 1 / c_out²)")
        title = "Training loss per noise level"
        if run_id:
            title += f": run {run_id[:8]}, step {ckpt_step}"
        ax.set_title(title)
        ax.grid(True, which="minor", color="0.93", linewidth=0.4)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
        fig.tight_layout()

        # PNG (the deliverable) plus a PDF of the same name
        save_figure(fig, view_a, close=True)
    outputs.append(view_a)
    LOG.info("sigma_loss plotter: wrote %s", view_a)

    # ---- View B STUB (M0+): sigma x variable heatmap ----
    # TODO: render a log-loss heatmap (rows=variables, cols=sigma) to expose
    # which variables drive the extreme-band trade-off. Not implemented in M0.

    return outputs
