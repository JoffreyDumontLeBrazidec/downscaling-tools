"""
plot.py — flexible loss curve plots from loader.py RunData dicts.

TWO MAIN PLOTS
--------------
  plot_key_vars(runs)   6-panel grid: one subplot per key surface variable
                        shows val MSE per var, all runs overlaid
                        + aggregate train/val loss as a summary panel on top

  plot_overview(runs)   3-panel: aggregate train+val, LR schedule, val-all MSE

HOUSE STYLE
-----------
  Figures follow ``eval.plotting``: one colour per run from the colour-blind-safe sequence
  (``sequence_style``); the validation loss is a solid line; the training loss is drawn twice
  in the run's colour, raw as a faint thin line and smoothed (exponential moving average) as a
  bold dashed line; x axis "Training step"; panel titles use the variable names of the
  variable table. Every figure is written as PNG (150 dpi) and as a PDF with the same stem.

  Metric names changed between training code versions (``val_mse_metric/sfc_10u/1`` and
  ``train_weighted_mse_loss_epoch`` in older runs, ``val_out_hres_mse_metric/out_hres/
  sfc_10u_scale_0`` and ``train_multi_dataset_loss_epoch`` in newer ones); each panel draws
  whichever spelling a run carries.

CUSTOMIZE AT THE TOP
--------------------
  KEY_VARS   — which 6 vars get the big grid (change freely)
  VAR_LABEL  — human-readable axis labels for each var
  TRAIN_KEY / VAL_KEY — which aggregate metrics to use for the overview

STANDALONE USAGE
----------------
  python plot.py <experiment_dir> [min_steps] [name_filter]

  e.g.  python plot.py ~/scratch/aifs/logs/mlflow/909682684414341917 50000 o320
"""

from pathlib import Path
import math
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):  # run as a script: make `eval.plotting` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ─── user-configurable constants ──────────────────────────────────────────────

KEY_VARS = [
    "val_mse_metric/sfc_10u/1",
    "val_mse_metric/sfc_10v/1",
    "val_mse_metric/sfc_2t/1",
    "val_mse_metric/sfc_tp/1",     # may be absent in some runs → skipped silently
    "val_mse_metric/sfc_msl/1",
    "val_mse_metric/z_500/1",
]

VAR_LABEL = {
    "val_mse_metric/sfc_10u/1":  "10 m zonal wind",
    "val_mse_metric/sfc_10v/1":  "10 m meridional wind",
    "val_mse_metric/sfc_2t/1":   "2 m temperature",
    "val_mse_metric/sfc_tp/1":   "Total precipitation",
    "val_mse_metric/sfc_msl/1":  "Mean sea level pressure",
    "val_mse_metric/z_500/1":    "500 hPa geopotential height",
    # downscaling output metrics — shown in overview if present
    "val_out_hres_mse_metric/sfc_10u/1_scale_0": "10 m zonal wind, high-resolution output",
    "val_out_hres_mse_metric/z_500/1_scale_0":   "500 hPa geopotential height, high-resolution output",
}

TRAIN_KEY = "train_weighted_mse_loss_epoch"
VAL_KEY   = "val_weighted_mse_loss_epoch"
LR_KEY    = "lr-AdamW"
VAL_ALL   = "val_mse_metric/all/1"

# Newer anemoi versions log the same quantities under these names.
_ALIASES = {
    TRAIN_KEY: ("train_multi_dataset_loss_epoch",),
    VAL_KEY: ("val_multi_dataset_loss_epoch",),
    VAL_ALL: ("val_out_hres_mse_metric/out_hres/all_scale_0",),
    "val_out_hres_mse_metric/all/1_scale_0": ("val_out_hres_mse_metric/out_hres/all_scale_0",),
}
_OLD_VAR_KEY = re.compile(r"^val_mse_metric/(?P<var>[^/]+)/1$")
_NEW_VAR_KEY = re.compile(r"^val_out_hres_mse_metric/(?:out_hres/)?(?P<var>.+?)(?:/1)?_scale_0$")

# Set True to use log scale on all per-variable val MSE panels.
# Useful when runs have very different absolute scales (e.g. mixed resolutions).
LOG_SCALE_VARS = False

# Smoothing span (in logged points) of the bold training-loss curve; the raw curve stays faint.
SMOOTH_FRACTION = 0.05
RAW_ALPHA = 0.25

MSE_LABEL = "Validation MSE (as logged)"
LOSS_LABEL = "Loss"
LR_LABEL = "Learning rate"

_FAMILY_NAMES = {
    "q": "Specific humidity", "t": "Temperature", "u": "Zonal wind", "v": "Meridional wind",
    "w": "Vertical velocity", "z": "Geopotential height", "r": "Relative humidity",
}


# ─── colours and names ────────────────────────────────────────────────────────

def _palette(n):
    """n distinct colours from the house colour-blind-safe sequence."""
    from eval.plotting import sequence_colors

    return sequence_colors(n)


def _color_map(runs):
    """Return {run_name: style dict} , consistent across all plots."""
    from eval.plotting import sequence_style

    names = sorted(runs.keys())
    return {name: sequence_style(i) for i, name in enumerate(names)}


def _run_label(name, data):
    from eval.plotting import shorten_run_label

    step = data.get("max_step")
    label = shorten_run_label(str(name))
    return f"{label} (to step {int(step):,})" if step else label


def _var_of_key(metric_key):
    """'sfc_10u' from either spelling of a per-variable validation MSE key, else None."""
    for rx in (_OLD_VAR_KEY, _NEW_VAR_KEY):
        m = rx.match(metric_key)
        if m:
            return m.group("var")
    return None


def _var_title(metric_key):
    """Readable panel title for a per-variable metric key."""
    if metric_key in VAR_LABEL:
        return VAR_LABEL[metric_key]
    var = _var_of_key(metric_key)
    if var is None:
        return metric_key.replace("_", " ")
    if var.startswith("pl_") and var[3:] in _FAMILY_NAMES:
        return f"{_FAMILY_NAMES[var[3:]]}, all pressure levels"
    if var in ("all",):
        return "All variables"
    from eval.plotting import variable_spec

    return variable_spec(var[4:] if var.startswith("sfc_") else var).name


def _metric(data, metric_key):
    """The metric under ``metric_key`` or one of its newer spellings, else None."""
    metrics = data["metrics"]
    if metric_key in metrics:
        return metrics[metric_key]
    for alias in _ALIASES.get(metric_key, ()):
        if alias in metrics:
            return metrics[alias]
    var = _var_of_key(metric_key)
    if var is not None:
        for key in metrics:
            if key != metric_key and _var_of_key(key) == var:
                return metrics[key]
    return None


# ─── core drawing helper ──────────────────────────────────────────────────────

def _plot_metric(ax, runs, metric_key, colors_by_name, label=True):
    """Plot one metric for all runs on ax. Returns True if anything was drawn."""
    drawn = False
    for name, data in sorted(runs.items()):
        m = _metric(data, metric_key)
        if m is None:
            continue
        style = dict(colors_by_name[name])
        style["linewidth"] = 1.8
        ax.plot(m["steps"], m["vals"], label=name if label else None, **style)
        drawn = True
    return drawn


def _plot_losses(ax, runs, colors_by_name):
    """Validation loss solid; training loss raw (faint) plus smoothed (bold dashed)."""
    from eval.plotting.spec_helpers import smooth_series

    drawn = False
    for name, data in sorted(runs.items()):
        style = colors_by_name[name]
        t = _metric(data, TRAIN_KEY)
        v = _metric(data, VAL_KEY)
        if t:
            vals = np.asarray(t["vals"], dtype=float)
            span = max(1, int(round(SMOOTH_FRACTION * vals.size)))
            ax.plot(t["steps"], vals, color=style["color"], linewidth=0.8, alpha=RAW_ALPHA,
                    zorder=2)
            ax.plot(t["steps"], smooth_series(vals, span), color=style["color"],
                    linewidth=2.0, linestyle=(0, (5, 2)), zorder=3)
            drawn = True
        if v:
            ax.plot(v["steps"], v["vals"], color=style["color"], linewidth=2.2, zorder=4)
            drawn = True
    return drawn


def _style(ax, ylabel, title=None):
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def _loss_key_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color="0.3", linewidth=2.2, label="validation loss"),
        Line2D([0], [0], color="0.3", linewidth=2.0, linestyle=(0, (5, 2)),
               label="training loss, smoothed"),
        Line2D([0], [0], color="0.3", linewidth=0.8, alpha=0.5, label="training loss, raw"),
    ]


def _add_figure_legend(fig, colors_by_name, ncol=3, runs=None):
    """Single shared legend below the figure: one entry per run plus the line-style key."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], linewidth=2.2,
               label=_run_label(n, runs[n]) if runs else n)
        for n, s in sorted(colors_by_name.items())
    ]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, ncol),
               bbox_to_anchor=(0.5, -0.01))


def _all_val_metric_keys(runs):
    """Return all per-variable validation MSE keys, excluding the aggregate key."""
    keys = set()
    for data in runs.values():
        for key in data["metrics"]:
            if key.startswith("val_mse_metric/"):
                if key == VAL_ALL:
                    continue
                keys.add(key)
            elif key.startswith("val_out_hres_mse_metric/") and _var_of_key(key) not in (None, "all"):
                keys.add(key)
    return sorted(keys)


def _save(fig, output):
    from eval.plotting import save_figure
    from eval.plotting.spec_helpers import format_steps

    from matplotlib.ticker import LogFormatterSciNotation

    for ax in fig.axes:
        if ax.axison and ax.get_xscale() == "linear":
            format_steps(ax)
        if ax.axison and ax.get_yscale() == "log":
            # label intermediate ticks when a log axis spans less than about two decades
            ax.yaxis.set_minor_formatter(
                LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.5)))

    written = save_figure(fig, output, close=True)
    print(f"Saved: {Path(written[0]).resolve()} (+ PDF)")


# ─── plot 1: key variable panels ──────────────────────────────────────────────

def plot_key_vars(runs, output="key_vars.png"):
    """
    Big grid: top row = aggregate train+val, bottom rows = 6 key-var val MSE.

    Layout  (3 cols × 3 rows):
      row 0:  [aggregate train+val (wide)]  [lr schedule]
      row 1:  [10u]  [10v]  [2t]
      row 2:  [tp]   [msl]  [z500]

    Single shared legend below the figure — colours consistent across all panels.
    """
    from eval.plotting import AXIS, eval_style

    with eval_style():
        colors  = _color_map(runs)
        n_vars  = len(KEY_VARS)
        n_cols  = 3
        n_rows  = 1 + (n_vars + n_cols - 1) // n_cols   # 1 header row + var rows

        fig = plt.figure(figsize=(5.4 * n_cols, 3.6 * n_rows))

        # ── row 0: aggregate summary (spans 2 cols) + LR (1 col) ─────────────
        ax_agg = fig.add_subplot(n_rows, n_cols, (1, 2))   # spans cols 1-2
        ax_lr  = fig.add_subplot(n_rows, n_cols, 3)

        _plot_losses(ax_agg, runs, colors)
        _style(ax_agg, LOSS_LABEL, title="Training and validation loss")
        ax_agg.legend(handles=_loss_key_handles(), loc="upper right")

        _plot_metric(ax_lr, runs, LR_KEY, colors, label=False)
        _style(ax_lr, LR_LABEL, title="Learning rate schedule")
        ax_lr.set_yscale("log")

        # ── rows 1+: per-variable val MSE ─────────────────────────────────────
        for i, var_key in enumerate(KEY_VARS):
            row = 1 + i // n_cols
            col = 1 + i % n_cols
            ax  = fig.add_subplot(n_rows, n_cols, row * n_cols + col)

            drawn = _plot_metric(ax, runs, var_key, colors, label=False)
            _style(ax, MSE_LABEL, title=_var_title(var_key))
            if LOG_SCALE_VARS and drawn:
                ax.set_yscale("log")

            if not drawn:
                ax.text(0.5, 0.5, "not logged by these runs", transform=ax.transAxes,
                        ha="center", va="center", color="0.45", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        # ── shared elements ───────────────────────────────────────────────────
        fig.supxlabel(AXIS["step"], y=0.045)
        fig.suptitle("Training curves: loss and validation MSE of the key variables")
        _add_figure_legend(fig, colors, ncol=min(len(runs), 3), runs=runs)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        _save(fig, output)


# ─── plot 2: overview ─────────────────────────────────────────────────────────

def plot_overview(runs, output="overview.png"):
    """
    3-panel overview (+ optional hres panel for downscaling runs):
      (1) aggregate train (dashed) + val (solid) on same axes
      (2) val_mse_metric/all  — overall val MSE
      (3) LR schedule
      (4) [optional] val_out_hres_mse_metric/all — downscaling output MSE
    """
    from eval.plotting import AXIS, eval_style

    with eval_style():
        colors = _color_map(runs)

        # check if hres metrics exist in any run (and are not already the overall panel)
        hres_all = "val_out_hres_mse_metric/all/1_scale_0"
        has_hres = any(
            k.startswith("val_out_hres_mse_metric")
            for data in runs.values()
            for k in data["metrics"]
        ) and any(_metric(d, VAL_ALL) is not _metric(d, hres_all) for d in runs.values())

        n_panels = 3 + (1 if has_hres else 0)
        fig, axs = plt.subplots(n_panels, 1, figsize=(11, 3.2 * n_panels), sharex=True)

        # panel 0: aggregate train + val
        _plot_losses(axs[0], runs, colors)
        _style(axs[0], LOSS_LABEL, title="Training and validation loss")
        axs[0].legend(handles=_loss_key_handles(), loc="upper right")

        # panel 1: val_mse_metric/all  — log scale avoids one run crushing the others
        _plot_metric(axs[1], runs, VAL_ALL, colors, label=False)
        _style(axs[1], MSE_LABEL, title="Validation MSE, all variables")
        axs[1].set_yscale("log")

        # panel 2: LR
        _plot_metric(axs[2], runs, LR_KEY, colors, label=False)
        _style(axs[2], LR_LABEL, title="Learning rate schedule")
        axs[2].set_yscale("log")

        # panel 3: hres MSE (downscaling) if present
        if has_hres:
            _plot_metric(axs[3], runs, hres_all, colors, label=False)
            _style(axs[3], MSE_LABEL, title="Validation MSE of the high-resolution output")
        axs[-1].set_xlabel(AXIS["step"])

        fig.suptitle("Training overview")
        _add_figure_legend(fig, colors, ncol=min(len(runs), 3), runs=runs)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        _save(fig, output)


def plot_all_vars(runs, output="all_vars.png"):
    """Plot every available per-variable validation MSE in one shared grid."""
    from eval.plotting import AXIS, eval_style

    metric_keys = _all_val_metric_keys(runs)
    if not metric_keys:
        raise ValueError("No per-variable validation MSE metrics found.")

    with eval_style():
        colors = _color_map(runs)
        n_vars = len(metric_keys)
        n_cols = min(4, n_vars)
        n_rows = math.ceil(n_vars / n_cols)
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.4 * n_cols, 3.2 * n_rows),
            sharex=False,
            squeeze=False,
        )

        for ax, metric_key in zip(axs.flat, metric_keys):
            drawn = _plot_metric(ax, runs, metric_key, colors, label=False)
            _style(ax, MSE_LABEL if ax in axs[:, 0] else "", title=_var_title(metric_key))
            if LOG_SCALE_VARS and drawn:
                ax.set_yscale("log")
            if not drawn:
                ax.text(0.5, 0.5, "not logged by these runs", transform=ax.transAxes,
                        ha="center", va="center", color="0.45", fontsize=10)

        for ax in axs.flat[n_vars:]:
            ax.axis("off")

        fig.supxlabel(AXIS["step"], y=0.045)
        fig.suptitle("Validation MSE of every variable")
        _add_figure_legend(fig, colors, ncol=min(len(runs), 3), runs=runs)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        _save(fig, output)


# ─── convenience: plot all ────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path.home() / "perm" / "training_logs_lots"


def plot_all(runs, output_dir=None):
    """Generate the standard MLflow plot bundle. Call this from scripts or notebooks."""
    d = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    plot_key_vars(runs, output=d / "key_vars.png")
    plot_overview(runs, output=d / "overview.png")
    plot_all_vars(runs, output=d / "all_vars.png")


# ─── standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    # lazy import so the module is usable without loader on sys.path
    sys.path.insert(0, str(Path(__file__).parent))
    from loader import load, filter_runs

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp_dir    = sys.argv[1]
    min_steps  = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    name_filt  = sys.argv[3] if len(sys.argv) > 3 else None

    runs = load(exp_dir)
    runs = filter_runs(runs, min_steps=min_steps, name_contains=name_filt)

    print(f"\nPlotting {len(runs)} run(s):")
    for name, d in sorted(runs.items(), key=lambda x: -x[1]["max_step"]):
        print(f"  {d['max_step']:>8,}  {name}")
    print()

    output_dir = sys.argv[4] if len(sys.argv) > 4 else None
    plot_all(runs, output_dir=output_dir)
