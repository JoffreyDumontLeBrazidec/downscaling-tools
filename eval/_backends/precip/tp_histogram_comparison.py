#!/usr/bin/env python3
"""
tp_histogram_comparison.py — PDF comparing TP distributions across INPUT
(x_interp), TRUTH (y), and PREDICTION (y_pred).

Default layout:
  compact: one page with overall truth-vs-pred distribution and wet-tail zoom.

Diagnostic layout:
  1) Matrix page: rows = HIGHLIGHT_STEPS, cols = linear hist / log-y hist / CDF
     (each cell overlays the three series).
  2) All-steps overlay: one column per series, all leads stacked with a
     viridis colour gradient.

Usage:
    python tp_histogram_comparison.py \\
        --predictions-dir /path/to/predictions/ \\
        --out-pdf /path/to/tp_histograms.pdf \\
        --run-label "310c04ad tp-ln12 6h view"
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages


HIGHLIGHT_STEPS = [6, 24, 48, 72, 120]

# Series naming + colours (input/truth/pred). Truth & pred mirror the matplotlib
# default cycle so they match other evaluators' plots; input gets a neutral grey.
SERIES = ("input", "truth", "pred")
SERIES_LABEL = {"input": "Input", "truth": "Truth", "pred": "Pred"}
SERIES_COLOR = {"input": "#555555", "truth": "C0", "pred": "C1"}


def parse_prediction_filename(path: Path):
    m = re.match(r"predictions_(\d{8})_step(\d{3})\.nc", path.name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def find_tp_index(ds: xr.Dataset) -> int:
    ws = list(ds["weather_state"].values)
    if "tp" not in ws:
        raise ValueError(f"'tp' not found in weather_state: {ws}")
    return ws.index("tp")


def _tp_member_values(ds: xr.Dataset, name: str, tp_idx: int, ensemble_member_index: int) -> np.ndarray:
    da = ds[name]
    if "weather_state" in da.dims:
        da = da.isel(weather_state=tp_idx)
    else:
        da = da[..., tp_idx]
    if "sample" in da.dims:
        da = da.isel(sample=0)
    if "ensemble_member" in da.dims:
        da = da.isel(ensemble_member=ensemble_member_index)
    return np.asarray(da.values).ravel().astype(np.float32, copy=False)


def load_tp_by_step(predictions_dir: Path, *, ensemble_member_index: int = 0) -> dict:
    """Load tp values for input/truth/pred, grouped by step.

    `x_interp` is the LRES-to-HRES bilinear of the model input — same grid as
    `y`/`y_pred`, so the three series concatenate identically.
    """
    files = sorted(predictions_dir.glob("predictions_*_step*.nc"))
    step_data: dict[int, dict[str, list[np.ndarray]]] = {}
    for f in files:
        _, step = parse_prediction_filename(f)
        if step is None:
            continue
        ds = xr.open_dataset(f)
        try:
            tp_idx = find_tp_index(ds)
            truth = _tp_member_values(ds, "y", tp_idx, ensemble_member_index)
            pred = _tp_member_values(ds, "y_pred", tp_idx, ensemble_member_index)
            if "x_interp" in ds.variables:
                inp = _tp_member_values(ds, "x_interp", tp_idx, ensemble_member_index)
            else:
                inp = None
        finally:
            ds.close()
        bucket = step_data.setdefault(step, {s: [] for s in SERIES})
        bucket["truth"].append(truth)
        bucket["pred"].append(pred)
        if inp is not None:
            bucket["input"].append(inp)

    for step, bucket in step_data.items():
        for s in SERIES:
            if bucket[s]:
                bucket[s] = np.concatenate(bucket[s])
            else:
                bucket[s] = np.array([], dtype=np.float64)
    return step_data


def _values_for_clip(arr: np.ndarray) -> np.ndarray:
    """Hist display: clip negatives to zero (pred can briefly go negative)."""
    return np.maximum(arr, 0.0)


def _plot_hist(ax, arr: np.ndarray, bins: np.ndarray, label: str, color: str, *, log: bool):
    if arr.size == 0:
        return
    ax.hist(_values_for_clip(arr), bins=bins, alpha=0.5, density=True,
            label=label, color=color, histtype="stepfilled", edgecolor=color)
    if log:
        ax.set_yscale("log")


def _plot_cdf(ax, arr: np.ndarray, label: str, color: str, *, xmax: float):
    if arr.size == 0:
        return
    sorted_v = np.sort(_values_for_clip(arr))
    cdf = np.linspace(0, 1, sorted_v.size)
    n_plot = min(10000, sorted_v.size)
    idx = np.linspace(0, sorted_v.size - 1, n_plot, dtype=int)
    ax.plot(sorted_v[idx], cdf[idx], label=label, color=color, linewidth=1.2)
    ax.set_xlim(0, xmax)


def matrix_page(pdf: PdfPages, step_data: dict, run_label: str):
    """Page 2: rows × 3 cols (linear / log / CDF), one row per HIGHLIGHT_STEP."""
    rows = [s for s in HIGHLIGHT_STEPS if s in step_data]
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.0 * n + 1.0), squeeze=False)
    fig.suptitle(f"TP distributions per lead — input / truth / pred — {run_label}",
                 fontsize=11)

    for r, step in enumerate(rows):
        truth = step_data[step]["truth"]
        # vmax driven by truth's heavy tail; pred/input use same range
        vmax = float(np.percentile(truth, 99.9)) if truth.size else 1e-6
        if vmax <= 0:
            vmax = max(truth.max() if truth.size else 0.0, 1e-6)
        bins = np.linspace(0, vmax, 100)

        ax_lin, ax_log, ax_cdf = axes[r]
        for k in SERIES:
            arr = step_data[step][k]
            _plot_hist(ax_lin, arr, bins, SERIES_LABEL[k], SERIES_COLOR[k], log=False)
            _plot_hist(ax_log, arr, bins, SERIES_LABEL[k], SERIES_COLOR[k], log=True)
            _plot_cdf(ax_cdf, arr, SERIES_LABEL[k], SERIES_COLOR[k], xmax=vmax)

        ax_lin.set_ylabel(f"step {step:03d}h\nDensity", fontsize=9)
        ax_cdf.set_ylabel("CDF", fontsize=9)
        if r == 0:
            ax_lin.set_title("Linear", fontsize=10)
            ax_log.set_title("Log-y", fontsize=10)
            ax_cdf.set_title("CDF", fontsize=10)
            ax_lin.legend(fontsize=8, loc="upper right")
        if r == n - 1:
            for ax in (ax_lin, ax_log, ax_cdf):
                ax.set_xlabel("TP (m)", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def all_steps_overlay_page(pdf: PdfPages, step_data: dict, run_label: str):
    """Page 3: one column per series, all leads overlaid (viridis gradient)."""
    steps = sorted(step_data.keys())
    if not steps:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"All-steps overlay — {run_label}", fontsize=11)

    cmap = plt.cm.viridis(np.linspace(0, 1, len(steps)))
    all_truth = np.concatenate([step_data[s]["truth"] for s in steps
                                if step_data[s]["truth"].size > 0])
    vmax = float(np.percentile(all_truth, 99.9)) if all_truth.size else 1e-6
    if vmax <= 0:
        vmax = 1e-6
    bins = np.linspace(0, vmax, 80)

    for col, k in enumerate(SERIES):
        ax = axes[col]
        for i, step in enumerate(steps):
            arr = step_data[step][k]
            if arr.size == 0:
                continue
            label = f"{step:03d}h" if step in HIGHLIGHT_STEPS else None
            ax.hist(_values_for_clip(arr), bins=bins, alpha=0.4, density=True,
                    color=cmap[i], histtype="step", linewidth=0.9, label=label)
        ax.set_title(SERIES_LABEL[k], fontsize=10)
        ax.set_xlabel("TP (m)")
        ax.set_yscale("log")
        if col == 0:
            ax.set_ylabel("Density (log)")
        if col == 2:
            ax.legend(fontsize=7, ncol=2, title="lead", loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _combined_values(step_data: dict, key: str) -> np.ndarray:
    arrays = [bucket[key] for bucket in step_data.values() if bucket[key].size > 0]
    if not arrays:
        return np.array([], dtype=np.float64)
    return _values_for_clip(np.concatenate(arrays)) * 1000.0


def _density(counts: np.ndarray, bins: np.ndarray, n_total: int) -> np.ndarray:
    return counts / max(n_total, 1) / np.diff(bins)


def _draw_compact_hist_panel(
    ax,
    bins: np.ndarray,
    truth_density: np.ndarray,
    pred_density: np.ndarray,
    *,
    title: str,
    xlim: tuple[float, float],
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])
    mask = (centers >= xlim[0]) & (centers <= xlim[1])
    ax.step(centers[mask], truth_density[mask], where="mid", color="#222222", linewidth=1.7, label="Truth")
    ax.step(centers[mask], pred_density[mask], where="mid", color="#d95f02", linewidth=1.7, label="Prediction")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.grid(True, which="major", alpha=0.25)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("TP accumulation (mm / 6h)")
    ax.set_ylabel("Density")


def compact_page(pdf: PdfPages, step_data: dict, run_label: str):
    """One-page truth-vs-pred summary: overall distribution + wet-tail zoom."""
    truth = _combined_values(step_data, "truth")
    pred = _combined_values(step_data, "pred")
    if truth.size == 0 or pred.size == 0:
        return

    combined = np.concatenate([truth, pred])
    q995 = float(np.nanpercentile(combined, 99.5))
    q9995 = float(np.nanpercentile(combined, 99.95))
    xmax = max(5.0, q9995)
    bins = np.linspace(0.0, xmax, 180)
    truth_counts = np.histogram(np.clip(truth, bins[0], bins[-1]), bins=bins)[0]
    pred_counts = np.histogram(np.clip(pred, bins[0], bins[-1]), bins=bins)[0]
    truth_density = _density(truth_counts, bins, truth.size)
    pred_density = _density(pred_counts, bins, pred.size)

    steps = ", ".join(f"{s}h" for s in sorted(step_data))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    fig.suptitle(f"TP distribution: truth vs prediction | {run_label} | steps {steps}", fontsize=12)
    _draw_compact_hist_panel(
        axes[0],
        bins,
        truth_density,
        pred_density,
        title="Overall distribution",
        xlim=(0.0, max(2.0, q995)),
    )
    _draw_compact_hist_panel(
        axes[1],
        bins,
        truth_density,
        pred_density,
        title="Wet-tail zoom",
        xlim=(max(0.25, q995 * 0.15), xmax),
    )
    axes[1].legend(frameon=False, loc="upper right")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="TP histogram comparison: input vs truth vs prediction"
    )
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--ensemble-member-index", type=int, default=0)
    parser.add_argument("--style", choices=("compact", "diagnostic"), default="compact")
    args = parser.parse_args()

    if not args.predictions_dir.is_dir():
        sys.exit(f"predictions-dir not found: {args.predictions_dir}")

    print(f"Loading TP data from {args.predictions_dir}...")
    step_data = load_tp_by_step(args.predictions_dir, ensemble_member_index=args.ensemble_member_index)
    if not step_data:
        sys.exit("No prediction files found")

    steps = sorted(step_data.keys())
    has_input = any(step_data[s]["input"].size > 0 for s in steps)
    print(f"Found {len(steps)} steps: {steps[0]:03d}..{steps[-1]:03d}"
          f"{' (input present)' if has_input else ' (no input in NCs)'}")

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(args.out_pdf)) as pdf:
        if args.style == "compact":
            compact_page(pdf, step_data, args.run_label)
        else:
            matrix_page(pdf, step_data, args.run_label)
            all_steps_overlay_page(pdf, step_data, args.run_label)

    print(f"Done: {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
