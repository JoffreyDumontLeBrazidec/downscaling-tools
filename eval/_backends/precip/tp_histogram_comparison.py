#!/usr/bin/env python3
"""
tp_histogram_comparison.py — PDF comparing TP distributions across INPUT
(interpolation baseline), TRUTH, and PREDICTION.

All distributions are accumulated as fixed-bin counts while streaming through
the prediction files, so memory stays flat at o2560 scale (a full in-memory
load of 100 slices x 26.3M points would need ~30 GB).

Truth: the embedded `y` tp channel when populated, else the per-date truth
GRIB given via --truth-grib-tpl (the o1280->o2560 main-lane bundles carried
no tp truth). Input: the embedded `x_interp` tp channel when it is a real
series, else the o1280 driver member tp interpolated through the cached
nearest-neighbour index (--baseline-grib-tpl / --interp-index-cache); tp is
output-only on the o2560 lane, so its exported x_interp is all zero and is
never plotted as data.

All axes are in mm per 6h window. Colours follow the house roles: truth black,
model red, interpolated input blue dashed. The pages go to one PDF plus a PNG
per page in ``<name>_pages/``.

Usage:
    python -m eval._backends.precip.tp_histogram_comparison \\
        --predictions-dir /path/to/predictions/ \\
        --out-pdf /path/to/tp_histograms.pdf \\
        --run-label "o2560 pristine 300k"
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from eval._backends.precip.sources import (
    LresInterpBaseline,
    PrecipTruthSource,
    is_degenerate_channel,
)

HIGHLIGHT_STEPS = [6, 24, 48, 72, 120]

SERIES = ("input", "truth", "pred")
SERIES_LABEL = {"input": "Input (interpolated)", "truth": "Truth", "pred": "Model"}
# Colours by role (eval.plotting.roles): truth black, model red, input blue dashed.
SERIES_ROLE = {"input": "input", "truth": "truth", "pred": "model"}
SERIES_COLOR = {"input": "#1f77b4", "truth": "#000000", "pred": "#d62728"}
TP_LABEL = "Total precipitation, 6 h accumulation (mm)"

MM = 1000.0


def parse_prediction_filename(path: Path):
    m = re.match(r"predictions_(\d{8})_step(\d{3})\.nc", path.name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


class StreamingDist:
    """Fixed-bin distribution accumulator (mm). Negatives clip into bin 0."""

    EDGES = np.concatenate([[0.0], np.geomspace(0.01, 2048.0, 600)])

    def __init__(self):
        self.counts = np.zeros(self.EDGES.size - 1, dtype=np.int64)
        self.n = 0
        self.neg = 0
        self.max = -np.inf

    def update(self, vals_mm: np.ndarray) -> None:
        v = vals_mm[np.isfinite(vals_mm)]
        if v.size == 0:
            return
        self.n += v.size
        self.neg += int((v < 0).sum())
        self.max = max(self.max, float(v.max()))
        self.counts += np.histogram(np.clip(v, 0.0, self.EDGES[-1]),
                                    bins=self.EDGES)[0]

    @property
    def empty(self) -> bool:
        return self.n == 0

    def merged(self, other: "StreamingDist") -> "StreamingDist":
        out = StreamingDist()
        out.counts = self.counts + other.counts
        out.n = self.n + other.n
        out.neg = self.neg + other.neg
        out.max = max(self.max, other.max)
        return out

    def density(self) -> np.ndarray:
        widths = np.diff(self.EDGES)
        return self.counts / max(self.n, 1) / widths

    def cdf(self) -> np.ndarray:
        c = np.cumsum(self.counts)
        return c / max(self.n, 1)

    def quantile(self, q: float) -> float:
        c = np.cumsum(self.counts)
        if c[-1] == 0:
            return 0.0
        idx = int(np.searchsorted(c, q / 100.0 * c[-1], side="left"))
        idx = min(idx, self.EDGES.size - 2)
        return float(0.5 * (self.EDGES[idx] + self.EDGES[idx + 1]))


def find_tp_index(ds: xr.Dataset, var: str) -> int:
    ws = [str(s) for s in ds["weather_state"].values]
    if var not in ws:
        raise ValueError(f"'{var}' not found in weather_state: {ws}")
    return ws.index(var)


def _member_channel(ds: xr.Dataset, name: str, tp_idx: int, mi: int) -> np.ndarray:
    da = ds[name]
    if "sample" in da.dims:
        da = da.isel(sample=0)
    if "ensemble_member" in da.dims:
        da = da.isel(ensemble_member=mi)
    return da.values[:, tp_idx].astype(np.float64)


def _member_id(ds: xr.Dataset, mi: int) -> int:
    raw = str(ds.attrs.get("member_ids", ""))
    if raw:
        try:
            ids = [int(x) for x in raw.split(",")]
            return ids[mi]
        except (ValueError, IndexError):
            pass
    return mi + 1


def accumulate_tp_by_step(
    predictions_dir: Path,
    *,
    ensemble_member_index: int = 0,
    var: str = "tp",
    truth_grib_tpl: str = "",
    baseline_grib_tpl: str = "",
    interp_index_cache: str = "",
) -> dict[int, dict[str, StreamingDist]]:
    """Stream every predictions file into per-step per-series StreamingDists."""
    files = sorted(predictions_dir.glob("predictions_*_step*.nc"))
    if not files:
        raise FileNotFoundError(f"No predictions_*.nc in {predictions_dir}")

    truth_src = baseline_src = None
    truth_mode = baseline_mode = None
    step_data: dict[int, dict[str, StreamingDist]] = {}

    for f in files:
        date, step = parse_prediction_filename(f)
        if step is None:
            continue
        ds = xr.open_dataset(f)
        try:
            ti = find_tp_index(ds, var)
            mi = ensemble_member_index
            pred = _member_channel(ds, "y_pred", ti, mi)

            truth = _member_channel(ds, "y", ti, mi)
            if truth_mode is None:
                truth_mode = "embedded-y" if np.isfinite(truth).mean() > 0.99 \
                    else ("grib" if truth_grib_tpl else "missing")
                print(f"truth source: {truth_mode}")
            if truth_mode == "grib":
                truth_src = truth_src or PrecipTruthSource(truth_grib_tpl, var=var)
                truth = truth_src.load(date, step).astype(np.float64)
                truth_src.verify_grid(ds["lat_hres"].values, ds["lon_hres"].values)
            elif truth_mode == "missing":
                truth = None

            if "x_interp" in ds.variables:
                inp = _member_channel(ds, "x_interp", ti, mi)
            else:
                inp = None
            if baseline_mode is None:
                if inp is not None and not is_degenerate_channel(inp):
                    baseline_mode = "x_interp"
                elif baseline_grib_tpl:
                    baseline_mode = "grib"
                else:
                    baseline_mode = "missing"
                print(f"input/baseline source: {baseline_mode}")
            if baseline_mode == "grib":
                if baseline_src is None:
                    baseline_src = LresInterpBaseline(
                        baseline_grib_tpl, interp_index_cache or None, var=var)
                    baseline_src.ensure_index(
                        ds["lat_hres"].values, ds["lon_hres"].values,
                        probe_date=date)
                inp = baseline_src.load(date, step, _member_id(ds, mi)).astype(np.float64)
            elif baseline_mode == "missing":
                inp = None
        finally:
            ds.close()

        bucket = step_data.setdefault(
            step, {s: StreamingDist() for s in SERIES})
        bucket["pred"].update(pred * MM)
        if truth is not None:
            bucket["truth"].update(truth * MM)
        if inp is not None:
            bucket["input"].update(inp * MM)
    return step_data


# ---------------------------------------------------------------------------
# Pages (all counts-based; x axes in mm/6h)
# ---------------------------------------------------------------------------

def _overall(step_data: dict, key: str) -> StreamingDist:
    total = StreamingDist()
    for bucket in step_data.values():
        total = total.merged(bucket[key])
    return total


def _count_text(n: int) -> str:
    """Compact sample count: 1234 -> "1234", 2.5e6 -> "2.5 M"."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} M"
    if n >= 10_000:
        return f"{n / 1e3:.0f} k"
    return str(n)


def _series_label(key: str, dist: StreamingDist) -> str:
    return f"{SERIES_LABEL[key]} (n = {_count_text(dist.n)} values)"


def _series_style(key: str, **overrides) -> dict:
    from eval.plotting import role_style

    return role_style(SERIES_ROLE[key], **overrides)


def _emit(pdf, fig, name: str) -> None:
    """Add a page to a FigureBook (PDF + PNG) or, for old callers, to a PdfPages."""
    if hasattr(pdf, "add"):
        pdf.add(fig, name=name)
    else:
        pdf.savefig(fig)
        plt.close(fig)


def _plot_density(ax, dist: StreamingDist, label: str, color: str, *,
                  xlim, log: bool, **style):
    if dist.empty:
        return
    centers = 0.5 * (dist.EDGES[:-1] + dist.EDGES[1:])
    mask = (centers >= xlim[0]) & (centers <= xlim[1])
    kw = {"color": color, "linewidth": 1.4}
    kw.update(style)
    ax.step(centers[mask], dist.density()[mask], where="mid", label=label, **kw)
    if log:
        ax.set_yscale("log")
    ax.set_xlim(*xlim)


def matrix_page(pdf: PdfPages, step_data: dict, run_label: str):
    """Rows x 3 cols (linear / log / CDF), one row per HIGHLIGHT_STEP."""
    from eval.plotting import AXIS, eval_style
    from eval.plotting.labels import pdf_label

    rows = [s for s in HIGHLIGHT_STEPS if s in step_data]
    if not rows:
        return
    n = len(rows)
    with eval_style():
        fig, axes = plt.subplots(n, 3, figsize=(15, 3.0 * n + 1.0), squeeze=False)
        fig.suptitle(f"Total precipitation distributions by lead time: {run_label}")
        for r, step in enumerate(rows):
            truth = step_data[step]["truth"]
            vmax = max(truth.quantile(99.9), 1.0) if not truth.empty else \
                max(step_data[step]["pred"].quantile(99.9), 1.0)
            ax_lin, ax_log, ax_cdf = axes[r]
            for k in SERIES:
                dist = step_data[step][k]
                style = _series_style(k, linewidth=1.6)
                label = _series_label(k, dist)
                _plot_density(ax_lin, dist, label, style.pop("color"),
                              xlim=(0, vmax), log=False, **style)
                style = _series_style(k, linewidth=1.6)
                _plot_density(ax_log, dist, label, style.pop("color"),
                              xlim=(0, vmax), log=True, **style)
                if not dist.empty:
                    centers = 0.5 * (dist.EDGES[:-1] + dist.EDGES[1:])
                    ax_cdf.plot(centers, dist.cdf(), label=label, **_series_style(k, linewidth=1.6))
                    ax_cdf.set_xlim(0, vmax)
            ax_lin.set_ylabel(f"Lead time {step} h\n{pdf_label('mm')}", fontsize=9)
            ax_log.set_ylabel(pdf_label("mm"), fontsize=9)
            ax_cdf.set_ylabel(AXIS["cdf"], fontsize=9)
            ax_lin.legend(fontsize=7, loc="upper right")
            if r == 0:
                ax_lin.set_title("Linear density axis")
                ax_log.set_title("Logarithmic density axis")
                ax_cdf.set_title("Cumulative distribution")
            if r == n - 1:
                for ax in (ax_lin, ax_log, ax_cdf):
                    ax.set_xlabel(TP_LABEL)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        _emit(pdf, fig, "by_lead_time")


def all_steps_overlay_page(pdf: PdfPages, step_data: dict, run_label: str):
    """One column per series, all leads overlaid; the colour encodes lead time (viridis)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from eval.plotting import AXIS, eval_style
    from eval.plotting.labels import pdf_label

    steps = sorted(step_data.keys())
    if not steps:
        return
    with eval_style():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        fig.suptitle(f"Total precipitation distribution at every lead time: {run_label}")
        lead_norm = Normalize(vmin=min(steps), vmax=max(steps) if max(steps) > min(steps) else min(steps) + 1)
        cmap = plt.get_cmap("viridis")
        truth_total = _overall(step_data, "truth")
        vmax = max(truth_total.quantile(99.9), 1.0) if not truth_total.empty else \
            max(_overall(step_data, "pred").quantile(99.9), 1.0)
        for col, k in enumerate(SERIES):
            ax = axes[col]
            n_total = 0
            for step in steps:
                dist = step_data[step][k]
                if dist.empty:
                    continue
                n_total += dist.n
                centers = 0.5 * (dist.EDGES[:-1] + dist.EDGES[1:])
                mask = centers <= vmax
                ax.step(centers[mask], dist.density()[mask], where="mid",
                        color=cmap(lead_norm(step)), linewidth=1.0, alpha=0.8)
            ax.set_title(f"{SERIES_LABEL[k]} (n = {_count_text(n_total)} values)")
            ax.set_xlabel(TP_LABEL)
            ax.set_yscale("log")
            ax.set_xlim(0, vmax)
            if col == 0:
                ax.set_ylabel(f"{pdf_label('mm')}, logarithmic axis")
        cb = fig.colorbar(ScalarMappable(norm=lead_norm, cmap=cmap), ax=list(axes), pad=0.015, fraction=0.03)
        cb.set_label(AXIS["lead"])
        _emit(pdf, fig, "all_lead_times")


def compact_page(pdf: PdfPages, step_data: dict, run_label: str):
    """One-page summary: overall distribution + wet-tail zoom (all series)."""
    from eval.plotting import eval_style
    from eval.plotting.labels import pdf_label

    dists = {k: _overall(step_data, k) for k in SERIES}
    if dists["pred"].empty:
        return
    ref = dists["truth"] if not dists["truth"].empty else dists["pred"]
    q995 = ref.quantile(99.5)
    xmax = max(5.0, ref.quantile(99.95), dists["pred"].quantile(99.95))
    steps = ", ".join(f"{s}" for s in sorted(step_data))
    with eval_style():
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
        fig.suptitle(f"Total precipitation distribution: {run_label} (lead times {steps} h)")
        for ax, title, xlim in [
            (axes[0], "Overall distribution", (0.0, max(2.0, q995))),
            (axes[1], "Wet tail", (max(0.25, q995 * 0.15), xmax)),
        ]:
            for k in SERIES:
                if not dists[k].empty:
                    style = _series_style(k, linewidth=1.8)
                    _plot_density(ax, dists[k], _series_label(k, dists[k]), style.pop("color"),
                                  xlim=xlim, log=True, **style)
            ax.set_title(title)
            ax.set_xlabel(TP_LABEL)
            ax.set_ylabel(f"{pdf_label('mm')}, logarithmic axis")
        axes[1].legend(loc="upper right")
        _emit(pdf, fig, "overall")


def main():
    parser = argparse.ArgumentParser(
        description="TP histogram comparison: input vs truth vs prediction")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--ensemble-member-index", type=int, default=0)
    parser.add_argument("--style", choices=("compact", "diagnostic"), default="compact")
    parser.add_argument("--var", type=str, default="tp")
    parser.add_argument("--truth-grib-tpl", type=str, default="",
                        help="Per-date truth GRIB template with {date}; used "
                             "when the embedded y tp channel is missing.")
    parser.add_argument("--baseline-grib-tpl", type=str, default="",
                        help="Per-date o1280 member tp GRIB template with "
                             "{date}; used when x_interp tp is degenerate.")
    parser.add_argument("--interp-index-cache", type=str, default="",
                        help="Path of the cached lres->hres NN interp index.")
    args = parser.parse_args()

    if not args.predictions_dir.is_dir():
        sys.exit(f"predictions-dir not found: {args.predictions_dir}")

    print(f"Streaming TP data from {args.predictions_dir}...")
    step_data = accumulate_tp_by_step(
        args.predictions_dir,
        ensemble_member_index=args.ensemble_member_index,
        var=args.var,
        truth_grib_tpl=args.truth_grib_tpl,
        baseline_grib_tpl=args.baseline_grib_tpl,
        interp_index_cache=args.interp_index_cache,
    )
    if not step_data:
        sys.exit("No prediction files found")

    steps = sorted(step_data.keys())
    has_truth = any(not step_data[s]["truth"].empty for s in steps)
    has_input = any(not step_data[s]["input"].empty for s in steps)
    print(f"Found {len(steps)} steps: {steps[0]:03d}..{steps[-1]:03d}"
          f" | truth={'yes' if has_truth else 'NO'}"
          f" | input/baseline={'yes' if has_input else 'NO'}")

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    from eval.plotting import FigureBook

    with FigureBook(args.out_pdf, png=True) as pdf:
        if args.style == "compact":
            compact_page(pdf, step_data, args.run_label)
        else:
            matrix_page(pdf, step_data, args.run_label)
            all_steps_overlay_page(pdf, step_data, args.run_label)

    print(f"Done: {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
