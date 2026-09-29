#!/usr/bin/env python3
"""
tp_histogram_comparison.py — distribution of 6 h precipitation for TRUTH, MODEL
and the interpolated INPUT (interpolation baseline), on one support.

All distributions are accumulated as fixed-bin counts while streaming through
the prediction files, so memory stays flat at o2560 scale (a full in-memory
load of 100 slices x 26.3M points would need ~30 GB).

Truth: the embedded `y` tp channel when populated, else the per-date truth
GRIB given via --truth-grib-tpl (the o1280->o2560 main-lane bundles carried
no tp truth). Input: the embedded `x_interp` tp channel when it is a real
series, else the o1280 driver member tp interpolated through the cached
nearest-neighbour index (--baseline-grib-tpl / --interp-index-cache); tp is
output-only on the o2560 lane, so its exported x_interp is all zero and is
never plotted as data. On a regional run the truth is served on the run's own
support (verify_grid before the first load), so all three series count the
same grid points.

Two pages, one PDF plus a PNG per page in ``<name>_pages/``:

1. ``distribution``: all lead times pooled. Top, the exceedance frequency
   (fraction of grid points at or above a threshold) against the threshold,
   both axes logarithmic, from the wet threshold (0.1 mm) to the largest value.
   Bottom, the frequency bias (series exceedance frequency / truth exceedance
   frequency) of the model and of the input. The exceedance curve is used
   instead of a histogram because it does not alias the packing quantum of
   the GRIB truth into empty and full bins, and the wet tail stays readable.
2. ``by_lead_time``: the same two rows, one small column per lead time
   (at most six lead times).

The bin counts behind both pages are written to ``<name>_counts.npz`` so the
pages can be redrawn without re-reading the predictions (``load_counts``).

All axes are in mm per 6 h window. Colours follow the house roles: truth black,
model red, interpolated input blue dashed.

Usage:
    python -m eval.evaluators.precip_dist.core.tp_histogram_comparison \\
        --predictions-dir /path/to/predictions/ \\
        --out-pdf /path/to/tp_histograms.pdf \\
        --run-label "o2560 pristine 300k"
"""
import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eval.shared.precip.sources import (
    LresInterpBaseline,
    PrecipTruthSource,
    is_degenerate_channel,
)

HIGHLIGHT_STEPS = [6, 24, 48, 72, 120]
MAX_LEAD_PANELS = 6

SERIES = ("input", "truth", "pred")
SERIES_LABEL = {"input": "Input (interpolated)", "truth": "Truth", "pred": "Model"}
# Colours by role (eval.plotting.roles): truth black, model red, input blue dashed.
SERIES_ROLE = {"input": "input", "truth": "truth", "pred": "model"}
SERIES_COLOR = {"input": "#1f77b4", "truth": "#000000", "pred": "#d62728"}
TP_LABEL = "Total precipitation, 6 h accumulation (mm)"
THRESHOLD_LABEL = "Threshold, 6 h precipitation (mm)"

MM = 1000.0
WET_THRESHOLD_MM = 0.1
# The frequency bias is drawn only where the truth has at least this many values
# at or above the threshold; above that the ratio is sampling noise.
MIN_TRUTH_COUNT = 100


def parse_prediction_filename(path: Path):
    m = re.match(r"predictions_(\d{8})_step(\d{3})\.nc", path.name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


class StreamingDist:
    """Fixed-bin distribution accumulator (mm). Negatives clip into bin 0.

    Bin 0 is [0, 0.01) mm; above it the edges are logarithmic, 120 per decade,
    with 0.1, 1, 10, 100 and 1000 mm as exact edges.
    """

    EDGES = np.concatenate([[0.0], 10.0 ** np.linspace(-2.0, 3.5, 661)])

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

    def count_at_or_above(self) -> np.ndarray:
        """Number of values at or above each lower bin edge, EDGES[:-1]."""
        return np.cumsum(self.counts[::-1])[::-1]

    def exceedance(self) -> np.ndarray:
        """Fraction of values at or above each lower bin edge, EDGES[:-1]."""
        return self.count_at_or_above() / max(self.n, 1)

    def fraction_at_or_above(self, threshold_mm: float) -> float:
        i = int(np.searchsorted(self.EDGES, threshold_mm * (1 - 1e-9)))
        i = min(max(i, 0), self.counts.size - 1)
        return float(self.counts[i:].sum()) / max(self.n, 1)

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
    info: dict | None = None,
) -> dict[int, dict[str, StreamingDist]]:
    """Stream every predictions file into per-step per-series StreamingDists.

    When ``info`` is a dict it is filled with the truth and input sources, the
    member id and the dates, for the figure footers.
    """
    files = sorted(predictions_dir.glob("predictions_*_step*.nc"))
    if not files:
        raise FileNotFoundError(f"No predictions_*.nc in {predictions_dir}")

    truth_src = baseline_src = None
    truth_mode = baseline_mode = None
    step_data: dict[int, dict[str, StreamingDist]] = {}
    dates: set[str] = set()
    member_id = None

    for f in files:
        date, step = parse_prediction_filename(f)
        if step is None:
            continue
        dates.add(date)
        ds = xr.open_dataset(f)
        try:
            ti = find_tp_index(ds, var)
            mi = ensemble_member_index
            if member_id is None:
                member_id = _member_id(ds, mi)
            pred = _member_channel(ds, "y_pred", ti, mi)

            truth = _member_channel(ds, "y", ti, mi)
            if truth_mode is None:
                truth_mode = "embedded-y" if np.isfinite(truth).mean() > 0.99 \
                    else ("grib" if truth_grib_tpl else "missing")
                print(f"truth source: {truth_mode}")
            if truth_mode == "grib":
                truth_src = truth_src or PrecipTruthSource(truth_grib_tpl, var=var)
                # Declare the run's grid first: on a regional run the support
                # index must exist before the first truth is served.
                truth_src.verify_grid(ds["lat_hres"].values, ds["lon_hres"].values)
                truth = truth_src.load(date, step).astype(np.float64)
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
    if info is not None:
        info.update({
            "truth_source": (f"GRIB {Path(truth_grib_tpl).name}" if truth_mode == "grib"
                             else {"embedded-y": "embedded y", "missing": "none"}.get(
                                 truth_mode, str(truth_mode))),
            "input_source": (
                f"GRIB {Path(baseline_grib_tpl).name}, nearest-neighbour interpolated"
                if baseline_mode == "grib"
                else {"x_interp": "embedded x_interp", "missing": "none"}.get(
                    baseline_mode, str(baseline_mode))),
            "member_id": member_id,
            "dates": sorted(dates),
        })
    return step_data


# ---------------------------------------------------------------------------
# Counts file (the data behind the pages)
# ---------------------------------------------------------------------------

def save_counts(path: Path, step_data: dict, info: dict) -> None:
    arrays = {"edges": StreamingDist.EDGES}
    for step, bucket in step_data.items():
        for k, d in bucket.items():
            arrays[f"s{step:03d}_{k}_counts"] = d.counts
            arrays[f"s{step:03d}_{k}_stats"] = np.array(
                [d.n, d.neg, d.max if np.isfinite(d.max) else np.nan])
    arrays["info_json"] = np.array(json.dumps(info))
    np.savez_compressed(path, **arrays)


def load_counts(path: Path) -> tuple[dict, dict]:
    """Inverse of save_counts: (step_data, info)."""
    z = np.load(path, allow_pickle=False)
    if not np.array_equal(z["edges"], StreamingDist.EDGES):
        raise ValueError(f"{path}: bin edges differ from StreamingDist.EDGES")
    step_data: dict[int, dict[str, StreamingDist]] = {}
    for key in z.files:
        m = re.match(r"s(\d{3})_(\w+)_counts$", key)
        if not m:
            continue
        step, k = int(m.group(1)), m.group(2)
        d = StreamingDist()
        d.counts = z[key].astype(np.int64)
        n, neg, mx = z[f"s{step:03d}_{k}_stats"]
        d.n, d.neg = int(n), int(neg)
        d.max = float(mx) if np.isfinite(mx) else -np.inf
        step_data.setdefault(step, {s: StreamingDist() for s in SERIES})[k] = d
    return step_data, json.loads(str(z["info_json"]))


# ---------------------------------------------------------------------------
# Pages (all counts-based; x axes in mm per 6 h window)
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


def _mm_text(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


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


def _present(dists: dict) -> list[str]:
    """Series with data, in drawing order (input under truth under model)."""
    return [k for k in SERIES if not dists[k].empty]


def _draw_exceedance(ax, dists: dict, *, legend_stats: bool, lw_scale: float = 1.0) -> None:
    edges = StreamingDist.EDGES[:-1]
    keep = edges >= WET_THRESHOLD_MM * (1 - 1e-9)
    for k in _present(dists):
        d = dists[k]
        count = d.count_at_or_above()
        m = keep & (count > 0)
        style = _series_style(k)
        style["linewidth"] *= lw_scale
        label = SERIES_LABEL[k]
        if legend_stats:
            label = (f"{label}: wet {100 * d.fraction_at_or_above(WET_THRESHOLD_MM):.0f} %, "
                     f"99.9th percentile {_mm_text(d.quantile(99.9))} mm, "
                     f"maximum {_mm_text(d.max)} mm")
        ax.plot(edges[m], count[m] / d.n, label=label, **style)
    ax.set_xscale("log")
    ax.set_yscale("log")


def _draw_frequency_bias(ax, dists: dict, *, lw_scale: float = 1.0) -> tuple[float, float]:
    """Series / truth exceedance frequency, where the truth has enough values."""
    truth = dists["truth"]
    edges = StreamingDist.EDGES[:-1]
    t_count = truth.count_at_or_above()
    m = (edges >= WET_THRESHOLD_MM * (1 - 1e-9)) & (t_count >= MIN_TRUTH_COUNT)
    ax.axhline(1.0, color="0.25", linewidth=0.9, zorder=1)
    ratios = []
    for k in ("input", "pred"):
        d = dists[k]
        if d.empty or truth.empty or not m.any():
            continue
        r = (d.exceedance()[m] / truth.exceedance()[m])
        style = _series_style(k)
        style["linewidth"] *= lw_scale
        ax.plot(edges[m], r, **style)
        ratios.append(r[r > 0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    if ratios:
        r = np.concatenate(ratios)
        span = float(np.nanmax(np.abs(np.log2(r)))) if r.size else 1.0
    else:
        span = 1.0
    span = min(max(span * 1.1, np.log2(1.5)), 3.0)  # between 1.5x and 8x either way
    lo, hi = 2.0 ** -span, 2.0 ** span
    ax.set_ylim(lo, hi)
    ticks = [t for t in (0.125, 0.25, 0.5, 1, 2, 4, 8) if lo <= t <= hi]
    if len(ticks) < 3:
        ticks = [t for t in (0.5, 0.67, 0.8, 1, 1.25, 1.5, 2) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return lo, hi


def _x_limits(dists: dict) -> tuple[float, float]:
    top = max(d.max for d in dists.values() if not d.empty)
    return WET_THRESHOLD_MM, max(top, 1.0) * 1.3


def _mm_ticks(ax, xmax: float) -> None:
    ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000) if t <= xmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _cases_text(step_data: dict, info: dict) -> str:
    steps = sorted(step_data)
    dates = info.get("dates") or []
    lead = (f"lead times {steps[0]} to {steps[-1]} h" if len(steps) > 1
            else f"lead time {steps[0]} h")
    n_cases = len(dates) * len(steps) if dates else None
    mem = info.get("member_id")
    parts = []
    if n_cases:
        parts.append(f"{len(dates)} dates × {len(steps)} lead times ({lead}) = {n_cases} cases")
    else:
        parts.append(lead)
    if mem is not None:
        parts.append(f"member {mem}")
    return ", ".join(parts)


def _support_text(dists: dict) -> str:
    ns = {k: dists[k].n for k in _present(dists)}
    if len(set(ns.values())) == 1:
        return (f"each series {_count_text(next(iter(ns.values())))} grid-point values "
                "on the same grid points")
    return "UNEQUAL supports: " + ", ".join(
        f"{SERIES_LABEL[k].lower()} {_count_text(n)}" for k, n in ns.items())


def _footer(info: dict) -> str:
    return textwrap.fill(
        f"Truth: {info.get('truth_source', 'unknown')}. Input: {info.get('input_source', 'unknown')}. "
        f"Frequency bias drawn where the truth has at least {MIN_TRUTH_COUNT} values at or above "
        f"the threshold. Values below 0 count as dry.", width=150)


def distribution_page(pdf, step_data: dict, run_label: str, info: dict | None = None):
    """Main figure: exceedance frequency and frequency bias, all lead times pooled."""
    from eval.plotting import eval_style

    info = info or {}
    dists = {k: _overall(step_data, k) for k in SERIES}
    if dists["pred"].empty:
        return
    xlo, xhi = _x_limits(dists)
    with eval_style():
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(9.0, 7.4), sharex=True, constrained_layout=True,
            gridspec_kw={"height_ratios": [3.0, 1.35]})
        head = f"{run_label}: " if run_label else ""
        fig.suptitle(f"{head}distribution of 6 h precipitation\n"
                     f"{_cases_text(step_data, info)}; {_support_text(dists)}",
                     fontsize=11.5)
        _draw_exceedance(ax, dists, legend_stats=True)
        ax.set_ylabel("Fraction of grid points\nat or above the threshold")
        ax.set_title("Exceedance frequency", loc="left")
        ax.legend(loc="lower left", fontsize=8.5)
        n_min = min(dists[k].n for k in _present(dists))
        ax.set_ylim(0.5 / n_min, 1.0)
        if not dists["truth"].empty:
            _draw_frequency_bias(axr, dists)
            axr.set_ylabel("Frequency bias\n(series / truth)")
            axr.set_title("Frequency bias: above 1 means the threshold is exceeded "
                          "more often than in the truth", loc="left", fontsize=9.5)
        else:
            axr.set_visible(False)
        axr.set_xlim(xlo, xhi)
        _mm_ticks(axr, xhi)
        axr.set_xlabel(THRESHOLD_LABEL)
        fig.text(0.5, -0.01, _footer(info), ha="center", va="top", fontsize=7.5, color="0.35")
        _emit(pdf, fig, "distribution")


def _lead_steps(step_data: dict) -> list[int]:
    """Every lead time when there are few; else the highlight steps or an even subset."""
    steps = sorted(step_data)
    if len(steps) <= MAX_LEAD_PANELS:
        return steps
    chosen = [s for s in HIGHLIGHT_STEPS if s in step_data]
    if len(chosen) >= 2:
        return chosen
    idx = np.unique(np.linspace(0, len(steps) - 1, MAX_LEAD_PANELS).round().astype(int))
    return [steps[i] for i in idx]


def by_lead_time_page(pdf, step_data: dict, run_label: str, info: dict | None = None):
    """Compact figure: one small column per lead time, same two rows as the main figure."""
    from eval.plotting import eval_style

    info = info or {}
    steps = _lead_steps(step_data)
    if len(steps) < 2:
        return
    all_dists = {k: _overall(step_data, k) for k in SERIES}
    xlo, xhi = _x_limits(all_dists)
    n = len(steps)
    with eval_style():
        fig, axes = plt.subplots(
            2, n, figsize=(2.75 * n + 0.8, 5.4), sharex=True, sharey="row",
            constrained_layout=True, gridspec_kw={"height_ratios": [2.2, 1.2]}, squeeze=False)
        head = f"{run_label}: " if run_label else ""
        dates = info.get("dates") or []
        per = f", {len(dates)} dates per lead time" if dates else ""
        fig.suptitle(f"{head}6 h precipitation distribution by lead time{per}", fontsize=11.5)
        y_min = 1.0
        lims = []
        for c, step in enumerate(steps):
            dists = step_data[step]
            ax, axr = axes[0, c], axes[1, c]
            _draw_exceedance(ax, dists, legend_stats=False, lw_scale=0.75)
            n_min = min(dists[k].n for k in _present(dists))
            y_min = min(y_min, 0.5 / n_min)
            ax.set_title(f"Lead time {step} h", fontsize=10)
            if not dists["truth"].empty:
                lims.append(_draw_frequency_bias(axr, dists, lw_scale=0.75))
            axr.set_xlabel("Threshold (mm)")
        axes[0, 0].set_ylim(y_min, 1.0)
        if lims:
            lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
            axes[1, 0].set_ylim(lo, hi)
        axes[0, 0].set_ylabel("Fraction at or\nabove the threshold")
        axes[1, 0].set_ylabel("Frequency bias\n(series / truth)")
        axes[1, 0].set_xlim(xlo, xhi)
        for axr in axes[1]:
            _mm_ticks(axr, xhi)
            axr.tick_params(axis="x", labelrotation=0, labelsize=8)
        handles = [plt.Line2D([], [], **_series_style(k)) for k in _present(all_dists)]
        fig.legend(handles, [SERIES_LABEL[k] for k in _present(all_dists)],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=9)
        fig.text(0.5, -0.09, _footer(info), ha="center", va="top", fontsize=7.5, color="0.35")
        _emit(pdf, fig, "by_lead_time")


def render(out_pdf: Path, step_data: dict, run_label: str, info: dict) -> list[Path]:
    from eval.plotting import FigureBook

    with FigureBook(out_pdf, png=True) as book:
        distribution_page(book, step_data, run_label, info)
        by_lead_time_page(book, step_data, run_label, info)
    return book.paths


def main():
    parser = argparse.ArgumentParser(
        description="TP distribution comparison: input vs truth vs prediction")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="")
    parser.add_argument("--ensemble-member-index", type=int, default=0)
    parser.add_argument("--style", choices=("compact", "diagnostic"), default="compact",
                        help="Accepted for old callers; both values write the same two pages.")
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
    info: dict = {}
    step_data = accumulate_tp_by_step(
        args.predictions_dir,
        ensemble_member_index=args.ensemble_member_index,
        var=args.var,
        truth_grib_tpl=args.truth_grib_tpl,
        baseline_grib_tpl=args.baseline_grib_tpl,
        interp_index_cache=args.interp_index_cache,
        info=info,
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
    stem = args.out_pdf.with_suffix("")
    save_counts(stem.with_name(stem.name + "_counts.npz"), step_data, info)
    render(args.out_pdf, step_data, args.run_label, info)
    print(f"Done: {args.out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
