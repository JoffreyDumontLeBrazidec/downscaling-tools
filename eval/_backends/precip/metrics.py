"""Precipitation score primitives (all inputs/outputs in mm per 6h window).

Kept free of I/O so the numbers are unit-testable. Quantiles come from a
fixed-resolution histogram (0.02 mm bins) rather than a full sort: at o2560
scale a sort per member-slice would dominate the runtime, and 0.02 mm is far
below any decision threshold used on this lane.
"""
from __future__ import annotations

import numpy as np

# Histogram support for quantile estimation: 0..2048 mm at 0.02 mm.
HIST_MAX_MM = 2048.0
HIST_BIN_MM = 0.02
WET_THRESHOLD_MM = 0.1


def _hist_counts(vals_mm: np.ndarray) -> np.ndarray:
    edges = int(HIST_MAX_MM / HIST_BIN_MM)
    clipped = np.clip(vals_mm, 0.0, HIST_MAX_MM - HIST_BIN_MM / 2)
    return np.bincount((clipped / HIST_BIN_MM).astype(np.int64), minlength=edges)


def hist_quantiles(vals_mm: np.ndarray, qs) -> list[float]:
    """Approximate upper quantiles (in mm) from the fixed histogram.

    Negatives are clipped into the zero bin, which cannot disturb the upper
    tail. Accuracy is one bin width (0.02 mm).
    """
    counts = _hist_counts(vals_mm)
    cum = np.cumsum(counts)
    total = cum[-1]
    out = []
    for q in qs:
        target = q / 100.0 * total
        idx = int(np.searchsorted(cum, target, side="left"))
        out.append((idx + 0.5) * HIST_BIN_MM)
    return out


def field_stats(vals_mm: np.ndarray, *, wet_threshold_mm: float = WET_THRESHOLD_MM) -> dict:
    """Distribution stats of one field slice (finite values only)."""
    v = vals_mm[np.isfinite(vals_mm)]
    if v.size == 0:
        return {"n": 0}
    p99, p999 = hist_quantiles(v, (99.0, 99.9))
    return {
        "n": int(v.size),
        "mean_mm": float(v.mean()),
        "max_mm": float(v.max()),
        "p99_mm": p99,
        "p999_mm": p999,
        "wet_frac": float((v > wet_threshold_mm).mean()),
        "neg_frac": float((v < 0.0).mean()),
        "min_mm": float(v.min()),
    }


def pair_scores(pred_mm: np.ndarray, truth_mm: np.ndarray) -> dict:
    """Pointwise skill of one field against truth (both mm)."""
    m = np.isfinite(pred_mm) & np.isfinite(truth_mm)
    p, t = pred_mm[m].astype(np.float64), truth_mm[m].astype(np.float64)
    if p.size == 0:
        return {"n": 0}
    diff = p - t
    ps, ts = p.std(), t.std()
    corr = float(((p - p.mean()) * (t - t.mean())).mean() / (ps * ts)) \
        if ps > 0 and ts > 0 else float("nan")
    return {
        "n": int(p.size),
        "rmse_mm": float(np.sqrt(np.mean(diff * diff))),
        "mae_mm": float(np.mean(np.abs(diff))),
        "bias_mm": float(diff.mean()),
        "corr": corr,
    }


def nanmean(values) -> float:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))
