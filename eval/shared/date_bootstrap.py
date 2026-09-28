"""Bootstrap over initial dates (cluster bootstrap), vectorised.

Members and leads of the same initial date are correlated, so every uncertainty is
computed by resampling whole dates with replacement: a resample draws n_dates
dates from the n_dates available, and every value of a drawn date enters (as many
times as the date was drawn). Implemented with a multiplicity matrix M
(n_boot x n_dates): the resampled mean is (M @ per-date sums) / (M @ per-date
counts), so thousands of resamples cost one matrix product.

All functions ignore NaN values (a missing measurement is not a zero) and report
the number of values and dates that entered. Several groups (for example two arms,
or the model and the truth) are resampled with the SAME dates, which is what a
difference between them needs.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

DEFAULT_N_BOOT = 4000
DEFAULT_SEED = 20260923


def multiplicity(n_dates: int, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_dates, size=(n_boot, n_dates))
    M = np.zeros((n_boot, n_dates), dtype=np.float64)
    rows = np.repeat(np.arange(n_boot), n_dates)
    np.add.at(M, (rows, draws.ravel()), 1.0)
    return M


def _per_date(dates_all: Sequence, dates, values, powers=(0, 1)):
    """Per-date sums of values**p for p in powers, over the date list dates_all."""
    dates = np.asarray(dates)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    pos = {d: i for i, d in enumerate(dates_all)}
    idx = np.array([pos[d] for d in dates[ok]], dtype=np.int64)
    out = []
    for p in powers:
        w = values[ok] ** p if p else np.ones(int(ok.sum()))
        out.append(np.bincount(idx, weights=w, minlength=len(dates_all)))
    return out


def _summary(point, boots, n, n_dates, ci=0.95):
    b = boots[np.isfinite(boots)]
    if b.size < 2 or not np.isfinite(point):
        return {"value": float(point) if np.isfinite(point) else None, "se": None,
                "ci_lo": None, "ci_hi": None, "n": int(n), "n_dates": int(n_dates),
                "n_boot_valid": int(b.size)}
    lo, hi = np.percentile(b, [50 * (1 - ci), 100 - 50 * (1 - ci)])
    return {"value": float(point), "se": float(np.std(b, ddof=1)), "ci_lo": float(lo),
            "ci_hi": float(hi), "n": int(n), "n_dates": int(n_dates),
            "n_boot_valid": int(b.size)}


def boot_mean(dates, values, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
              ci: float = 0.95) -> dict:
    """Mean of the non-NaN values with its date-clustered standard error and interval."""
    dates_all = sorted(set(np.asarray(dates)[np.isfinite(np.asarray(values, dtype=float))].tolist()))
    if not dates_all:
        return _summary(np.nan, np.array([]), 0, 0, ci)
    cnt, s1 = _per_date(dates_all, dates, values)
    point = s1.sum() / cnt.sum()
    M = multiplicity(len(dates_all), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (M @ s1) / (M @ cnt)
    return _summary(point, boots, cnt.sum(), len(dates_all), ci)


def boot_mean_diff(dates_a, values_a, dates_b, values_b, n_boot: int = DEFAULT_N_BOOT,
                   seed: int = DEFAULT_SEED, ci: float = 0.95) -> dict:
    """mean(a) - mean(b), both groups resampled with the same dates.

    The two groups need not be paired member by member (the truth is not paired
    with the model); only the dates are shared.
    """
    va = np.asarray(values_a, dtype=float)
    vb = np.asarray(values_b, dtype=float)
    dates_all = sorted(set(np.asarray(dates_a)[np.isfinite(va)].tolist())
                       | set(np.asarray(dates_b)[np.isfinite(vb)].tolist()))
    if not dates_all:
        return _summary(np.nan, np.array([]), 0, 0, ci)
    ca, sa = _per_date(dates_all, dates_a, va)
    cb, sb = _per_date(dates_all, dates_b, vb)
    if ca.sum() == 0 or cb.sum() == 0:
        return _summary(np.nan, np.array([]), 0, len(dates_all), ci)
    point = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    M = multiplicity(len(dates_all), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (M @ sa) / (M @ ca) - (M @ sb) / (M @ cb)
    out = _summary(point, boots, ca.sum() + cb.sum(), len(dates_all), ci)
    out["n_a"] = int(ca.sum())
    out["n_b"] = int(cb.sum())
    return out


def boot_slope(dates, x, y, n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
               ci: float = 0.95) -> dict:
    """Least-squares slope of y on x and the scatter (residual standard deviation)
    about that line, each with date-clustered standard errors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = np.asarray(dates)[ok]
    x, y = x[ok], y[ok]
    dates_all = sorted(set(d.tolist()))
    empty = {"slope": _summary(np.nan, np.array([]), 0, 0, ci),
             "scatter": _summary(np.nan, np.array([]), 0, 0, ci)}
    if len(x) < 3:
        return empty
    pos = {dd: i for i, dd in enumerate(dates_all)}
    idx = np.array([pos[dd] for dd in d])
    nd = len(dates_all)
    S = np.stack([np.bincount(idx, weights=w, minlength=nd)
                  for w in (np.ones_like(x), x, y, x * x, x * y, y * y)], axis=1)

    def fit(T):
        n, sx, sy, sxx, sxy, syy = (T[..., k] for k in range(6))
        with np.errstate(invalid="ignore", divide="ignore"):
            cxx = sxx - sx * sx / n
            cxy = sxy - sx * sy / n
            cyy = syy - sy * sy / n
            slope = cxy / cxx
            ssr = np.maximum(cyy - slope * cxy, 0.0)
            scatter = np.sqrt(ssr / (n - 2))
        return slope, scatter

    s0, sc0 = fit(S.sum(axis=0))
    M = multiplicity(nd, n_boot, seed)
    sb, scb = fit(M @ S)
    return {"slope": _summary(float(s0), sb, len(x), nd, ci),
            "scatter": _summary(float(sc0), scb, len(x), nd, ci)}


def date_key(values: Iterable) -> list:
    return [str(v) for v in values]
