"""Statistics shared by every gate of ``eval.cli verdict`` (trusted evaluation suite, docs note
``epics/misc/20261119_trusted_evaluation_suite.md`` s4.0).

* Lead bands: +12..+72, +84..+120, +132..+180, +192..+240 h (per-lead lines are reported, not gated).
* Date blocks: two rules.
  - ``fixed5``: consecutive 5-day blocks counted from the first date (MI month sets: 30 dates = 6 blocks).
  - ``chain``: a chain of start dates 5 days or less apart is one block (XSET-1; the rule of ``xset_table.py``).
* SE = delete-one-block jackknife. Interval = value +- t(0.975, n-1) x SE. MDE = (t0.975 + t0.80) x SE.
* No scipy: Student t quantiles come from a table (df 1-30, then the normal value).
"""
from __future__ import annotations

import datetime as dt
from typing import Callable, Iterable

import numpy as np

LEAD_BANDS = ((12, 72), (84, 120), (132, 180), (192, 240))
BLOCK_DAYS = 5

# Student t quantiles, df = 1..30 (index 0 = df 1)
_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
         2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
_T80 = (1.376, 1.061, 0.978, 0.941, 0.920, 0.906, 0.896, 0.889, 0.883, 0.879, 0.876, 0.873, 0.870, 0.868, 0.866,
        0.865, 0.863, 0.862, 0.861, 0.860, 0.859, 0.858, 0.858, 0.857, 0.856, 0.856, 0.855, 0.855, 0.854, 0.854)


def t975(df: int) -> float:
    if df < 1:
        return float("nan")
    return _T975[df - 1] if df <= 30 else 1.960


def t80(df: int) -> float:
    if df < 1:
        return float("nan")
    return _T80[df - 1] if df <= 30 else 0.842


def mde(se: float, n_units: int) -> float:
    return (t975(n_units - 1) + t80(n_units - 1)) * se if np.isfinite(se) and n_units > 1 else float("nan")


def band_of(lead: int) -> str | None:
    for lo, hi in LEAD_BANDS:
        if lo <= int(lead) <= hi:
            return band_name(lo, hi)
    return None


def band_name(lo: int, hi: int) -> str:
    return f"+{lo}..+{hi} h"


def _day(d: str) -> dt.date:
    return dt.datetime.strptime(str(d)[:8], "%Y%m%d").date()


def blocks(dates: Iterable[str], rule: str = "fixed5") -> dict[str, str]:
    """date -> block label."""
    ds = sorted({str(d) for d in dates})
    if not ds:
        return {}
    if rule == "fixed5":
        d0 = _day(ds[0])
        return {d: f"b{(_day(d) - d0).days // BLOCK_DAYS:03d}" for d in ds}
    if rule == "chain":
        out, lab = {}, None
        for i, d in enumerate(ds):
            if i == 0 or (_day(d) - _day(ds[i - 1])).days > BLOCK_DAYS:
                lab = d
            out[d] = lab
        return out
    if rule == "date":
        return {d: d for d in ds}
    raise ValueError(f"unknown block rule {rule!r} (fixed5, chain, date)")


def jackknife(stat: Callable[[np.ndarray], float], n: int) -> tuple[float, float]:
    """stat(keep mask over n units) -> value. Returns (full value, delete-one SE); SE is nan below 3 units."""
    full = float(stat(np.ones(n, bool)))
    if n < 3:
        return full, float("nan")
    th = np.empty(n)
    for i in range(n):
        k = np.ones(n, bool)
        k[i] = False
        th[i] = stat(k)
    th = th[np.isfinite(th)]
    if len(th) < 3:
        return full, float("nan")
    return full, float(np.sqrt((n - 1) / n * np.sum((th - th.mean()) ** 2)))


def pct_ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """100 x (sum a / sum b - 1) over units, with its jackknife SE. a, b = per-unit sums (same units)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return jackknife(lambda k: 100.0 * (a[k].sum() / b[k].sum() - 1.0) if b[k].sum() > 0 else np.nan, len(a))


def interval(value: float, se: float, n_units: int) -> tuple[float, float]:
    t = t975(n_units - 1)
    if not (np.isfinite(se) and np.isfinite(t)):
        return float("nan"), float("nan")
    return value - t * se, value + t * se


# --------------------------------------------------------------------------------------------- states
PASS, FAIL, INCONCLUSIVE, MISSING = "PASS", "FAIL", "INCONCLUSIVE", "MISSING"


def state_not_worse(lo: float, hi: float, margin: float) -> str:
    """Non-inferiority gate (positive = worse). PASS: upper end of the 95 % interval below the margin. FAIL: lower end
    above the margin (worse than the margin beyond noise). Otherwise INCONCLUSIVE.

    Deviation from note s4.0 as first written ("FAIL = lower end above 0"), B1, 19 Nov 2026: with lower end above 0 a run
    identical to the reference fails each sub-line with probability 2.5 %, and the 14 hard gates hold about 60 sub-lines,
    so an identical run would FAIL about 4 times in 5 (seen in the synthetic tests). A significant degradation smaller
    than the margin is INCONCLUSIVE or PASS, never FAIL; the margin is the owner's tolerance."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return INCONCLUSIVE
    if hi < margin:
        return PASS
    if lo > margin:
        return FAIL
    return INCONCLUSIVE


def state_better(lo: float, hi: float, noise_ok: bool) -> str:
    """Gain gate (positive = worse, so better = interval below 0). PASS: upper end below 0. A gain inside noise is
    INCONCLUSIVE where ``noise_ok`` (the +192..+240 h band), FAIL elsewhere; worse beyond noise is always FAIL."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return INCONCLUSIVE
    if hi < 0:
        return PASS
    if lo > 0:
        return FAIL
    return INCONCLUSIVE if noise_ok else FAIL


def combine(states: Iterable[str]) -> str:
    """Gate state from its sub-lines: FAIL if any FAIL, PASS if all PASS, MISSING if none, else INCONCLUSIVE."""
    s = list(states)
    if not s or all(x == MISSING for x in s):
        return MISSING
    if FAIL in s:
        return FAIL
    if all(x == PASS for x in s):
        return PASS
    return INCONCLUSIVE
