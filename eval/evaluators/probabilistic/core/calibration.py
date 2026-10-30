"""Calibration of an ensemble: pooled spread against error, and rank histograms.

The owner's rule for the scores (30 Oct 2026): report fair CRPS together with the
ensemble-mean RMSE, the spread, the spread/RMSE ratio with the ``sqrt((m+1)/m)``
correction, and rank histograms with ``m+1`` bins; break all of these down by lead and
region. This module holds the arithmetic; ``scoring.py`` feeds it from the files.

Spread and RMSE as roots of pooled mean squares
-----------------------------------------------
For a calibrated ``m``-member ensemble the expected squared error of the ensemble mean
is ``(m+1)/m`` times the expected ensemble variance (ddof 1), so the ratio of the two
roots reads ``sqrt(m/(m+1))`` (0.953 for ten members), and ``ratio * sqrt((m+1)/m)``
reads 1. Both roots are taken AFTER pooling the area-weighted mean squares over the
points of a region and over the dates (``PooledMoments``): a mean of per-date ratios would
not have this property, and a ratio of means of standard deviations (the old ``spread``
metric, a mean of pointwise standard deviations, over the mean of per-date RMSEs) is
biased low by Jensen. The old metrics are kept as they are; the new ones carry new names
(``spread_rms``, ``rmse_ens_mean`` pooled, ``spread_skill_ratio``,
``spread_skill_ratio_corrected``).

Rank histograms
---------------
The rank of the truth among the ``m`` members, 0 (below every member) to ``m`` (above every
member), ``m+1`` bins. Ties are broken uniformly at random, every tie, not only exact
zeros: this is what makes zero precipitation (truth 0 and several members 0) spread over
the bins it belongs to instead of piling into the lowest one. The draw uses
``numpy.random.default_rng`` seeded from ``RANK_TIE_SEED`` and the (date, lead,
variable) of the field (``tie_rng``), so two runs of the evaluator on the same files give
the same counts whatever the order of the files. Counts are plain point counts (a reduced
Gaussian grid is already close to equal-area; a region's counts are not area-weighted).

Two numbers summarise a histogram (``summarize_histogram``): the fraction of the counts in
the two outer bins against its flat expectation ``2/(m+1)`` (above: under-dispersed or
biased; below: over-dispersed), and ``flatness_rms``, the root mean square of the relative
deviation of the bin frequencies from flat (0 for a flat histogram; 0.3 means the bins
deviate by 30 % on average). ``rank_mean_norm`` (mean rank over ``m`` minus 0.5) is
positive when the truth tends to sit above the members (the ensemble is biased low).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Fixed seed of the tie-breaking draw of the rank histograms (see module docstring).
RANK_TIE_SEED = 31


def correction_factor(n_members: int) -> float:
    """``sqrt((m+1)/m)``: multiplies spread/RMSE so a calibrated ensemble reads 1."""
    m = int(n_members)
    if m < 1:
        return math.nan
    return math.sqrt((m + 1.0) / m)


def spread_skill_ratio(mean_variance: float, mean_sq_error: float, n_members: int) -> tuple[float, float]:
    """``(ratio, corrected ratio)`` from pooled mean ensemble variance and mean squared error."""
    if not (math.isfinite(mean_variance) and math.isfinite(mean_sq_error)) or mean_sq_error <= 0.0:
        return math.nan, math.nan
    ratio = math.sqrt(max(mean_variance, 0.0) / mean_sq_error)
    return ratio, ratio * correction_factor(n_members)


@dataclass
class PooledMoments:
    """Area-weighted mean squares of one (lead, region, variable, band), pooled over dates.

    Each date adds its own area-weighted mean ensemble variance (ddof 1) and mean squared
    error of the ensemble mean, so a date counts once whatever its number of valid points
    (the region's weights are renormalised per date). ``input_variance`` is the same for the
    low-resolution input members when the files carry them.
    """

    n_members: int = 0
    n_dates: int = 0
    n_points_total: int = 0
    sum_variance: float = 0.0
    sum_sq_error: float = 0.0
    sum_input_variance: float = 0.0
    n_dates_input: int = 0
    input_members: int = 0
    fcrps: list[float] = field(default_factory=list)
    crps: list[float] = field(default_factory=list)

    def add_date(self, *, mean_variance: float, mean_sq_error: float, n_points: int, n_members: int,
                 fcrps: float, crps: float, input_variance: float | None = None,
                 input_members: int = 0) -> None:
        if not (math.isfinite(mean_variance) and math.isfinite(mean_sq_error)):
            return
        self.n_members = int(n_members)
        self.n_dates += 1
        self.n_points_total += int(n_points)
        self.sum_variance += float(mean_variance)
        self.sum_sq_error += float(mean_sq_error)
        if math.isfinite(fcrps):
            self.fcrps.append(float(fcrps))
        if math.isfinite(crps):
            self.crps.append(float(crps))
        if input_variance is not None and math.isfinite(input_variance):
            self.sum_input_variance += float(input_variance)
            self.n_dates_input += 1
            self.input_members = int(input_members)

    def summary(self) -> dict[str, Any]:
        n = self.n_dates
        if n == 0:
            return {}
        mean_var = self.sum_variance / n
        mean_sq = self.sum_sq_error / n
        ratio, corrected = spread_skill_ratio(mean_var, mean_sq, self.n_members)
        out: dict[str, Any] = {
            "n_members": self.n_members,
            "n_dates": n,
            "n_points_total": self.n_points_total,
            "fcrps_mean": _mean(self.fcrps),
            "fcrps_stderr": _stderr(self.fcrps),
            "crps_mean": _mean(self.crps),
            "spread_rms": math.sqrt(max(mean_var, 0.0)),
            "rmse_ens_mean": math.sqrt(max(mean_sq, 0.0)),
            "spread_skill_ratio": ratio,
            "spread_skill_ratio_corrected": corrected,
            "correction_factor": correction_factor(self.n_members),
            "spread_input_rms": math.nan,
            "input_n_members": self.input_members,
            "spread_input_share": math.nan,
        }
        if self.n_dates_input:
            input_rms = math.sqrt(max(self.sum_input_variance / self.n_dates_input, 0.0))
            out["spread_input_rms"] = input_rms
            # input large-scale spread over the model's: how much of the band's spread comes
            # from the input members (above 1: the model has less spread than its input)
            out["spread_input_share"] = input_rms / out["spread_rms"] if out["spread_rms"] > 0 else math.nan
        return out


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else math.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


# ------------------------------------------------------------------ rank histograms

def tie_rng(date: str | int, step: int, variable: str, *, seed: int = RANK_TIE_SEED) -> np.random.Generator:
    """The tie-breaking generator of one field: fixed seed, independent of the file order."""
    var_code = sum(ord(c) * 31 ** i for i, c in enumerate(str(variable))) % (2 ** 31)
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(str(date)), int(step), var_code]))


def truth_ranks(forecasts: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank of the truth among the members per point, 0..m, ties broken uniformly at random.

    Points with a non-finite truth or member get rank -1 (not counted).
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    valid = np.isfinite(truth) & np.all(np.isfinite(forecasts), axis=0)
    below = np.sum(forecasts < truth[np.newaxis, :], axis=0)
    equal = np.sum(forecasts == truth[np.newaxis, :], axis=0)
    # a tie with k members: the truth takes any of the k+1 positions with equal chance
    rank = below + rng.integers(0, equal + 1)
    rank = np.where(valid, rank, -1)
    return rank.astype(np.int64)


def rank_counts(ranks: np.ndarray, n_members: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Histogram counts over ``m+1`` bins of the valid ranks (optionally inside ``mask``)."""
    r = ranks if mask is None else ranks[mask]
    r = r[r >= 0]
    return np.bincount(r, minlength=int(n_members) + 1).astype(np.int64)


def summarize_histogram(counts: np.ndarray) -> dict[str, float]:
    """Outer-two-bin fraction (and its flat expectation), rms flatness, normalised mean rank."""
    counts = np.asarray(counts, dtype=np.float64)
    n_bins = counts.size
    total = float(counts.sum())
    if n_bins < 2 or total <= 0.0:
        return {"outer_two_fraction": math.nan, "outer_two_expected": 2.0 / n_bins if n_bins else math.nan,
                "flatness_rms": math.nan, "rank_mean_norm": math.nan, "chi2_per_bin": math.nan}
    freq = counts / total
    flat = 1.0 / n_bins
    ranks = np.arange(n_bins, dtype=np.float64)
    return {
        "outer_two_fraction": float(freq[0] + freq[-1]),
        "outer_two_expected": 2.0 / n_bins,
        "flatness_rms": float(np.sqrt(np.mean((freq / flat - 1.0) ** 2))),
        "rank_mean_norm": float(np.sum(freq * ranks) / (n_bins - 1) - 0.5),
        # Pearson chi-square of the counts against flat, per bin (grows with the count)
        "chi2_per_bin": float(np.sum((counts - total * flat) ** 2 / (total * flat)) / n_bins),
    }


def histogram_shape(summary: dict[str, float], *, tolerance: float = 0.10) -> str:
    """A word for the read: flat, U-shaped (under-dispersed), dome (over-dispersed), sloped."""
    outer, expected = summary.get("outer_two_fraction", math.nan), summary.get("outer_two_expected", math.nan)
    if not (math.isfinite(outer) and math.isfinite(expected)):
        return "n/a"
    if abs(summary.get("rank_mean_norm", 0.0)) > tolerance:
        return "sloped (biased)"
    if outer > expected * (1.0 + 2.0 * tolerance):
        return "U-shaped (under-dispersed)"
    if outer < expected * (1.0 - 2.0 * tolerance):
        return "dome (over-dispersed)"
    return "flat"
