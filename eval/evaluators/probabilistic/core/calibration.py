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

Uncertainty of the ratio (rule 32 v2)
-------------------------------------
The corrected ratio comes with a 95 % bootstrap interval over dates
(``spread_skill_ratio_corrected_lo95`` / ``_hi95``) and a bootstrap standard error
(``_se``): the dates are resampled with replacement ``RATIO_BOOTSTRAP_RESAMPLES`` (2000)
times with the fixed seed ``RATIO_BOOTSTRAP_SEED``, and the ratio is recomputed from the
per-date mean squares exactly as the point estimate is (each date counts once, roots after
pooling, then the correction). Fewer than two dates give no interval (NaN), and the verdict
then falls back to the point estimate. The verdict reads the interval three ways
(``verdict.py``): fully inside [0.95, 1.05] is calibrated, fully outside is not, and an
interval that crosses 0.95 or 1.05 is "undecided".

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

Precipitation gets a dry floor before ranking (``dry_floor``): members and truth below
``rank_dry_threshold_mm`` (0.1 mm, the WMO trace threshold) are set to exactly 0, so a
member at 0.03 mm over a dry truth is a tie (broken at random as above) and not "above the
truth". The evaluator works in the native unit of the files, metres of accumulated water
(``eval/plotting/variables.py``), so 0.1 mm is 1e-4 m; ``scoring.py`` makes the conversion
(``DRY_THRESHOLD_MM_TO_NATIVE``). The same threshold defines two disagreement rates per
(lead, variable, region) (``dry_wet_disagreement``): the false-drizzle rate, points where
the truth is dry and EVERY member is wet, and the missed-rain rate, truth wet and every
member dry. They are fractions of the valid points, written to ``rank_histograms.csv``.

Numbers that summarise a histogram (``summarize_histogram``): the fractions of the counts in
the lowest bin (``bin_0_fraction``, truth below every member) and in the highest bin
(``bin_m_fraction``, truth above every member), each against the flat expectation
``1/(m+1)``, and their sum ``outer_two_fraction`` against ``2/(m+1)``; ``flatness_rms``, the
root mean square of the relative deviation of the bin frequencies from flat (0 for a flat
histogram; 0.3 means the bins deviate by 30 % on average); ``rank_mean_norm`` (mean rank
over ``m`` minus 0.5), positive when the truth tends to sit above the members (the ensemble
is biased low). ``histogram_shape`` turns them into a word: "U-shaped" needs BOTH outer bins
high; a pile in one outer bin only is "biased high" (bin 0: the members sit above the
truth) or "biased low" (bin m), never a U.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Fixed seed of the tie-breaking draw of the rank histograms (see module docstring).
RANK_TIE_SEED = 31
# Bootstrap of the corrected spread/RMSE over dates: resamples and fixed seed (see module docstring).
RATIO_BOOTSTRAP_RESAMPLES = 2000
RATIO_BOOTSTRAP_SEED = 32
RATIO_CI_LEVEL = 0.95
# Dry threshold of the rank histograms of precipitation, in mm (0.1 mm: the trace threshold).
DEFAULT_RANK_DRY_THRESHOLD_MM = 0.1


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
    variances: list[float] = field(default_factory=list)
    sq_errors: list[float] = field(default_factory=list)
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
        self.variances.append(float(mean_variance))
        self.sq_errors.append(float(mean_sq_error))
        if math.isfinite(fcrps):
            self.fcrps.append(float(fcrps))
        if math.isfinite(crps):
            self.crps.append(float(crps))
        if input_variance is not None and math.isfinite(input_variance):
            self.sum_input_variance += float(input_variance)
            self.n_dates_input += 1
            self.input_members = int(input_members)

    def bootstrap_ratio(self, *, n_resamples: int = RATIO_BOOTSTRAP_RESAMPLES,
                        seed: int = RATIO_BOOTSTRAP_SEED, level: float = RATIO_CI_LEVEL) -> dict[str, float]:
        """``(lo, hi, se)`` of the corrected ratio from dates resampled with replacement.

        The resampled ratio pools the per-date mean squares exactly as ``summary`` does (each
        date once, roots after pooling, then ``sqrt((m+1)/m)``), so the interval is around
        the point estimate; the draw is fixed by ``seed``. NaN with fewer than two dates.
        """
        var = np.asarray(self.variances, dtype=np.float64)
        sq = np.asarray(self.sq_errors, dtype=np.float64)
        n = var.size
        if n < 2 or n_resamples < 2:
            return {"lo": math.nan, "hi": math.nan, "se": math.nan}
        rng = np.random.default_rng(int(seed))
        idx = rng.integers(0, n, size=(int(n_resamples), n))
        mean_var = var[idx].mean(axis=1)
        mean_sq = sq[idx].mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = np.sqrt(np.maximum(mean_var, 0.0) / mean_sq) * correction_factor(self.n_members)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size < 2:
            return {"lo": math.nan, "hi": math.nan, "se": math.nan}
        alpha = (1.0 - float(level)) / 2.0
        lo, hi = np.quantile(ratios, [alpha, 1.0 - alpha])
        return {"lo": float(lo), "hi": float(hi), "se": float(np.std(ratios, ddof=1))}

    def summary(self, *, bootstrap_resamples: int = RATIO_BOOTSTRAP_RESAMPLES,
                bootstrap_seed: int = RATIO_BOOTSTRAP_SEED) -> dict[str, Any]:
        n = self.n_dates
        if n == 0:
            return {}
        mean_var = self.sum_variance / n
        mean_sq = self.sum_sq_error / n
        ratio, corrected = spread_skill_ratio(mean_var, mean_sq, self.n_members)
        ci = self.bootstrap_ratio(n_resamples=bootstrap_resamples, seed=bootstrap_seed)
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
            "spread_skill_ratio_corrected_lo95": ci["lo"],
            "spread_skill_ratio_corrected_hi95": ci["hi"],
            "spread_skill_ratio_corrected_se": ci["se"],
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


def dry_floor(values: np.ndarray, threshold: float) -> np.ndarray:
    """Values below ``threshold`` (same unit as the values) set to exactly 0; NaN stays NaN.

    Applied to precipitation members and truth before ranking, so trace amounts under the
    dry threshold are ties with a dry truth and not ranks (see the module docstring).
    """
    values = np.asarray(values, dtype=np.float64)
    if not (math.isfinite(threshold) and threshold > 0.0):
        return values
    return np.where(np.isfinite(values) & (values < threshold), 0.0, values)


def dry_wet_disagreement(forecasts: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """Per point: ``false_drizzle`` (truth dry, every member wet), ``missed_rain`` (truth wet,
    every member dry) and ``valid`` (finite truth and members); wet is ``>= threshold``."""
    forecasts = np.asarray(forecasts, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    valid = np.isfinite(truth) & np.all(np.isfinite(forecasts), axis=0)
    wet = forecasts >= threshold
    truth_wet = truth >= threshold
    return {
        "false_drizzle": valid & ~truth_wet & np.all(wet, axis=0),
        "missed_rain": valid & truth_wet & np.all(~wet, axis=0),
        "valid": valid,
    }


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
        return {"bin_0_fraction": math.nan, "bin_m_fraction": math.nan,
                "bin_expected": 1.0 / n_bins if n_bins else math.nan,
                "outer_two_fraction": math.nan, "outer_two_expected": 2.0 / n_bins if n_bins else math.nan,
                "flatness_rms": math.nan, "rank_mean_norm": math.nan, "chi2_per_bin": math.nan}
    freq = counts / total
    flat = 1.0 / n_bins
    ranks = np.arange(n_bins, dtype=np.float64)
    return {
        "bin_0_fraction": float(freq[0]),
        "bin_m_fraction": float(freq[-1]),
        "bin_expected": flat,
        "outer_two_fraction": float(freq[0] + freq[-1]),
        "outer_two_expected": 2.0 / n_bins,
        "flatness_rms": float(np.sqrt(np.mean((freq / flat - 1.0) ** 2))),
        "rank_mean_norm": float(np.sum(freq * ranks) / (n_bins - 1) - 0.5),
        # Pearson chi-square of the counts against flat, per bin (grows with the count)
        "chi2_per_bin": float(np.sum((counts - total * flat) ** 2 / (total * flat)) / n_bins),
    }


def histogram_shape(summary: dict[str, float], *, tolerance: float = 0.10) -> str:
    """A word for the read: flat, U-shaped (under-dispersed), dome (over-dispersed), biased, sloped.

    An outer bin is "high" above ``(1 + 2 tolerance)`` times its flat expectation. Both outer
    bins high is the U of an under-dispersed ensemble; ONE outer bin high is a one-sided pile
    and reads "biased high" (bin 0: the truth sits below every member too often) or "biased
    low" (bin m), not a U. Then a tilted histogram is "sloped (biased)", a low outer pair a
    dome, the rest flat.
    """
    outer, expected = summary.get("outer_two_fraction", math.nan), summary.get("outer_two_expected", math.nan)
    if not (math.isfinite(outer) and math.isfinite(expected)):
        return "n/a"
    f0, fm = summary.get("bin_0_fraction", math.nan), summary.get("bin_m_fraction", math.nan)
    if not (math.isfinite(f0) and math.isfinite(fm)):
        # an older summary without the separate fractions: split the pair evenly
        f0 = fm = outer / 2.0
    high = (expected / 2.0) * (1.0 + 2.0 * tolerance)
    if f0 > high and fm > high:
        return "U-shaped (under-dispersed)"
    if f0 > high:
        return "biased high (pile below all members)"
    if fm > high:
        return "biased low (pile above all members)"
    if abs(summary.get("rank_mean_norm", 0.0)) > tolerance:
        return "sloped (biased)"
    if outer < expected * (1.0 - 2.0 * tolerance):
        return "dome (over-dispersed)"
    return "flat"
