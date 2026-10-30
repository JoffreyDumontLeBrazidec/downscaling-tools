"""Calibration arithmetic: corrected spread/RMSE, rank histograms, random tie-breaking."""
from __future__ import annotations

import math

import numpy as np
import pytest

from eval.evaluators.probabilistic.core.calibration import (
    RATIO_BOOTSTRAP_SEED,
    PooledMoments,
    correction_factor,
    dry_floor,
    dry_wet_disagreement,
    histogram_shape,
    rank_counts,
    spread_skill_ratio,
    summarize_histogram,
    tie_rng,
    truth_ranks,
)

M = 10
N = 200_000


def _ensemble(member_std: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Truth N(0, 1) per point; members N(0, member_std^2): calibrated when member_std = 1."""
    rng = np.random.default_rng(seed)
    truth = rng.standard_normal(N)
    members = rng.standard_normal((M, N)) * member_std
    return members, truth


def _pooled(members: np.ndarray, truth: np.ndarray, n_dates: int = 3) -> dict:
    pooled = PooledMoments()
    for k in range(n_dates):
        sl = slice(k * N // n_dates, (k + 1) * N // n_dates)
        f, y = members[:, sl], truth[sl]
        pooled.add_date(mean_variance=float(np.var(f, axis=0, ddof=1).mean()),
                        mean_sq_error=float(np.mean((f.mean(axis=0) - y) ** 2)),
                        n_points=y.size, n_members=M, fcrps=1.0, crps=1.1)
    return pooled.summary()


def test_correction_factor_for_ten_members():
    assert correction_factor(10) == pytest.approx(math.sqrt(1.1))
    ratio, corrected = spread_skill_ratio(1.0, 1.1, 10)   # var 1, error var (m+1)/m
    assert ratio == pytest.approx(math.sqrt(10 / 11), abs=1e-12)
    assert corrected == pytest.approx(1.0, abs=1e-12)


def test_calibrated_ensemble_reads_one_after_correction_and_flat_histogram():
    members, truth = _ensemble(1.0)
    s = _pooled(members, truth)
    assert s["spread_skill_ratio"] == pytest.approx(0.953, abs=0.02)
    assert abs(s["spread_skill_ratio_corrected"] - 1.0) <= 0.02
    assert s["n_dates"] == 3 and s["n_members"] == M and s["n_points_total"] == N
    counts = rank_counts(truth_ranks(members, truth, tie_rng("20230101", 24, "2t")), M)
    assert counts.size == M + 1 and counts.sum() == N
    h = summarize_histogram(counts)
    assert h["flatness_rms"] < 0.05
    assert h["outer_two_fraction"] == pytest.approx(2 / 11, abs=0.01)
    assert histogram_shape(h) == "flat"


def test_underdispersed_ensemble_reads_below_one_and_u_shape():
    members, truth = _ensemble(0.5)
    s = _pooled(members, truth)
    assert s["spread_skill_ratio_corrected"] < 0.9
    h = summarize_histogram(rank_counts(truth_ranks(members, truth, tie_rng("20230101", 24, "2t")), M))
    assert h["outer_two_fraction"] > 2 / 11 * 1.5
    assert histogram_shape(h) == "U-shaped (under-dispersed)"


def test_overdispersed_ensemble_reads_above_one_and_dome():
    members, truth = _ensemble(2.0)
    s = _pooled(members, truth)
    assert s["spread_skill_ratio_corrected"] > 1.1
    h = summarize_histogram(rank_counts(truth_ranks(members, truth, tie_rng("20230101", 24, "2t")), M))
    assert h["outer_two_fraction"] < 2 / 11 * 0.7
    assert histogram_shape(h) == "dome (over-dispersed)"


def test_zero_precipitation_ties_do_not_pile_into_the_lowest_bin():
    # dry truth and dry members everywhere: every rank is a full tie
    members = np.zeros((M, N))
    truth = np.zeros(N)
    counts = rank_counts(truth_ranks(members, truth, tie_rng("20230101", 24, "tp")), M)
    freq = counts / counts.sum()
    assert freq[0] < 0.12 and np.all(np.abs(freq - 1 / 11) < 0.01)
    # partial ties: truth 0, four members 0, six members wet; the rank is uniform on 0..4
    members = np.vstack([np.zeros((4, N)), np.full((6, N), 0.5)])
    counts = rank_counts(truth_ranks(members, truth, tie_rng("20230101", 24, "tp")), M)
    assert counts[5:].sum() == 0
    assert np.all(np.abs(counts[:5] / N - 0.2) < 0.01)


def test_tie_breaking_is_reproducible_and_order_independent():
    members = np.zeros((M, 1000))
    truth = np.zeros(1000)
    a = truth_ranks(members, truth, tie_rng("20230101", 24, "tp"))
    b = truth_ranks(members, truth, tie_rng("20230101", 24, "tp"))
    c = truth_ranks(members, truth, tie_rng("20230102", 24, "tp"))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_invalid_points_are_not_counted_and_masks_apply():
    members = np.random.default_rng(1).standard_normal((M, 100))
    truth = np.random.default_rng(2).standard_normal(100)
    truth[:10] = np.nan
    members[0, 10:20] = np.nan
    ranks = truth_ranks(members, truth, tie_rng("20230101", 24, "2t"))
    assert np.all(ranks[:20] == -1) and np.all(ranks[20:] >= 0)
    mask = np.zeros(100, dtype=bool)
    mask[50:] = True
    assert rank_counts(ranks, M, mask).sum() == 50
    assert rank_counts(ranks, M).sum() == 80


# ------------------------------------------------------------ rule 32 v2: ratio interval, dry floor, shape

def _pooled_dates(n_dates: int, seed: int = 0, member_std: float = 1.0, points: int = 2000) -> PooledMoments:
    """``n_dates`` calibrated (or not) dates of ``points`` points each, pooled."""
    rng = np.random.default_rng(seed)
    pooled = PooledMoments()
    for _ in range(n_dates):
        truth = rng.standard_normal(points)
        members = rng.standard_normal((M, points)) * member_std
        pooled.add_date(mean_variance=float(np.var(members, axis=0, ddof=1).mean()),
                        mean_sq_error=float(np.mean((members.mean(axis=0) - truth) ** 2)),
                        n_points=points, n_members=M, fcrps=1.0, crps=1.1)
    return pooled


def test_bootstrap_interval_contains_the_point_and_shrinks_with_more_dates():
    widths = []
    for n_dates in (5, 30, 120):
        s = _pooled_dates(n_dates).summary()
        lo, hi, se = (s["spread_skill_ratio_corrected_lo95"], s["spread_skill_ratio_corrected_hi95"],
                      s["spread_skill_ratio_corrected_se"])
        assert lo <= s["spread_skill_ratio_corrected"] <= hi
        assert 0.0 < se < hi - lo
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]
    # about 1/sqrt(n): 24 times the dates, about a fifth of the width (loose: the dates are few)
    assert widths[2] < widths[0] / 2.5
    # a calibrated ensemble: the interval covers 1 (sampling noise of 120 dates is far below 0.05)
    assert widths[2] < 0.05


def test_bootstrap_is_fixed_by_its_seed_and_needs_two_dates():
    pooled = _pooled_dates(8)
    a = pooled.bootstrap_ratio()
    b = pooled.bootstrap_ratio()
    c = pooled.bootstrap_ratio(seed=RATIO_BOOTSTRAP_SEED + 1)
    assert a == b and a != c
    assert pooled.bootstrap_ratio(n_resamples=200)["lo"] != a["lo"]
    one = _pooled_dates(1)
    s = one.summary()
    assert math.isnan(s["spread_skill_ratio_corrected_lo95"]) and math.isnan(s["spread_skill_ratio_corrected_se"])
    assert math.isfinite(s["spread_skill_ratio_corrected"])
    # an under-dispersed ensemble: the whole interval is below 0.95
    s = _pooled_dates(30, member_std=0.5).summary()
    assert s["spread_skill_ratio_corrected_hi95"] < 0.95


def test_dry_floor_makes_trace_members_over_a_dry_truth_a_tie():
    thr = 0.1 / 1000.0                                   # 0.1 mm in the metres of the files
    truth = np.zeros(N)
    trace = np.full((M, N), 0.00003)                     # 0.03 mm: drizzle under the threshold
    # without the floor every member is above the truth: the whole histogram sits in bin 0
    raw = rank_counts(truth_ranks(trace, truth, tie_rng("20230101", 24, "tp")), M)
    assert raw[0] == N
    # with the floor the members are exact zeros: a full tie, broken at random over all bins
    floored = rank_counts(truth_ranks(dry_floor(trace, thr), dry_floor(truth, thr), tie_rng("20230101", 24, "tp")), M)
    assert np.all(np.abs(floored / N - 1 / 11) < 0.01)
    # amounts at or above the threshold are untouched; NaN stays NaN; a zero threshold is the identity
    x = np.array([0.0, 0.00005, 0.0001, 0.002, np.nan])
    np.testing.assert_array_equal(dry_floor(x, thr)[:4], [0.0, 0.0, 0.0001, 0.002])
    assert np.isnan(dry_floor(x, thr)[4])
    np.testing.assert_array_equal(dry_floor(x, 0.0)[:4], x[:4])


def test_false_drizzle_and_missed_rain_rates_on_a_hand_built_case():
    thr = 1e-4
    # 10 points: 0-3 truth dry; members all wet at 0-1 (false drizzle), one dry member at 2, all dry at 3;
    # 4-7 truth wet: all members dry at 4-5 (missed rain), one wet member at 6, all wet at 7;
    # 8 NaN truth (not valid), 9 one NaN member (not valid)
    truth = np.array([0.0, 0.00005, 0.0, 0.0, 0.001, 0.0002, 0.003, 0.001, np.nan, 0.0])
    members = np.full((3, 10), 0.0005)
    members[:, 2] = [0.0005, 0.0, 0.0005]
    members[:, 3] = 0.0
    members[:, 4:6] = 0.00009
    members[:, 6] = [0.0, 0.0, 0.002]
    members[0, 9] = np.nan
    d = dry_wet_disagreement(members, truth, thr)
    assert d["valid"].sum() == 8
    np.testing.assert_array_equal(np.flatnonzero(d["false_drizzle"]), [0, 1])
    np.testing.assert_array_equal(np.flatnonzero(d["missed_rain"]), [4, 5])
    # the rates the evaluator writes: counts over the valid points
    assert d["false_drizzle"].sum() / d["valid"].sum() == 0.25
    assert d["missed_rain"].sum() / d["valid"].sum() == 0.25


def test_one_sided_pile_is_biased_not_u_shaped():
    flat = np.full(M + 1, 1000)
    both = flat.copy()
    both[0] = both[-1] = 3000
    low_pile = flat.copy()
    low_pile[0] = 5000                 # truth below every member too often: the members sit too high
    high_pile = flat.copy()
    high_pile[-1] = 5000               # truth above every member too often: the members sit too low
    assert histogram_shape(summarize_histogram(both)) == "U-shaped (under-dispersed)"
    assert histogram_shape(summarize_histogram(low_pile)).startswith("biased high")
    assert histogram_shape(summarize_histogram(high_pile)).startswith("biased low")
    assert "U-shaped" not in histogram_shape(summarize_histogram(low_pile))
    assert histogram_shape(summarize_histogram(flat)) == "flat"
    s = summarize_histogram(low_pile)
    assert s["bin_0_fraction"] == pytest.approx(5000 / 15000) and s["bin_m_fraction"] == pytest.approx(1000 / 15000)
    assert s["bin_expected"] == pytest.approx(1 / 11) and s["outer_two_fraction"] == pytest.approx(6000 / 15000)
    # an older summary without the separate fractions still gets a word
    assert histogram_shape({"outer_two_fraction": 0.4, "outer_two_expected": 2 / 11, "rank_mean_norm": 0.0}) \
        == "U-shaped (under-dispersed)"
