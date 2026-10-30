"""Calibration arithmetic: corrected spread/RMSE, rank histograms, random tie-breaking."""
from __future__ import annotations

import math

import numpy as np
import pytest

from eval.evaluators.probabilistic.core.calibration import (
    PooledMoments,
    correction_factor,
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
