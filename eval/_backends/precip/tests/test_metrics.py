"""Unit tests for precip metric primitives."""
from __future__ import annotations

import numpy as np

from eval._backends.precip import metrics as M


def test_hist_quantiles_match_percentile():
    # The histogram quantile is a step-function empirical quantile;
    # np.percentile interpolates between order statistics. In a heavy tail the
    # two differ by a few bins of sampling noise, so the contract is agreement
    # to ~0.5% of the value (still far below any decision threshold).
    rng = np.random.default_rng(7)
    vals = rng.gamma(0.5, 4.0, size=200_000)  # heavy-tailed like precip (mm)
    for q in (99.0, 99.9):
        approx = M.hist_quantiles(vals, [q])[0]
        exact = float(np.percentile(vals, q))
        assert abs(approx - exact) <= max(5 * M.HIST_BIN_MM, 0.005 * exact)


def test_pair_scores_exact_small():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    pred = truth + 1.0
    s = M.pair_scores(pred, truth)
    assert s["rmse_mm"] == 1.0
    assert s["bias_mm"] == 1.0
    assert abs(s["corr"] - 1.0) < 1e-12
    assert s["mae_mm"] == 1.0


def test_pair_scores_masks_nans():
    truth = np.array([0.0, np.nan, 2.0])
    pred = np.array([1.0, 5.0, np.nan])
    s = M.pair_scores(pred, truth)
    assert s["n"] == 1
    assert s["rmse_mm"] == 1.0


def test_field_stats_wet_and_negative_fractions():
    vals = np.array([-0.5, 0.0, 0.05, 0.2, 5.0])
    s = M.field_stats(vals, wet_threshold_mm=0.1)
    assert s["n"] == 5
    assert s["wet_frac"] == 2 / 5
    assert s["neg_frac"] == 1 / 5
    assert s["max_mm"] == 5.0
    assert s["min_mm"] == -0.5


def test_field_stats_reports_the_far_tail():
    # A bulk plus three progressively rarer levels, sized so each reported
    # quantile falls strictly inside a different one: p99 in the bulk, p99.9
    # above it, and p99.99 in the far tail where this lane's precipitation
    # deficit lives. Values are bin centres, so p99.99 may sit half a bin
    # (0.01 mm) above the true maximum.
    vals = np.concatenate([np.full(99_200, 1.0), np.full(650, 20.0),
                           np.full(100, 100.0), np.full(50, 500.0)])
    st = M.field_stats(vals)
    assert st["p99_mm"] < st["p999_mm"] < st["p9999_mm"]
    assert abs(st["p99_mm"] - 1.0) < 0.05
    assert abs(st["p9999_mm"] - 500.0) < 0.05
    assert st["max_mm"] == 500.0
