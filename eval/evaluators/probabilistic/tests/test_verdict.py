"""The owner's rule on the scores, as a truth table."""
from __future__ import annotations

import math

import pytest

from eval.evaluators.probabilistic.core.verdict import (
    BETTER,
    BETTER_CENTRE,
    NA,
    TIE,
    WORSE,
    count_labels,
    paired_difference,
    verdict_label,
)


@pytest.mark.parametrize("run_fcrps,ref_fcrps,run_rmse,ref_rmse,ratio,expected", [
    (0.90, 1.00, 1.90, 2.00, 1.00, BETTER),              # all three conditions hold
    (0.90, 1.00, 1.90, 2.00, 1.05, BETTER),              # at the edge of the tolerance
    (0.90, 1.00, 1.90, 2.00, 0.95, BETTER),
    (0.90, 1.00, 1.90, 2.00, 1.06, BETTER_CENTRE),       # over-dispersed beyond 0.05
    (0.90, 1.00, 1.90, 2.00, 0.80, BETTER_CENTRE),       # under-dispersed
    (0.90, 1.00, 2.10, 2.00, 1.00, BETTER_CENTRE),       # CRPS better, RMSE worse
    (0.90, 1.00, 2.00, 2.00, 1.00, BETTER_CENTRE),       # RMSE equal is not an improvement
    (1.00, 1.00, 1.90, 2.00, 1.00, WORSE),               # CRPS equal is not a win
    (1.10, 1.00, 1.90, 2.00, 1.00, WORSE),               # CRPS loss, whatever the rest
    (1.10, 1.00, 2.50, 2.00, 0.50, WORSE),
    (math.nan, 1.00, 1.90, 2.00, 1.00, NA),
    (0.90, 1.00, math.nan, 2.00, 1.00, NA),
    (0.90, 1.00, 1.90, 2.00, None, NA),
])
def test_truth_table(run_fcrps, ref_fcrps, run_rmse, ref_rmse, ratio, expected):
    assert verdict_label(run_fcrps, ref_fcrps, run_rmse, ref_rmse, ratio) == expected
    # strings as read from a CSV are accepted too
    assert verdict_label(str(run_fcrps), str(ref_fcrps), str(run_rmse), str(ref_rmse), str(ratio)) == expected


def test_tie_label_is_within_two_paired_standard_errors_and_removable():
    assert verdict_label(0.99, 1.00, 1.9, 2.0, 1.0, crps_diff_se=0.01) == TIE
    assert verdict_label(1.01, 1.00, 1.9, 2.0, 1.0, crps_diff_se=0.01) == TIE
    assert verdict_label(0.97, 1.00, 1.9, 2.0, 1.0, crps_diff_se=0.01) == BETTER
    assert verdict_label(1.03, 1.00, 1.9, 2.0, 1.0, crps_diff_se=0.01) == WORSE
    # without a standard error there is no tie; tie_se_multiple=None removes the label
    assert verdict_label(0.99, 1.00, 1.9, 2.0, 1.0) == BETTER
    assert verdict_label(0.99, 1.00, 1.9, 2.0, 1.0, crps_diff_se=0.01, tie_se_multiple=None) == BETTER
    assert verdict_label(0.99, 1.00, 1.9, 2.0, 1.0, crps_diff_se=0.0) == BETTER


def test_paired_difference_by_date():
    run = {"20230101": 1.0, "20230102": 1.2, "20230103": 0.9, "20230104": 5.0}
    ref = {"20230101": 1.1, "20230102": 1.3, "20230103": 0.8}
    mean, se, n = paired_difference(run, ref)
    assert n == 3 and mean == pytest.approx(-0.1 / 3)
    assert se == pytest.approx(0.11547 / math.sqrt(3), abs=1e-4)
    assert paired_difference({"a": 1.0}, {"a": 2.0}) == (-1.0, 0.0, 1)
    assert paired_difference({}, {"a": 1.0})[2] == 0


def test_count_labels_keeps_the_fixed_order():
    assert count_labels([WORSE, BETTER, TIE, BETTER]) == {BETTER: 2, WORSE: 1, TIE: 1}
