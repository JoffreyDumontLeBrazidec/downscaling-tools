"""The owner's rule 32 v2 on the scores, as a truth table."""
from __future__ import annotations

import math

import pytest

from eval.evaluators.probabilistic.core.verdict import (
    AS_FAR,
    BETTER,
    BETTER_CALIBRATION,
    BETTER_CENTRE,
    BETTER_CENTRE_UNDECIDED,
    CLOSER,
    CRPS_ONLY,
    CRPS_ONLY_UNDECIDED,
    FURTHER,
    INSIDE,
    JUDGED_BANDS,
    NA,
    NOT_JUDGED,
    OUTSIDE,
    TIE,
    UNDECIDED,
    UNDECIDED_WHY,
    WORSE,
    base_label,
    calibration_state,
    count_labels,
    paired_difference,
    verdict_label,
)


@pytest.mark.parametrize("run_fcrps,ref_fcrps,run_rmse,ref_rmse,ratio,expected", [
    (0.90, 1.00, 1.90, 2.00, 1.00, BETTER),              # centre better, calibration inside
    (0.90, 1.00, 1.90, 2.00, 1.05, BETTER),              # at the edge of the tolerance
    (0.90, 1.00, 1.90, 2.00, 0.95, BETTER),
    (0.90, 1.00, 1.90, 2.00, 1.06, BETTER_CENTRE),       # centre better, over-dispersed beyond 0.05
    (0.90, 1.00, 1.90, 2.00, 0.80, BETTER_CENTRE),       # centre better, under-dispersed
    (0.90, 1.00, 2.10, 2.00, 1.00, BETTER_CALIBRATION),  # RMSE worse, calibration inside
    (0.90, 1.00, 2.00, 2.00, 1.00, BETTER_CALIBRATION),  # RMSE equal is not a better centre
    (0.90, 1.00, 2.10, 2.00, 0.80, CRPS_ONLY),           # neither
    (1.00, 1.00, 1.90, 2.00, 1.00, WORSE),               # CRPS equal is not a win
    (1.10, 1.00, 1.90, 2.00, 1.00, WORSE),               # CRPS loss, whatever the rest
    (1.10, 1.00, 2.50, 2.00, 0.50, WORSE),
    (math.nan, 1.00, 1.90, 2.00, 1.00, NA),
    (0.90, 1.00, math.nan, 2.00, 1.00, NA),
    (0.90, 1.00, 1.90, 2.00, None, NA),
])
def test_truth_table_on_the_point_estimate(run_fcrps, ref_fcrps, run_rmse, ref_rmse, ratio, expected):
    # no interval and no reference ratio: the point decides, and the "outside" label has no suffix
    assert verdict_label(run_fcrps, ref_fcrps, run_rmse, ref_rmse, ratio) == expected
    # strings as read from a CSV are accepted too
    assert verdict_label(str(run_fcrps), str(ref_fcrps), str(run_rmse), str(ref_rmse), str(ratio)) == expected


@pytest.mark.parametrize("centre_better,lo,hi,expected", [
    (True, 0.97, 1.03, BETTER),                                           # interval inside
    (True, 1.07, 1.15, f"{BETTER_CENTRE} {FURTHER}"),                     # interval outside
    (True, 0.98, 1.08, f"{BETTER_CENTRE_UNDECIDED} {UNDECIDED_WHY}"),     # crosses 1.05
    (True, 0.90, 1.00, f"{BETTER_CENTRE_UNDECIDED} {UNDECIDED_WHY}"),     # crosses 0.95
    (True, 0.90, 1.10, f"{BETTER_CENTRE_UNDECIDED} {UNDECIDED_WHY}"),     # covers both edges
    (False, 0.97, 1.03, BETTER_CALIBRATION),
    (False, 1.07, 1.15, CRPS_ONLY),
    (False, 0.98, 1.08, f"{CRPS_ONLY_UNDECIDED} {UNDECIDED_WHY}"),
])
def test_truth_table_on_the_interval(centre_better, lo, hi, expected):
    run_rmse = 1.9 if centre_better else 2.1
    # the interval decides the calibration state; the point estimate only feeds the closer/further
    # suffix of the "outside" label, so a misleading point (1.00 or 1.10) changes nothing else
    points = ((lo + hi) / 2,) if expected.startswith(BETTER_CENTRE) else (1.00, 1.10, (lo + hi) / 2)
    for point in points:
        label = verdict_label(0.9, 1.0, run_rmse, 2.0, point, ref_ratio_corrected=1.05,
                              run_ratio_lo=lo, run_ratio_hi=hi)
        assert label == expected, (point, label)


def test_undecided_is_never_better_model():
    for lo, hi in ((0.94, 1.04), (0.96, 1.06), (0.5, 1.5)):
        assert BETTER != verdict_label(0.9, 1.0, 1.9, 2.0, 1.0, run_ratio_lo=lo, run_ratio_hi=hi)
        assert "undecided" in verdict_label(0.9, 1.0, 1.9, 2.0, 1.0, run_ratio_lo=lo, run_ratio_hi=hi)


def test_closer_or_further_suffix_compares_with_the_reference():
    outside = dict(run_ratio_lo=1.10, run_ratio_hi=1.20)
    assert verdict_label(0.9, 1.0, 1.9, 2.0, 1.15, ref_ratio_corrected=1.30, **outside) == f"{BETTER_CENTRE} {CLOSER}"
    assert verdict_label(0.9, 1.0, 1.9, 2.0, 1.15, ref_ratio_corrected=1.02, **outside) == f"{BETTER_CENTRE} {FURTHER}"
    assert verdict_label(0.9, 1.0, 1.9, 2.0, 1.15, ref_ratio_corrected=0.85, **outside) == f"{BETTER_CENTRE} {AS_FAR}"
    # under-dispersed run against an over-dispersed reference: distances to 1 are what count
    assert verdict_label(0.9, 1.0, 1.9, 2.0, 0.80, ref_ratio_corrected=1.30) == f"{BETTER_CENTRE} {CLOSER}"
    # no reference ratio: no suffix; the suffix is never on the other labels
    assert verdict_label(0.9, 1.0, 1.9, 2.0, 1.15, **outside) == BETTER_CENTRE
    assert verdict_label(0.9, 1.0, 2.1, 2.0, 1.15, ref_ratio_corrected=1.30, **outside) == CRPS_ONLY


def test_calibration_state_three_ways():
    assert calibration_state(1.00) == INSIDE and calibration_state(1.05) == INSIDE and calibration_state(0.95) == INSIDE
    assert calibration_state(1.06) == OUTSIDE and calibration_state(0.80) == OUTSIDE
    assert calibration_state(1.00, 0.96, 1.04) == INSIDE
    assert calibration_state(1.00, 0.95, 1.05) == INSIDE          # edges count as inside
    assert calibration_state(1.10, 1.06, 1.20) == OUTSIDE and calibration_state(0.9, 0.80, 0.94) == OUTSIDE
    assert calibration_state(1.00, 0.94, 1.04) == UNDECIDED and calibration_state(1.00, 0.96, 1.06) == UNDECIDED
    # a half interval (one bound missing or NaN) falls back to the point
    assert calibration_state(1.00, math.nan, 1.10) == INSIDE and calibration_state("1.00", None, None) == INSIDE
    assert calibration_state("1.2", "", "") == OUTSIDE


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


def test_count_labels_keeps_the_fixed_order_and_drops_suffixes():
    labels = [WORSE, BETTER, TIE, BETTER, f"{BETTER_CENTRE} {CLOSER}", f"{BETTER_CENTRE} {FURTHER}",
              f"{CRPS_ONLY_UNDECIDED} {UNDECIDED_WHY}", NOT_JUDGED]
    assert count_labels(labels) == {BETTER: 2, BETTER_CENTRE: 2, CRPS_ONLY_UNDECIDED: 1, WORSE: 1, TIE: 1}
    assert base_label(f"{BETTER_CENTRE} {CLOSER}") == BETTER_CENTRE and base_label(BETTER) == BETTER
    assert JUDGED_BANDS == ("full", "full_inner", "fine") and "large" not in JUDGED_BANDS
