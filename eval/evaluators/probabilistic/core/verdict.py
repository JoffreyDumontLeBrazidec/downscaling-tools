"""The verdict label on the scores of a run against its reference (owner's rule 32, v2).

The rule of 30 Oct 2026 ("call a CRPS win a 'better model' only if mean RMSE also improves
and corrected spread/RMSE stays within about 0.05 of 1; otherwise 'better centre, worse
calibration'; realism and spectra still come first") was revised after the first real-data
read (CA300 against DB300, E2 box, 30 dates x 10 members): the arithmetic was right, the
two labels were not. Version 2 keeps the two tests and names their four outcomes, and
reads the calibration test on the bootstrap interval of the ratio instead of its point.

``verdict_label`` applies the rule to one (variable, lead, region, band). First the CRPS:

``tie``
    NOT in the owner's rule, added by the harness: the fair CRPS difference is within
    ``TIE_SE_MULTIPLE`` (2) paired standard errors over dates (``paired_difference``); only
    given when the per-date values of both arms are at hand. ``TIE_SE_MULTIPLE = None``
    removes the label.
``worse CRPS``
    run fair CRPS at or above the reference's.

Then, for a CRPS win, two tests: the CENTRE (run ensemble-mean RMSE below the reference's)
and the CALIBRATION (the run's corrected spread/RMSE within ``RATIO_TOLERANCE`` (0.05) of
1). The calibration test is three-way when the ratio's 95 % bootstrap interval over dates
is known (``calibration_by_lead.csv`` columns ``spread_skill_ratio_corrected_lo95`` /
``_hi95``): the interval fully inside [0.95, 1.05] is "inside", fully outside is
"outside", and one that crosses 0.95 or 1.05 is "undecided". Without an interval (older
output, or one date) the point estimate decides inside/outside.

``better model``
    centre better AND calibration inside.
``better centre, calibration outside 0.05 (closer to 1 than reference)``
    centre better, calibration outside; the suffix says whether the run's ratio is closer to
    1 than the reference's or further from it (``(further from 1 than reference)``).
``better centre, calibration undecided (ratio CI crosses 0.95/1.05)``
    centre better, calibration undecided: never "better model".
``better calibration, not better centre``
    centre not better, calibration inside.
``CRPS win, neither centre nor calibration``
    centre not better, calibration outside.
``CRPS win, calibration undecided (ratio CI crosses 0.95/1.05)``
    centre not better, calibration undecided.
``n/a``
    a number is missing.

Bands. The label is given on the ``full``, ``full_inner`` and ``fine`` bands
(``JUDGED_BANDS``). The ``large`` band is reported, not judged (``NOT_JUDGED``): the scales
above the cut come from the input, so a win or a loss there is the input's, not the
model's. The bands are not additive (``P_L`` is a taper), so their labels are never summed.

Scores never outrank parts 1 and 2 of the read (physical structure and spectra); the
label is the verdict on the scores alone.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

# |corrected spread/RMSE - 1| allowed for "better model" ("within about 0.05 of 1")
RATIO_TOLERANCE = 0.05
# The tie label (harness addition, not the owner's rule): None removes it.
TIE_SE_MULTIPLE: float | None = 2.0
# 1e-9: so a ratio read as 1.05 from a CSV sits inside "within 0.05", not on a floating-point edge
_EDGE = 1e-9

BETTER = "better model"
BETTER_CENTRE = "better centre, calibration outside 0.05"
BETTER_CENTRE_UNDECIDED = "better centre, calibration undecided"
BETTER_CALIBRATION = "better calibration, not better centre"
CRPS_ONLY = "CRPS win, neither centre nor calibration"
CRPS_ONLY_UNDECIDED = "CRPS win, calibration undecided"
WORSE = "worse CRPS"
TIE = "tie"
NA = "n/a"
NOT_JUDGED = "not judged (large scales come from the input)"
# the base labels, in the order of the one-line summary (suffixes in brackets are not part of them)
LABELS = (BETTER, BETTER_CENTRE, BETTER_CENTRE_UNDECIDED, BETTER_CALIBRATION, CRPS_ONLY,
          CRPS_ONLY_UNDECIDED, WORSE, TIE, NA)

# the suffixes
CLOSER = "(closer to 1 than reference)"
FURTHER = "(further from 1 than reference)"
AS_FAR = "(as far from 1 as reference)"
UNDECIDED_WHY = "(ratio CI crosses 0.95/1.05)"

# calibration states
INSIDE, OUTSIDE, UNDECIDED = "inside", "outside", "undecided"

# bands the label is given on; the others are reported, not judged
JUDGED_BANDS = ("full", "full_inner", "fine")


def _finite(*values) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def calibration_state(ratio_corrected, lo=None, hi=None, *, ratio_tolerance: float = RATIO_TOLERANCE) -> str:
    """``inside`` / ``outside`` / ``undecided`` for the calibration test of one arm.

    With a finite interval ``[lo, hi]`` (the 95 % bootstrap interval of the corrected ratio
    over dates): fully inside ``[1 - tol, 1 + tol]`` is inside, fully outside is outside,
    and an interval that crosses either edge is undecided. Without one, the point decides.
    """
    low, high = 1.0 - ratio_tolerance - _EDGE, 1.0 + ratio_tolerance + _EDGE
    if _finite(lo, hi) and float(lo) <= float(hi):
        lo, hi = float(lo), float(hi)
        if lo >= low and hi <= high:
            return INSIDE
        if hi < low or lo > high:
            return OUTSIDE
        return UNDECIDED
    return INSIDE if low <= float(ratio_corrected) <= high else OUTSIDE


def _closer_suffix(run_ratio, ref_ratio) -> str:
    if not _finite(ref_ratio):
        return ""
    a, b = abs(float(run_ratio) - 1.0), abs(float(ref_ratio) - 1.0)
    if abs(a - b) <= _EDGE:
        return " " + AS_FAR
    return " " + (CLOSER if a < b else FURTHER)


def verdict_label(run_fcrps, ref_fcrps, run_rmse, ref_rmse, run_ratio_corrected, *,
                  ref_ratio_corrected=None, run_ratio_lo=None, run_ratio_hi=None,
                  crps_diff_se=None, ratio_tolerance: float = RATIO_TOLERANCE,
                  tie_se_multiple: float | None = TIE_SE_MULTIPLE) -> str:
    """The label of one (variable, lead, region, band); see the module docstring.

    ``crps_diff_se`` is the paired standard error over dates of ``run - reference`` fair
    CRPS; when given and ``tie_se_multiple`` is not None, a difference within that many
    standard errors is a ``tie``. ``run_ratio_lo`` / ``run_ratio_hi`` are the bootstrap
    interval of the run's corrected ratio (three-way calibration test when given);
    ``ref_ratio_corrected`` gives the closer/further suffix of the "outside" label.
    """
    if not _finite(run_fcrps, ref_fcrps):
        return NA
    diff = float(run_fcrps) - float(ref_fcrps)
    if tie_se_multiple is not None and _finite(crps_diff_se) and float(crps_diff_se) > 0.0 \
            and abs(diff) <= tie_se_multiple * float(crps_diff_se):
        return TIE
    if diff >= 0.0:
        return WORSE
    if not _finite(run_rmse, ref_rmse, run_ratio_corrected):
        return NA
    centre = float(run_rmse) < float(ref_rmse)
    state = calibration_state(run_ratio_corrected, run_ratio_lo, run_ratio_hi, ratio_tolerance=ratio_tolerance)
    if state == INSIDE:
        return BETTER if centre else BETTER_CALIBRATION
    if state == UNDECIDED:
        return f"{BETTER_CENTRE_UNDECIDED if centre else CRPS_ONLY_UNDECIDED} {UNDECIDED_WHY}"
    if centre:
        return BETTER_CENTRE + _closer_suffix(run_ratio_corrected, ref_ratio_corrected)
    return CRPS_ONLY


def base_label(label: str) -> str:
    """The label without its bracketed suffix (``closer``/``further``/``undecided why``)."""
    label = str(label)
    for suffix in (CLOSER, FURTHER, AS_FAR, UNDECIDED_WHY):
        if label.endswith(" " + suffix):
            return label[: -len(suffix) - 1]
    return label


def paired_difference(run_by_date: Mapping[str, float], ref_by_date: Mapping[str, float]) -> tuple[float, float, int]:
    """``(mean, standard error, n)`` of ``run - reference`` over the dates both arms hold.

    Standard error of the mean of the paired differences (ddof 1); 0 for one date, NaN for
    none.
    """
    dates = sorted(set(run_by_date) & set(ref_by_date))
    diffs = [float(run_by_date[d]) - float(ref_by_date[d]) for d in dates
             if _finite(run_by_date[d], ref_by_date[d])]
    n = len(diffs)
    if n == 0:
        return math.nan, math.nan, 0
    mean = sum(diffs) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    return mean, math.sqrt(var / n), n


def count_labels(labels: Iterable[str]) -> dict[str, int]:
    """How many of each base label, in the fixed label order, for a one-line summary.

    Suffixes are dropped; ``NOT_JUDGED`` is not a label and is not counted.
    """
    bases = [base_label(lab) for lab in labels]
    return {name: bases.count(name) for name in LABELS if name in bases}
