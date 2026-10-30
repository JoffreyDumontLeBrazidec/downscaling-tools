"""The verdict label on the scores of a run against its reference (owner's rule, 30 Oct 2026).

    "Call a CRPS win a 'better model' only if mean RMSE also improves and corrected
    spread/RMSE stays within about 0.05 of 1. Otherwise, write 'better centre, worse
    calibration'. Realism and spectra still come first, as usual."

``verdict_label`` applies the rule to one (variable, lead, region, band):

``better model``
    run fair CRPS below the reference's, run ensemble-mean RMSE below the reference's, and
    ``|run corrected spread/RMSE - 1| <= 0.05`` (``RATIO_TOLERANCE``).
``better centre, worse calibration``
    run fair CRPS below the reference's, but the RMSE or the calibration condition fails.
``worse CRPS``
    run fair CRPS at or above the reference's.
``tie``
    NOT in the owner's rule, added by the harness: the CRPS difference is within
    ``TIE_SE_MULTIPLE`` (2) paired standard errors over dates. The pairing is by date
    (``paired_difference``); the label is only given when the per-date values of both arms
    are at hand. Set ``TIE_SE_MULTIPLE`` to ``None`` to remove the label altogether.
``n/a``
    a number is missing.

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

BETTER = "better model"
BETTER_CENTRE = "better centre, worse calibration"
WORSE = "worse CRPS"
TIE = "tie"
NA = "n/a"
LABELS = (BETTER, BETTER_CENTRE, WORSE, TIE, NA)


def _finite(*values) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def verdict_label(run_fcrps, ref_fcrps, run_rmse, ref_rmse, run_ratio_corrected, *,
                  crps_diff_se=None, ratio_tolerance: float = RATIO_TOLERANCE,
                  tie_se_multiple: float | None = TIE_SE_MULTIPLE) -> str:
    """The label of one (variable, lead, region, band); see the module docstring.

    ``crps_diff_se`` is the paired standard error over dates of ``run - reference`` fair
    CRPS; when given and ``tie_se_multiple`` is not None, a difference within that many
    standard errors is a ``tie``.
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
    # 1e-9: so a ratio read as 1.05 from a CSV sits inside "within 0.05", not on a floating-point edge
    if float(run_rmse) < float(ref_rmse) and abs(float(run_ratio_corrected) - 1.0) <= ratio_tolerance + 1e-9:
        return BETTER
    return BETTER_CENTRE


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
    """How many of each label, in the fixed label order, for a one-line summary."""
    labels = list(labels)
    return {name: labels.count(name) for name in LABELS if name in labels}
