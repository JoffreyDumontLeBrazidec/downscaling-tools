"""Row classification and matching for TC stats rows."""

from __future__ import annotations

import os
import re
from typing import Any

from eval.scoreboard.types import RowClassification

# Default prefix→classification mapping. Callers can override.
DEFAULT_PREFIX_MAP: dict[str, RowClassification] = {
    "OPER": RowClassification.ANALYSIS,
    "ENFO": RowClassification.REFERENCE,
    "IP6Y": RowClassification.REFERENCE,
    "EEFO": RowClassification.EEFO,
    # Truth / input baseline rows present in some lanes (o48_o96, o320_o1280,
    # o1280_o2560). Without these they default to MODEL, giving find_model_row two
    # MODEL-classified rows so its "sole MODEL row" fallback returns None and the
    # model extreme columns silently drop (run-trust 2026-06-23 regression fix).
    "IEKM": RowClassification.UNKNOWN,    # high-res verifying analysis (truth)
    "TARGET": RowClassification.UNKNOWN,  # display-labelled truth ("target O1280")
    "INPUT": RowClassification.UNKNOWN,   # display-labelled input baseline ("input O320")
}

CHECKPOINT_TOKEN_RE = re.compile(r"(?:^|manual_)([0-9a-f]{7,64})(?:_|$)")


def classify_row(
    exp_label: str,
    *,
    prefix_map: dict[str, RowClassification] | None = None,
) -> RowClassification:
    """Classify a TC stats row by its experiment label prefix."""
    mapping = prefix_map or DEFAULT_PREFIX_MAP
    upper = exp_label.strip().upper()
    for prefix, classification in mapping.items():
        if upper.startswith(prefix.upper()):
            return classification
    return RowClassification.MODEL


def is_analysis_row(exp: str) -> bool:
    return classify_row(exp) == RowClassification.ANALYSIS


def is_reference_row(exp: str) -> bool:
    return classify_row(exp) == RowClassification.REFERENCE


def is_eefo_row(exp: str) -> bool:
    return classify_row(exp) == RowClassification.EEFO


_GRIB_STREAM_RE = re.compile(r"([A-Za-z]+)_[Oo]?\d")


def _grib_stream(path):
    """First product token of a bundle source GRIB, lowercased ('enfo', 'eefo', 'iekm'...)."""
    if not path:
        return None
    m = _GRIB_STREAM_RE.search(os.path.basename(str(path)))
    return m.group(1).lower() if m else None


def bundle_enfo_labels(lane_config, eval_config):
    """Display labels of the bundle curve(s) that ARE the ENFO ensemble.

    After _strip_bundle_duplicate_references removes the duplicate ENFO reference *curve*
    from the plot, the ENFO data still lives in the bundle input/target row. This lets the
    scoreboard keep its `enfo` column (labelled enfo, never 'target'/'input') by also
    accepting that row. Lane-aware via prepare.args product stream: ENFO can be the input
    (o1280_o2560, o48_o96) or the target (o320_o1280, o96_o320); truths (IEKM/DestinE)
    carry no enfo stream and are not included. Same spirit as o48_o96, whose input curve
    is already labelled "ENFO" and so classifies into the enfo column directly.
    """
    prep = ((lane_config.get("prepare") or {}).get("args")) or {}
    out = set()
    for label_keys, grib_key in (
        (("input_label",), "lres_sfc_grib"),
        (("target_nc_label", "target_label"), "target_sfc_grib"),
    ):
        if _grib_stream(prep.get(grib_key)) == "enfo":
            for lk in label_keys:
                lbl = eval_config.get(lk)
                if lbl:
                    out.add(str(lbl).strip())
                    break
    return out


def find_row_by_predicate(
    rows: list[dict[str, Any]],
    predicate,
) -> dict[str, Any] | None:
    for row in rows:
        if predicate(str(row.get("exp", "")).strip()):
            return row
    return None


def extract_checkpoint_token(text: str) -> str | None:
    match = CHECKPOINT_TOKEN_RE.search(str(text).strip())
    if match:
        return match.group(1)
    return None


def tc_candidates(run_id: str) -> list[str]:
    """Generate candidate labels for matching a run_id against TC rows."""
    candidates = [run_id.strip()]
    token = extract_checkpoint_token(run_id)
    if token:
        candidates.extend({token, token[:8], token[:7]})
    return [c for c in candidates if c]


def find_model_row(
    rows: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any] | None:
    """Find the TC stats row matching a given run_id.

    Tries exact match, then substring match, then falls back to
    the sole non-reference row if only one exists.
    """
    candidates = tc_candidates(run_id)

    for candidate in candidates:
        for row in rows:
            exp = str(row.get("exp", "")).strip()
            if exp == candidate:
                return row

    for candidate in candidates:
        for row in rows:
            exp = str(row.get("exp", "")).strip()
            if candidate in exp or exp in candidate:
                return row

    non_reference = [
        row for row in rows
        if classify_row(str(row.get("exp", "")).strip()) == RowClassification.MODEL
    ]
    if len(non_reference) == 1:
        return non_reference[0]

    return None
