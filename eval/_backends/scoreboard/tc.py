"""TC (tropical cyclone) raw extremes — model / OPER / ENFO side by side, no score.

Per the run-trust contract (epics/run-trust-and-validation/tc-extremes-contract-decision.md,
decided 2026-06-21), TC quality is reported as RAW extremes only — no score, no ratio,
no anchor, no curated-AN, no depth formula. For each event we emit four physical
quantities for each of three sources on the SAME grid:

    min MSLP (hPa), MSLP p0.1 (hPa), max 10m wind (m/s), wind p99.9 (m/s)

Sources:
    * model  — the matched experiment row (bare key, e.g. ``tc_<event>_mslp_min``)
    * OPER   — the operational-analysis baseline (``is_analysis_row``; ``_oper_`` keys)
    * ENFO   — the ensemble reference carrying the strongest tails (``is_reference_row``;
               ``_enfo_`` keys)

"Same grid" is the entire support contract: model/OPER/ENFO must be on the same grid so
the three columns are comparable. There is no anchor or ratio. Reading is by eye.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from eval._backends.scoreboard._utils import finite_float as _finite_float
from eval._backends.scoreboard.row_matching import (
    find_model_row,
    find_row_by_predicate,
    is_analysis_row,
    is_reference_row,
)

LOG = logging.getLogger(__name__)

# Raw stats keys we surface per source. Stats-JSON rows carry these directly
# (mslp_p001 = 0.1th percentile MSLP; wind_p9999 = 99.9th percentile 10m wind).
_RAW_METRIC_KEYS = ("mslp_min", "mslp_p001", "wind_max", "wind_p9999")


def load_tc_extreme_scores_from_json(
    stats_path: Path,
    *,
    run_id: str,
    event_names: tuple[str, ...] | list[str] | None = None,
    canonical_analysis_by_event: dict[str, dict[str, Any]] | None = None,
    canonical_eefo_by_event: dict[str, dict[str, Any]] | None = None,
    extreme_reference_expid: str | None = None,
) -> dict[str, float]:
    """Load RAW TC extremes (model / OPER / ENFO) from a stats JSON.

    For each requested event, locate the model row, the OPER analysis row
    (``is_analysis_row``) and the ENFO reference row (``is_reference_row``) from the
    SAME stats JSON, and emit the four raw extremes for each source.

    Output keys (hPa for MSLP, m/s for wind):
        model : ``<event>_{mslp_min,mslp_p001,wind_max,wind_p9999}``
        OPER  : ``<event>_oper_{...}``
        ENFO  : ``<event>_enfo_{...}``

    Parameters
    ----------
    stats_path : Path to the TC stats JSON file.
    run_id : The experiment run ID to match the model row.
    event_names : Which events to load. Defaults to ("idalia", "franklin").
    canonical_analysis_by_event, canonical_eefo_by_event, extreme_reference_expid :
        Accepted for backward-compatibility with the old scoring interface and the
        ``scoreboard_metrics`` wrapper; IGNORED under the raw-extremes contract
        (no anchor, no reference-match score).

    Returns
    -------
    dict mapping the keys above to raw float values. Sources/metrics that are
    missing or non-finite are simply omitted.
    """
    del canonical_analysis_by_event, canonical_eefo_by_event, extreme_reference_expid

    with stats_path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        data = {}
    events = data.get("events")
    if not isinstance(events, dict):
        return {}

    requested_events = tuple(event_names or ("idalia", "franklin"))
    scores: dict[str, float] = {}
    for event_name in requested_events:
        event_data = events.get(event_name)
        if not isinstance(event_data, dict):
            continue
        rows = event_data.get("extreme_tail", {}).get("rows", [])
        if not isinstance(rows, list):
            rows = event_data.get("rows", [])
        if not isinstance(rows, list):
            continue
        norm_rows = [row for row in rows if isinstance(row, dict)]

        # Same stats JSON => model / OPER / ENFO are all on the same grid (the only
        # remaining support rule). Emit each source's raw extremes side by side.
        model_row = find_model_row(norm_rows, run_id)
        oper_row = find_row_by_predicate(norm_rows, is_analysis_row)
        enfo_row = find_row_by_predicate(norm_rows, is_reference_row)

        for src_tag, src_row in (("", model_row), ("oper", oper_row), ("enfo", enfo_row)):
            if not isinstance(src_row, dict):
                continue
            prefix = f"{event_name}_{src_tag}_" if src_tag else f"{event_name}_"
            for raw_key in _RAW_METRIC_KEYS:
                v = _finite_float(src_row.get(raw_key))
                if v is not None:
                    scores[f"{prefix}{raw_key}"] = v
    return scores
