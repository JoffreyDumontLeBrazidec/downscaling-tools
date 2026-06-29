"""TC evaluator — raw extremes extraction (model / OPER / ENFO / EEFO).

Wraps eval._backends.scoreboard.tc with the standard evaluator interface:
returns list[dict] of {"metric", "value", "unit"} records.

Per the run-trust contract (decided 2026-06-21) TC quality is RAW extremes only —
no score, no ratio, no anchor. For each event we emit four physical quantities
(min MSLP, MSLP p0.1, max 10m wind, wind p99.9) for the model and the OPER / ENFO /
EEFO baselines from the SAME stats JSON (same grid). The baseline sources are
classified lane-agnostically by the backend (row prefix), so a lane only emits the
sources its stats JSON actually carries. Reading is by eye.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from eval._backends.scoreboard.row_matching import bundle_enfo_labels
from eval._backends.scoreboard.tc import load_tc_extreme_scores_from_json

LOG = logging.getLogger(__name__)

# Raw extremes emitted per source, with their physical units.
_RAW_METRICS = (
    ("mslp_min", "hPa"),
    ("mslp_p001", "hPa"),
    ("wind_max", "m/s"),
    ("wind_p9999", "m/s"),
)


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    run_id: str = "",
    stats_filename: str = "stats.json",
    **kwargs,
) -> list[dict[str, Any]]:
    """Score TC results from a stats JSON — RAW extremes only.

    Parameters
    ----------
    results_dir : Path to the evaluator output directory containing stats JSON.
    lane_config : Full lane configuration dict.
    eval_config : TC-specific config (lane_config["tc"]).
    run_id : Experiment run ID for row matching.
    stats_filename : Name of the stats JSON file.

    Returns
    -------
    List of {"metric": str, "value": float, "unit": str} records. For each event,
    emits ``tc_<event>_{mslp_min,mslp_p001,wind_max,wind_p9999}`` for the model plus
    the ``_oper_`` / ``_enfo_`` / ``_eefo_`` variants — all raw hPa / m/s, no score or
    ratio. A baseline source absent from the stats JSON is simply skipped.
    """
    results_dir = Path(results_dir)
    stats_path = results_dir / stats_filename

    if not stats_path.exists():
        LOG.warning("TC stats file not found: %s", stats_path)
        return []

    event_names = eval_config.get("events", [])
    if not event_names:
        LOG.warning("No TC events configured for scoring")
        return []

    # Keep the scoreboard `enfo` column even though the duplicate ENFO reference
    # curve is deduped from the plot: the ENFO data still lives in the bundle
    # input/target row (lane-aware). Same spirit as o48_o96 (input labelled 'ENFO').
    enfo_labels = bundle_enfo_labels(lane_config, eval_config)
    scores = load_tc_extreme_scores_from_json(
        stats_path,
        run_id=run_id,
        event_names=event_names,
        enfo_labels=enfo_labels,
    )

    records: list[dict[str, Any]] = []
    for event_name in event_names:
        # model (bare), OPER, ENFO, EEFO — raw extremes side by side on the same grid.
        # Sources the backend didn't find (no matching row in this lane's stats JSON)
        # are absent from `scores` and skipped by the `score_key in scores` guard below.
        for src_tag in ("", "oper", "enfo", "eefo"):
            for raw_key, unit in _RAW_METRICS:
                score_key = (
                    f"{event_name}_{src_tag}_{raw_key}" if src_tag
                    else f"{event_name}_{raw_key}"
                )
                metric = (
                    f"tc_{event_name}_{src_tag}_{raw_key}" if src_tag
                    else f"tc_{event_name}_{raw_key}"
                )
                if score_key in scores:
                    records.append({
                        "metric": metric,
                        "value": scores[score_key],
                        "unit": unit,
                    })

    return records
