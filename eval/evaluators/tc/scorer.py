"""TC evaluator — pure scoring math.

Wraps eval.scoreboard.tc scoring functions with the standard evaluator
interface: returns list[dict] of {"metric", "value", "unit"} records.

Scoring algorithms are identical to scoreboard/tc.py — this module
imports them directly to guarantee mathematical equivalence.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from eval._backends.scoreboard.tc import (
    load_tc_extreme_scores_from_json,
    multi_depth_tc_score,
    normalize_tc_rows,
)

LOG = logging.getLogger(__name__)


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    run_id: str = "",
    stats_filename: str = "stats.json",
    **kwargs,
) -> list[dict[str, Any]]:
    """Score TC results from a stats JSON.

    Parameters
    ----------
    results_dir : Path to the evaluator output directory containing stats JSON.
    lane_config : Full lane configuration dict.
    eval_config : TC-specific config (lane_config["tc"]).
    run_id : Experiment run ID for row matching.
    stats_filename : Name of the stats JSON file.

    Returns
    -------
    List of {"metric": str, "value": float, "unit": str} records.
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

    scores = load_tc_extreme_scores_from_json(
        stats_path,
        run_id=run_id,
        event_names=event_names,
        extreme_reference_expid=eval_config.get("extreme_reference_expid"),
    )

    records: list[dict[str, Any]] = []
    for event_name in event_names:
        if event_name in scores:
            records.append({
                "metric": f"tc_{event_name}_extreme_score",
                "value": scores[event_name],
                "unit": "score_0_1",
            })
        enfo_dev_key = f"{event_name}_enfo_dev"
        if enfo_dev_key in scores:
            records.append({
                "metric": f"tc_{event_name}_enfo_deviation",
                "value": scores[enfo_dev_key],
                "unit": "deviation",
            })
        enfo_match_key = f"{event_name}_enfo_match"
        if enfo_match_key in scores:
            records.append({
                "metric": f"tc_{event_name}_enfo_match",
                "value": scores[enfo_match_key],
                "unit": "score_0_1",
            })
        mslp_reach_key = f"{event_name}_mslp_reach"
        if mslp_reach_key in scores:
            records.append({
                "metric": f"tc_{event_name}_mslp_reach",
                "value": scores[mslp_reach_key],
                "unit": "position",
            })
        wind_reach_key = f"{event_name}_wind_reach"
        if wind_reach_key in scores:
            records.append({
                "metric": f"tc_{event_name}_wind_reach",
                "value": scores[wind_reach_key],
                "unit": "position",
            })
        # AN-anchored tail-extreme ratios (1.0 = matches AN; >1 = more extreme; <1 = less)
        for ratio_key in ("mslp_p001_ratio", "mslp_min_ratio", "wind_p9999_ratio", "wind_max_ratio"):
            sk = f"{event_name}_{ratio_key}"
            if sk in scores:
                records.append({
                    "metric": f"tc_{event_name}_{ratio_key}",
                    "value": scores[sk],
                    "unit": "ratio_vs_an",
                })
        # Raw physical extremes (hPa / m/s): matched-row deepest MSLP + strongest wind,
        # plus OPER/ENFO/EEFO anchors. Robust, hard-to-misread cross-check for the
        # support-sensitive extreme_score. (2026-06-18)
        for raw_key, unit in (("mslp_min", "hPa"), ("wind_max", "m/s")):
            sk = f"{event_name}_{raw_key}"
            if sk in scores:
                records.append({
                    "metric": f"tc_{event_name}_{raw_key}",
                    "value": scores[sk],
                    "unit": unit,
                })
            for anchor in ("oper", "enfo", "eefo"):
                ak = f"{event_name}_{anchor}_{raw_key}"
                if ak in scores:
                    records.append({
                        "metric": f"tc_{event_name}_{anchor}_{raw_key}",
                        "value": scores[ak],
                        "unit": unit,
                    })

    # Compute aggregate TC score (mean of per-event extreme scores)
    event_scores = [r["value"] for r in records if r["metric"].endswith("_extreme_score")]
    if event_scores:
        records.append({
            "metric": "tc_mean_extreme_score",
            "value": sum(event_scores) / len(event_scores),
            "unit": "score_0_1",
        })

    # Compute aggregate ENFO match (mean of per-event ENFO match scores)
    match_scores = [
        r["value"] for r in records
        if r["metric"].endswith("_enfo_match") and r["metric"] != "tc_mean_enfo_match"
    ]
    if match_scores:
        records.append({
            "metric": "tc_mean_enfo_match",
            "value": sum(match_scores) / len(match_scores),
            "unit": "score_0_1",
        })

    # Aggregate AN→ENFO reach positions per variable (mean across events).
    mslp_reaches = [
        r["value"] for r in records
        if r["metric"].endswith("_mslp_reach") and r["metric"] != "tc_mean_mslp_reach"
    ]
    if mslp_reaches:
        records.append({
            "metric": "tc_mean_mslp_reach",
            "value": sum(mslp_reaches) / len(mslp_reaches),
            "unit": "position",
        })
    wind_reaches = [
        r["value"] for r in records
        if r["metric"].endswith("_wind_reach") and r["metric"] != "tc_mean_wind_reach"
    ]
    if wind_reaches:
        records.append({
            "metric": "tc_mean_wind_reach",
            "value": sum(wind_reaches) / len(wind_reaches),
            "unit": "position",
        })

    # Aggregate AN-anchored tail-extreme ratios across events (mean per ratio key).
    for ratio_key in ("mslp_p001_ratio", "mslp_min_ratio", "wind_p9999_ratio", "wind_max_ratio"):
        agg_metric = f"tc_mean_{ratio_key}"
        per_event = [
            r["value"] for r in records
            if r["metric"].endswith(f"_{ratio_key}") and r["metric"] != agg_metric
        ]
        if per_event:
            records.append({
                "metric": agg_metric,
                "value": sum(per_event) / len(per_event),
                "unit": "ratio_vs_an",
            })

    return records
