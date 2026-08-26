"""Scoreboard scoring for precip_scores — reads scores.json written by run().

Every value is an overall (all dates, steps, members) aggregate in mm per 6h
window, except the ratios, which are dimensionless. The interp-baseline row is
part of the contract: "does the model beat interpolating its driving input"
must be answerable from the board alone.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def score(results_dir: Path, lane_config: dict, eval_config: dict) -> list[dict]:
    payload_path = Path(results_dir) / "scores.json"
    if not payload_path.exists():
        LOG.warning("precip_scores: no scores.json in %s", results_dir)
        return []
    payload = json.loads(payload_path.read_text())
    s = payload.get("summary", {})

    def rec(metric: str, value, unit: str):
        if value is None:
            return None
        return {"metric": metric, "value": float(value), "unit": unit}

    def ratio(num_key: str, den_key: str):
        num, den = s.get(num_key), s.get(den_key)
        if num is None or den in (None, 0):
            return None
        return num / den

    records = [
        rec("tp_rmse_mm", s.get("model_rmse_mm"), "mm/6h"),
        rec("tp_ens_rmse_mm", s.get("model_ens_rmse_mm"), "mm/6h"),
        rec("tp_bias_mm", s.get("model_bias_mm"), "mm/6h"),
        rec("tp_corr", s.get("model_corr"), "1"),
        rec("tp_p999_ratio", ratio("model_p999_mm", "truth_p999_mm"), "1"),
        rec("tp_max_ratio", ratio("model_max_mm", "truth_max_mm"), "1"),
        rec("tp_wet_frac_ratio", ratio("model_wet_frac", "truth_wet_frac"), "1"),
        rec("tp_neg_frac", s.get("model_neg_frac"), "1"),
        rec("tp_baseline_rmse_mm", s.get("baseline_rmse_mm"), "mm/6h"),
        rec("tp_baseline_corr", s.get("baseline_corr"), "1"),
        rec("tp_rmse_vs_baseline_ratio", s.get("model_over_baseline_rmse_ratio"), "1"),
    ]
    return [r for r in records if r is not None]
