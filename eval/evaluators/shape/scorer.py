"""Metric records for the shape evaluator.

``shape_<statistic>_<field>_<band>_<var>_s<step>_<window>`` = pooled mean over
(date, member) cells, and ``..._se`` = its date-clustered bootstrap standard error.
Only the three statistics of the design note are emitted (elongated fraction,
window aspect ratio, anisotropy index over the open ocean); summary.json holds more.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METRIC_STATS = {"elong_frac_gt3": "fraction", "ell_ratio_median": "ratio",
                "A_open_ocean": "ratio"}


def score(results_dir, lane_config: dict, eval_config: dict, **kwargs) -> list[dict[str, Any]]:
    path = Path(results_dir) / "summary.json"
    if not path.exists():
        return []
    recs = []
    for r in json.loads(path.read_text()).get("rows", []):
        if r["statistic"] not in METRIC_STATS or r["mean"] is None:
            continue
        stem = (f"shape_{r['statistic']}_{r['field']}_{r['band']}_{r['var']}_s{r['step']}_"
                f"{r['window']}")
        unit = METRIC_STATS[r["statistic"]]
        recs.append({"metric": stem, "value": r["mean"], "unit": unit})
        if r["se_date"] is not None:
            recs.append({"metric": stem + "_se", "value": r["se_date"], "unit": unit})
    return recs
