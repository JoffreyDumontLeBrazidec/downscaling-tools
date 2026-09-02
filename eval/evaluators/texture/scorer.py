"""Metric records for the texture evaluator.

One record per (state, stratum, statistic, side), named
``tex_{state}_{stratum}_{statistic}_{truth|model|ratio|delta|sd}``:

* ``_truth`` / ``_model``: mean over the (file, member) samples of the statistic
  measured on the truth / on the model output;
* ``_ratio``: mean of the per-sample model/truth ratio (variances only);
* ``_delta``: mean of the per-sample model - truth difference (correlations only);
* ``_sd``: standard deviation over samples of that difference -- the null
  scatter a later arm has to beat.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runner import DELTA_STATS, RATIO_STATS, STAT_NAMES

UNITS = {
    "resid_var": "normalised_variance",
    "fine_var": "normalised_variance",
    "zonal_diff_var": "normalised_variance",
    "fine_lag1_zonal": "correlation",
    "fine_nn_corr": "correlation",
    "top5_share": "fraction",
    "kurtosis": "excess_kurtosis",
}


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    path = Path(results_dir) / "texture.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())

    records: list[dict[str, Any]] = []

    def _add(metric: str, value, unit: str) -> None:
        if value is None:
            return
        records.append({"metric": metric, "value": value, "unit": unit})

    for row in payload.get("aggregate", []):
        stem = f"tex_{row['state']}_{row['stratum']}"
        for stat in STAT_NAMES:
            for side in ("truth", "model"):
                _add(f"{stem}_{stat}_{side}", row[side][stat]["mean"], UNITS[stat])
        for stat in RATIO_STATS:
            _add(f"{stem}_{stat}_ratio", row["ratio"][stat]["mean"], "ratio")
        for stat in DELTA_STATS:
            _add(f"{stem}_{stat}_delta", row["delta"][stat]["mean"], "correlation")
            _add(f"{stem}_{stat}_sd", row["delta"][stat]["sd"], "correlation")
    return records
