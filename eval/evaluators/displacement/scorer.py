"""Metric records for the displacement evaluator.

One record per (box, field, pair, quantity), named
``disp_{box}_{field}_{pair}_{quantity}``:

* ``_east_km`` / ``_north_km``  the mean offset that best aligns the two fields,
  positive meaning the second field's feature sits east or north of the first's;
* ``_distance_km``             the length of that shift;
* ``_east_km_sd`` and friends  the scatter over the (file, member) samples,
  which is what a claimed systematic shift has to beat;
* ``_corr_zero`` / ``_corr_best``  the correlation of the two fields without any
  shift and at the best shift; a large gain means the shift is doing real work;
* ``disp_{box}_msl_minimum_{pair}_km``  the distance between the two pressure
  minima, which is the same question asked of one identifiable feature.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runner import PAIRS


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    path = Path(results_dir) / "displacement.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    records: list[dict[str, Any]] = []

    def _add(metric: str, value, unit: str) -> None:
        if value is not None:
            records.append({"metric": metric, "value": value, "unit": unit})

    for row in payload.get("aggregate", []):
        stem = f"disp_{row['box']}_{row['field']}"
        for pair in PAIRS:
            entry = row.get(pair)
            if not entry:
                continue
            for quantity, unit in (("east_km", "km"), ("north_km", "km"),
                                   ("distance_km", "km")):
                _add(f"{stem}_{pair}_{quantity}", entry[quantity]["mean"], unit)
                _add(f"{stem}_{pair}_{quantity}_sd", entry[quantity]["sd"], unit)
            _add(f"{stem}_{pair}_corr_zero", entry["corr_zero"]["mean"], "correlation")
            _add(f"{stem}_{pair}_corr_best", entry["corr_best"]["mean"], "correlation")
        for pair, cell in (row.get("minimum_distance_km") or {}).items():
            _add(f"{stem}_minimum_{pair}_km", cell.get("mean"), "km")
        for src, cell in (row.get("minimum_value_hpa") or {}).items():
            _add(f"{stem}_minimum_{src}_hpa", cell.get("mean"), "hPa")
    return records
