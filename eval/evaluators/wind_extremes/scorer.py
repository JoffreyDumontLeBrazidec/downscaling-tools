"""Metric records for the wind-extreme evaluator.

One record per (box, source, statistic), named
``wx_{box}_{statistic}_{model|truth|input}``, plus the model-minus-truth
differences that carry the verdict:

* ``wx_{box}_retention{R}_{source}``  maximum of the R-km disk average divided
  by the raw maximum, averaged over the (file, member) samples;
* ``wx_{box}_retention{R}_delta``     model minus truth of that ratio, and
  ``_sd`` its scatter over samples, which is the null the difference must beat;
* ``wx_{box}_peak_{source}``          the raw maximum wind speed in m/s;
* ``wx_{box}_patch90_{source}``       size in grid points of the connected patch
  above 90 percent of the maximum that contains the maximum;
* ``wx_{box}_peakdisp_{pair}``        great-circle distance in km between the
  wind maxima of two sources.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runner import SOURCES


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    path = Path(results_dir) / "wind_extremes.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    records: list[dict[str, Any]] = []

    def _add(metric: str, value, unit: str) -> None:
        if value is not None:
            records.append({"metric": metric, "value": value, "unit": unit})

    for row in payload.get("aggregate", []):
        stem = f"wx_{row['box']}"
        for source in SOURCES:
            entry = row.get(source) or {}
            _add(f"{stem}_peak_{source}", (entry.get("peak") or {}).get("mean"), "m/s")
            _add(f"{stem}_patch90_{source}",
                 (entry.get("patch_points_90pct") or {}).get("mean"), "grid_points")
            _add(f"{stem}_patcharea90_{source}",
                 (entry.get("patch_area_90pct_km2") or {}).get("mean"), "km2")
            for radius, cell in (entry.get("retention") or {}).items():
                _add(f"{stem}_retention{radius}_{source}", cell.get("mean"), "ratio")
        delta = row.get("model_minus_truth") or {}
        for radius, cell in (delta.get("retention") or {}).items():
            _add(f"{stem}_retention{radius}_delta", cell.get("mean"), "ratio")
            _add(f"{stem}_retention{radius}_sd", cell.get("sd"), "ratio")
        _add(f"{stem}_peak_delta", (delta.get("peak") or {}).get("mean"), "m/s")
        for pair, cell in (row.get("peak_displacement_km") or {}).items():
            _add(f"{stem}_peakdisp_{pair}", cell.get("mean"), "km")
            _add(f"{stem}_peakdisp_{pair}_sd", cell.get("sd"), "km")
    return records
