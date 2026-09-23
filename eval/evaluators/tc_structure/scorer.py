"""Metric records for tc_structure: raw physical quantities, no score, no ratio.

One record per (event, score, field, lead band) named
``tcs_<event>_<score>_<field>_<band>`` holding the mean over members and valid
times, plus ``..._se`` holding its date-clustered bootstrap standard error.
``tcs_<event>_windpressure_{slope,scatter}_<field>_<band>`` give the wind-pressure
relation. Only the bands 24-48, 96-120 and all are emitted as metrics; every band
is in summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UNITS = {
    "pmin_hpa": "hPa", "displacement_km": "km", "rmw_km": "km", "vmax_tan_ms": "m/s",
    "maxwind300_ms": "m/s", "r34_km": "km", "r50_km": "km", "zeta50_s": "1/s",
    "zeta100_s": "1/s", "zeta200_s": "1/s", "asym_rmw": "fraction",
}
METRIC_BANDS = ("24-48", "96-120", "all")


def score(results_dir, lane_config: dict, eval_config: dict, **kwargs) -> list[dict[str, Any]]:
    path = Path(results_dir) / "summary.json"
    if not path.exists():
        return []
    summ = json.loads(path.read_text())
    recs: list[dict[str, Any]] = []

    def add(metric, value, unit):
        if value is not None:
            recs.append({"metric": metric, "value": value, "unit": unit})

    band_key = {"24-48": "24_48", "96-120": "96_120", "all": "all"}
    for ev, bands in summ.get("events", {}).items():
        for band in METRIC_BANDS:
            b = bands.get(band)
            if not b:
                continue
            bk = band_key[band]
            for field, scores in b.get("fields", {}).items():
                for s, st in scores.items():
                    add(f"tcs_{ev}_{s}_{field}_{bk}", st.get("value"), UNITS.get(s, ""))
                    add(f"tcs_{ev}_{s}_{field}_{bk}_se", st.get("se"), UNITS.get(s, ""))
            for field, wp in b.get("wind_pressure", {}).items():
                add(f"tcs_{ev}_windpressure_slope_{field}_{bk}", wp["slope"].get("value"), "m/s/hPa")
                add(f"tcs_{ev}_windpressure_scatter_{field}_{bk}", wp["scatter"].get("value"), "m/s")
    return recs
