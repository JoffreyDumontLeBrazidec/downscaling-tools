"""Band-level scores for the spectra-coherence evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    path = Path(results_dir) / "coherence.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())

    records: list[dict[str, Any]] = []
    for row in payload.get("band_summary", []):
        stem = "coh_{state}_{band}".format(state=row["state"], band=row["band"])
        records.append({
            "metric": stem + "_amplitude_ratio",
            "value": row["amplitude_ratio"],
            "unit": "ratio",
        })
        records.append({
            "metric": stem + "_coherence",
            "value": row["coherence"],
            "unit": "correlation",
        })
        records.append({
            "metric": stem + "_error_floor",
            "value": row["error_floor_phase_only"],
            "unit": "normalised",
        })
        if "interp_coherence" in row:
            records.append({
                "metric": stem + "_interp_coherence",
                "value": row["interp_coherence"],
                "unit": "correlation",
            })
    return records
