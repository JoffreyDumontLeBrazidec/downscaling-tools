"""Scorer for the ML-vs-ENFO spread proxy."""
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
    """Return compact metrics from spread_proxy_summary.json."""
    results_dir = Path(results_dir)
    summary_path = results_dir / "spread_proxy_summary.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text())
    records: list[dict[str, Any]] = []
    for metric, value in sorted(payload.get("headline_metrics", {}).items()):
        records.append({"metric": metric, "value": value, "unit": "ratio"})
    records.append({"metric": "spread_proxy_rows", "value": payload.get("n_rows", 0), "unit": "count"})
    records.append({"metric": "spread_proxy_skipped", "value": payload.get("skipped_count", 0), "unit": "count"})
    return records
