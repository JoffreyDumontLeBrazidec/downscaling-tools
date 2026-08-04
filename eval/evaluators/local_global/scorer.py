"""Scorer for local/global parity diagnostics."""
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
    summary_path = Path(results_dir) / "local_global_parity.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text())
    records: list[dict[str, Any]] = []
    for metric, value in sorted((payload.get("headline") or {}).items()):
        unit = "count" if metric == "files" else "native"
        records.append({"metric": f"local_global_{metric}", "value": value, "unit": unit})
    return records
