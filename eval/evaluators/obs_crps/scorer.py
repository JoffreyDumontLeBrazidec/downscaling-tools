"""Observation CRPS evaluator — score phase.

Surfaces a compact set of numbers so the result can reach a scoreboard rather
than living only in a plot.  The headline is the fair CRPS at a few reference
lead times, per parameter, on the northern hemisphere, which is the cell a
quaver surface scorecard is normally read from.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Lead times worth carrying as headline numbers: one short, one medium, one long.
_HEADLINE_STEPS = (24, 120, 240)
_HEADLINE_DOMAIN = "n.hem"


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    results_dir = Path(results_dir)
    if (results_dir / "skipped.json").exists():
        return []

    summary_path = results_dir / "summary_by_lead.csv"
    if not summary_path.exists():
        return []

    summary = pd.read_csv(summary_path)
    domain = (eval_config or {}).get("headline_domain", _HEADLINE_DOMAIN)
    steps = (eval_config or {}).get("headline_steps", _HEADLINE_STEPS)

    records: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        if row["domain"] != domain or int(row["step"]) not in set(int(s) for s in steps):
            continue
        records.append({
            "metric": f"obs_fcrps_{row['parameter']}_{domain}_step{int(row['step'])}",
            "value": float(row["fcrps"]),
            "unit": "native",
        })

    params_path = results_dir / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
        records.append({
            "metric": "obs_crps_ndates",
            "value": len(params.get("dates", [])),
            "unit": "count",
        })
    records.append({
        "metric": "obs_crps_rows",
        "value": int(len(summary)),
        "unit": "count",
    })
    return records
