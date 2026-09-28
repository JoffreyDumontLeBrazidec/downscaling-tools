"""Observation CRPS evaluator — score phase.

Surfaces a compact set of numbers so the result can reach a scoreboard rather
than living only in a plot. The headline is the experiment's fair CRPS at a few
reference lead times, per parameter, on the northern hemisphere, which is the
cell a quaver surface scorecard is normally read from. Alongside it, and more
useful for a decision, is the improvement over the coarse input, expressed as a
percentage of the input's own error: that is what the downscaling added.
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
    if "curve" not in summary:
        summary["curve"] = "experiment"
    domain = (eval_config or {}).get("headline_domain", _HEADLINE_DOMAIN)
    steps = {int(s) for s in (eval_config or {}).get("headline_steps", _HEADLINE_STEPS)}

    here = summary[(summary["domain"] == domain) & (summary["step"].isin(steps))]
    experiment = here[here["curve"] == "experiment"]
    inputs = here[here["curve"] == "input"].set_index(["parameter", "step"])

    records: list[dict[str, Any]] = []
    for _, row in experiment.iterrows():
        parameter, step = row["parameter"], int(row["step"])
        records.append({
            "metric": f"obs_fcrps_{parameter}_{domain}_step{step}",
            "value": float(row["fcrps"]),
            "unit": "native",
        })
        key = (parameter, step)
        if key in inputs.index:
            reference = float(inputs.loc[key, "fcrps"])
            if reference > 0:
                records.append({
                    "metric": f"obs_fcrps_gain_vs_input_{parameter}_{domain}_step{step}",
                    "value": 100.0 * (reference - float(row["fcrps"])) / reference,
                    "unit": "percent",
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
            "metric": "obs_crps_ncurves",
            "value": len(params.get("curves", {})),
            "unit": "count",
        })
    records.append({
        "metric": "obs_crps_rows",
        "value": int(len(summary)),
        "unit": "count",
    })
    return records
