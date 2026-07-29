"""Reduce the per-instance TC table to the dashboard's metrics.

GRID metrics (the ladder default), per event and per curve:
  * ``eye_deepest``      -- the deepest single instance. What the campaign cares about, but at
    ladder budgets it is an instance lottery: it can swing many hPa while the distribution does
    not move.
  * ``eye_casemin_mean`` -- the same quantity averaged over cases, so it moves only when the whole
    distribution moves.
  Read them TOGETHER: a trend is believable only where both move the same way.

DRAW metrics (the ultra-cheap tier), from N independent re-runs of one instance:
  * ``draws_eye_deepest`` -- deepest over the N draws. RANK-ONLY: it under-deepens by a roughly
    constant ~6.6 hPa, and the ranking works precisely because that offset is near-constant.
  * ``draws_eye_mean`` / ``draws_eye_sd`` / ``draws_n`` -- the draw distribution, so the reader can
    see whether the re-run spread is large enough to explain a rung-to-rung move.

Measured 2026-07-27/29; see
epics/training-diagnostics/metric-skill-gap/in-progress/20260727_ladder_tc_proxy_studyA.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

_REDUCTIONS = (
    ("eye_deepest", "eye", lambda a: float(a.min()), "hPa"),
    ("eye_casemin_mean", "eye", lambda a: float(a.min(axis=1).mean()), "hPa"),
    ("wind_peak", "peak", lambda a: float(a.max()), "m/s"),
    ("wind_casemax_mean", "peak", lambda a: float(a.max(axis=1).mean()), "m/s"),
)


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    run_id: str = "",
    **kwargs,
) -> list[dict[str, Any]]:
    """Return ``tcproxy_<event>[_<src>]_<key>`` and ``tcproxy_<event>_draws_*`` records.

    ``src`` is empty for the model and ``enfo`` / ``eefo`` for the target and input curves, which
    are read out of the same prediction files and are therefore invariant across rungs of one
    profile — a free integrity check: if they drift, the support moved.
    """
    path = Path(results_dir) / "per_instance.json"
    if not path.exists():
        LOG.warning("tc_proxy: per_instance.json not found at %s", path)
        return []

    report = json.loads(path.read_text())
    records: list[dict[str, Any]] = []

    for event, payload in (report.get("events") or {}).items():
        for curve, table in (payload.get("tables") or {}).items():
            src_tag = "" if curve == "model" else curve
            for suffix, table_key, reduce_fn, unit in _REDUCTIONS:
                arr = np.asarray(table[table_key], dtype=np.float64)
                if arr.ndim != 2 or arr.size == 0:
                    continue
                metric = (f"tcproxy_{event}_{src_tag}_{suffix}" if src_tag
                          else f"tcproxy_{event}_{suffix}")
                records.append({"metric": metric, "value": reduce_fn(arr), "unit": unit})

    draws = report.get("draws") or []
    if draws:
        events = sorted({e for d in draws for e in (d.get("events") or {})})
        for event in events:
            per_draw = []
            for d in draws:
                tbl = ((d.get("events") or {}).get(event) or {}).get("tables", {}).get("model")
                if not tbl:
                    continue
                per_draw.append(float(np.asarray(tbl["eye"], dtype=np.float64).min()))
            if not per_draw:
                continue
            a = np.asarray(per_draw)
            records.append({"metric": f"tcproxy_{event}_draws_n", "value": float(a.size),
                            "unit": "count"})
            records.append({"metric": f"tcproxy_{event}_draws_eye_deepest",
                            "value": float(a.min()), "unit": "hPa"})
            records.append({"metric": f"tcproxy_{event}_draws_eye_mean",
                            "value": float(a.mean()), "unit": "hPa"})
            if a.size > 1:
                records.append({"metric": f"tcproxy_{event}_draws_eye_sd",
                                "value": float(a.std(ddof=1)), "unit": "hPa"})
    return records
