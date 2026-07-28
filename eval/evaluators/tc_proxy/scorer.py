"""Reduce the per-instance TC table to the panel's metrics.

Two numbers per event and curve, deliberately:
  * ``eye_deepest``      -- the deepest single instance. This is what the campaign cares about,
    but on a cheap budget it is an instance lottery: it can swing many hPa while the underlying
    distribution does not move at all.
  * ``eye_casemin_mean`` -- mean over cases of the per-case ensemble minimum. Same physical
    quantity, averaged, so it moves only when the whole distribution moves.
Read them TOGETHER: a trend is only believable where both move the same way. Measured 2026-07-27,
see epics/training-diagnostics/metric-skill-gap/in-progress/20260727_ladder_tc_proxy_studyA.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

# (metric suffix, table key, reduction over the (case, member) array, unit)
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
    """Return ``tcproxy_<event>[_<src>]_<key>`` records.

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
                metric = (
                    f"tcproxy_{event}_{src_tag}_{suffix}" if src_tag
                    else f"tcproxy_{event}_{suffix}"
                )
                records.append({"metric": metric, "value": reduce_fn(arr), "unit": unit})
    return records
