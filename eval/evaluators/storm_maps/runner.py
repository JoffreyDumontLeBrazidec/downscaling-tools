"""storm_maps evaluator runner — thin wrapper over eval._backends.storm_maps.render.

Mirrors the region_plot runner signature so eval.cli can dispatch it. Reads the event/storm
box from the lane's ``tc`` config when present, else falls back to the tc_atlantic_mdr_west box.
"""
from __future__ import annotations

import logging
from pathlib import Path

from eval._backends.storm_maps.render import render

LOG = logging.getLogger(__name__)

# tc_atlantic_mdr_west defaults (lat0,lat1,lon0,lon1)
_DEFAULT_BOX = (5.0, 35.0, -100.0, -40.0)
_DEFAULT_STORM = (10.0, 35.0, -100.0, -80.0)


def _box_from_lane(lane_config: dict):
    """Best-effort event box + name from lane tc config; defaults otherwise."""
    box, storm, name = _DEFAULT_BOX, _DEFAULT_STORM, "storm"
    tc = lane_config.get("tc", {}) if isinstance(lane_config, dict) else {}
    if isinstance(tc, dict):
        events = tc.get("events")
        if events:
            name = events[0] if isinstance(events, (list, tuple)) else str(events)
        # optional explicit box override under tc.storm_box / tc.box
        for key in ("storm_box", "box"):
            b = tc.get(key)
            if isinstance(b, (list, tuple)) and len(b) == 4:
                storm = tuple(float(x) for x in b)
    return box, storm, name


def run(
    predictions_dir,
    lane_config,
    eval_config,
    *,
    output_dir=None,
    overwrite: bool = False,
    checkpoint=None,
    **kwargs,
) -> Path:
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "storm_maps"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"storm_maps output exists (use --overwrite): {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    box, storm, name = _box_from_lane(lane_config)
    step = str(kwargs.get("step", "072"))
    LOG.info("storm_maps: predictions=%s event_box=%s storm_box=%s step=%s", predictions_dir, box, storm, step)
    return render(
        predictions_dir, output_dir,
        event_box=box, event_name=name, step=step, storm_box=storm,
    )
