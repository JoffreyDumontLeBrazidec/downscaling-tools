"""storm_maps evaluator runner — thin wrapper over eval.evaluators.storm_maps.core.render.

Mirrors the region_plot runner signature so eval.cli can dispatch it. The event box, the storm
search box and the lead times can be set in the lane's ``storm_maps:`` block (see the package
docstring, ``python -m eval.cli describe storm_maps``); without that block the storm search box
comes from the lane's ``tc`` config when present, and every value falls back to the
tc_atlantic_mdr_west defaults below, so lanes that set nothing render exactly as before.
"""
from __future__ import annotations

import logging
from pathlib import Path

from eval.evaluators.storm_maps.core.render import render

LOG = logging.getLogger(__name__)

# tc_atlantic_mdr_west defaults (lat0,lat1,lon0,lon1)
_DEFAULT_BOX = (5.0, 35.0, -100.0, -40.0)
_DEFAULT_STORM = (10.0, 35.0, -100.0, -80.0)
_DEFAULT_STEPS = ("072",)


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


def _parse_box(value, key: str):
    """Turn a configured box into (lat0, lat1, lon0, lon1), or raise a clear error.

    Accepts a mapping with lat_min, lat_max, lon_min and lon_max, or a list of four numbers
    in the order lat0, lat1, lon0, lon1 (the same order as the ``tc`` block's ``box``).
    """
    try:
        if isinstance(value, dict):
            box = tuple(float(value[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max"))
        else:
            box = tuple(float(x) for x in value)
            if len(box) != 4:
                raise ValueError("expected four numbers")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"storm_maps.{key} must be a mapping with lat_min, lat_max, lon_min, lon_max "
            f"or a list [lat0, lat1, lon0, lon1]; got {value!r} ({exc})") from exc
    if not (box[0] < box[1] and box[2] < box[3]):
        raise ValueError(f"storm_maps.{key} needs lat_min < lat_max and lon_min < lon_max; got {value!r}")
    return box


def _parse_steps(value) -> list[str]:
    """Normalise configured lead times to the zero-padded strings used in the file names."""
    if isinstance(value, (str, int)):
        value = [value]
    try:
        steps = [f"{int(s):03d}" for s in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"storm_maps.steps must be a list of lead times in hours; got {value!r}") from exc
    if not steps:
        raise ValueError("storm_maps.steps must not be empty")
    return list(dict.fromkeys(steps))


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
    eval_config = eval_config or {}
    if eval_config.get("box") is not None:
        box = _parse_box(eval_config.get("box"), "box")
    if eval_config.get("storm_box") is not None:
        storm = _parse_box(eval_config.get("storm_box"), "storm_box")
    if kwargs.get("step") is not None:
        steps = _parse_steps(kwargs["step"])
    elif eval_config.get("steps") is not None:
        steps = _parse_steps(eval_config.get("steps"))
    else:
        steps = list(_DEFAULT_STEPS)
    LOG.info("storm_maps: predictions=%s event_box=%s storm_box=%s steps=%s", predictions_dir, box, storm, steps)
    if len(steps) == 1:
        return render(
            predictions_dir, output_dir,
            event_box=box, event_name=name, step=steps[0], storm_box=storm,
        )
    # Several lead times: the backend writes fixed file names, so each gets its own folder.
    for step in steps:
        step_dir = output_dir / f"step{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        render(
            predictions_dir, step_dir,
            event_box=box, event_name=name, step=step, storm_box=storm,
        )
    return output_dir
