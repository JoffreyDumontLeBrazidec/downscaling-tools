"""MARS retrieval and predictions_*.nc assembly for PrepML output.

Retrieves PrepML predictions from MARS/FDB using an expver identifier,
loads truth/input from reference GRIBs, and assembles them into the
same predictions_YYYYMMDD_stepNNN.nc format as manual inference.
"""
from __future__ import annotations

import re
from typing import Any

_SURFACE_PARAMS = {
    "10u", "10v", "2d", "2t", "cp", "hcc", "lcc", "mcc",
    "msl", "skt", "sp", "ssrd", "strd", "tcc", "tcw", "tp",
}

_PL_RE = re.compile(r"^([a-z]+)_(\d+)$")


def weather_state_to_mars(state: str) -> dict[str, Any]:
    """Map a weather state name to MARS request parameters.

    Surface states (e.g. '2t', '10u') -> {'param': '2t', 'levtype': 'sfc'}
    Pressure-level states (e.g. 'z_500') -> {'param': 'z', 'levtype': 'pl', 'level': 500}
    """
    if state in _SURFACE_PARAMS:
        return {"param": state, "levtype": "sfc"}

    m = _PL_RE.match(state)
    if m:
        return {"param": m.group(1), "levtype": "pl", "level": int(m.group(2))}

    raise ValueError(
        f"Unknown weather state '{state}'. "
        f"Expected a surface param ({sorted(_SURFACE_PARAMS)}) "
        f"or a pressure-level param like 'z_500', 't_850'."
    )


def group_weather_states_for_mars(
    states: list[str],
) -> dict[str, dict[str, list]]:
    """Group weather states into MARS request groups by levtype.

    Returns:
        {"sfc": {"params": [...]}, "pl": {"params": [...], "levels": [...]}}
    """
    groups: dict[str, dict[str, list]] = {
        "sfc": {"params": []},
        "pl": {"params": [], "levels": []},
    }
    seen_pl: set[str] = set()
    for state in states:
        mapped = weather_state_to_mars(state)
        levtype = mapped["levtype"]
        if levtype == "sfc":
            groups["sfc"]["params"].append(mapped["param"])
        else:
            param = mapped["param"]
            if param not in seen_pl:
                groups["pl"]["params"].append(param)
                seen_pl.add(param)
            level = mapped["level"]
            if level not in groups["pl"]["levels"]:
                groups["pl"]["levels"].append(level)
    return groups
