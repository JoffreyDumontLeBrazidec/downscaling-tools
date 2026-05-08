"""Generate PrepML inference YAML from lane config + checkpoint."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _dates_to_range(dates: list[str]) -> dict[str, Any]:
    """Convert a list of YYYYMMDD date strings to PrepML start/end/frequency."""
    parsed = sorted(datetime.strptime(d, "%Y%m%d") for d in dates)
    return {
        "start": parsed[0].strftime("%Y-%m-%d"),
        "end": parsed[-1].strftime("%Y-%m-%d"),
        "frequency": 24,
    }


def _members_to_loop(members: list[int]) -> str:
    """Convert member list to PrepML loop syntax (e.g. '1/to/10')."""
    members_sorted = sorted(members)
    if members_sorted == list(range(members_sorted[0], members_sorted[-1] + 1)):
        return f"{members_sorted[0]}/to/{members_sorted[-1]}"
    return "/".join(str(m) for m in members_sorted)


def _steps_to_lead_time(steps: list[int]) -> str:
    """Convert forecast step list to the PrepML lead_time string."""
    if not steps:
        raise ValueError("predict.steps must contain at least one forecast step")
    return f"{max(int(step) for step in steps)}h"


def generate_prepml_config(
    *,
    lane_config: dict,
    checkpoint_path: str,
    runner_override: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Generate a PrepML inference config dict from lane config + checkpoint.

    The returned dict can be written to YAML and passed to `prepml inference`.
    """
    predict = lane_config["predict"]
    prepml = lane_config["prepml"]

    venv = runner_override or prepml["venv"]
    lead_time = _steps_to_lead_time(predict["steps"])

    config: dict[str, Any] = {
        "description": description or f"Generated from eval.cli for {Path(checkpoint_path).stem}",
        "dates": _dates_to_range(predict["dates"]),
        "ensemble": {
            "loop": {
                "number": _members_to_loop(predict["members"]),
            },
        },
        "input": {
            "class": prepml["input"]["class"],
            "stream": prepml["input"]["stream"],
            "type": prepml["input"]["type"],
            "number": "{number}",
            "grid": prepml["input"]["grid"],
        },
        "output": {
            "class": prepml["output"]["class"],
            "stream": prepml["output"]["stream"],
            "type": prepml["output"]["type"],
            "number": "{member_number}",
        },
        "runner": {
            "name": prepml["runner"],
            "venv": venv,
        },
        "model": {
            "name": "anemoi",
            "runner": "downscaling",
            "checkpoint": str(checkpoint_path),
            "lead_time": lead_time,
            "development_hacks": {
                "time_step": prepml["time_step"],
                "output_template": prepml["output_template"],
                "constant_high_res_forcings_npz": prepml["forcings_npz"],
                "constant_high_res_forcings": list(prepml["constant_high_res_forcings"]),
                "high_res_input": list(prepml["high_res_input"]),
                "extra_args": dict(predict.get("sampler", {})),
            },
        },
        "platform": {
            "flavours": {
                "gpu": {
                    "submit_arguments": {
                        "time": prepml["platform"]["gpu"]["time"],
                    },
                    "late": f"-c +{prepml['platform']['gpu']['time'].split('-', 1)[-1]}",
                },
            },
        },
        "evaluation": False,
    }
    return config


def write_prepml_config(config: dict[str, Any], path: Path) -> Path:
    """Write PrepML config dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return path
