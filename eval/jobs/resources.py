"""Resource profile resolution for the evaluation pipeline.

Resolves per-stage SLURM resource requirements by merging host scheduler
defaults with lane-level ``resource_profiles`` overrides.
"""

from __future__ import annotations

import re
from typing import Any


_VALID_QOS = {"nf", "ng"}
_TIME_RE = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")
_MEM_RE = re.compile(r"^(\d+[GMKT]|0)$")
_PREDICT_DEFAULT_TIME = "48:00:00"
_EVALUATE_DEFAULT_TIME = "48:00:00"
_FULL_CONTRACT_PREDICT_TIME_FLOOR = _PREDICT_DEFAULT_TIME


def validate_resource_profile(profile: dict[str, Any]) -> list[str]:
    """Validate a single resource profile dict.  Returns list of error strings."""
    errors: list[str] = []
    if "qos" in profile and profile["qos"] not in _VALID_QOS:
        errors.append(f"qos must be one of {sorted(_VALID_QOS)}, got {profile['qos']!r}")
    if "time" in profile and not _TIME_RE.match(str(profile["time"])):
        errors.append(f"time must match HH:MM:SS, got {profile['time']!r}")
    if "mem" in profile and not _MEM_RE.match(str(profile["mem"])):
        errors.append(f"mem must match \\d+[GMKT] or '0', got {profile['mem']!r}")
    if "cpus" in profile:
        if not isinstance(profile["cpus"], int) or profile["cpus"] < 1:
            errors.append(f"cpus must be int > 0, got {profile['cpus']!r}")
    if "gpus" in profile:
        if not isinstance(profile["gpus"], int) or profile["gpus"] < 0:
            errors.append(f"gpus must be int >= 0, got {profile['gpus']!r}")
    if "ntasks_per_node" in profile:
        if not isinstance(profile["ntasks_per_node"], int) or profile["ntasks_per_node"] < 1:
            errors.append(f"ntasks_per_node must be int > 0, got {profile['ntasks_per_node']!r}")
    return errors


def _time_to_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return (hours * 3600) + (minutes * 60) + seconds


def _default_time_for_stage(stage: str, scheduler: dict[str, Any]) -> str:
    if stage == "predict":
        return scheduler.get("default_predict_time", _PREDICT_DEFAULT_TIME)
    if stage == "evaluate":
        # Evaluate is a LONG stage: quaver / tc / spectra_ecmwf do heavy MARS
        # reads + metview and routinely run many hours over a full month.
        # Default it to the same conservative 48h wall as predict so a lane
        # WITHOUT an explicit evaluate resource_profile can never inherit the
        # short host default_time (4h) and get TIME_LIMIT-killed mid-run.
        # (2026-07-06: the j7xn quaver month job died at an 8h wall; nf/ng
        # both permit 48h.)  Host may still tune this via default_evaluate_time.
        return scheduler.get("default_evaluate_time", _EVALUATE_DEFAULT_TIME)
    return scheduler.get("default_time", "04:00:00")


def _apply_predict_time_floor(lane_config: dict[str, Any], resources: dict[str, Any]) -> None:
    """Protect full production predict jobs from underspecified walltimes.

    The canonical full-eval contract is 5 dates x 5 lead steps x 10 members
    = 250 predictions. A July 2026 pristine eval lane accidentally overrode
    that contract down to a 2-hour AC predict budget and predict jobs timed
    out with `TIME_LIMIT`. Once a lane is running the full 250 prediction
    contract, force a very conservative walltime even if a custom lane
    override drifts lower than the safe production floor.
    """
    predict = lane_config.get("predict") or {}
    bundle_count = (
        len(predict.get("members") or [])
        * len(predict.get("dates") or [])
        * len(predict.get("steps") or [])
    )
    if bundle_count < 250:
        return

    if _time_to_seconds(str(resources["time"])) < _time_to_seconds(_FULL_CONTRACT_PREDICT_TIME_FLOOR):
        resources["time"] = _FULL_CONTRACT_PREDICT_TIME_FLOOR


def resolve_resources(
    lane_config: dict,
    host_config: dict,
    stage: str,
    evaluator: str | None = None,
) -> dict[str, Any]:
    """Resolve SLURM resource requirements for a pipeline stage.

    Parameters
    ----------
    lane_config : dict
        Lane configuration (from ``load_lane``).
    host_config : dict
        Host configuration (from ``load_host``).
    stage : str
        Pipeline stage: ``"predict"``, ``"evaluate"``, or ``"scoreboard"``.
    evaluator : str or None
        Evaluator name (e.g. ``"tc"``, ``"spectra"``).  Only used when
        *stage* is ``"evaluate"`` to look up evaluator-specific overrides.

    Returns
    -------
    dict
        Resource dict with keys: qos, time, mem, cpus, gpus, ntasks_per_node.
    """
    scheduler = host_config["scheduler"]

    # 1. Start with host scheduler defaults
    resources: dict[str, Any] = {
        "qos": scheduler["qos"],
        "time": _default_time_for_stage(stage, scheduler),
        "mem": scheduler.get("default_mem", "64G"),
        "cpus": scheduler.get("default_cpus", 16),
        "gpus": 0,
        "ntasks_per_node": 1,
    }

    profiles = lane_config.get("resource_profiles", {})

    # 2. Overlay stage-level profile
    if stage in profiles:
        resources.update(profiles[stage])

    # 3. Overlay evaluator-specific profile
    if evaluator and f"evaluate_{evaluator}" in profiles:
        resources.update(profiles[f"evaluate_{evaluator}"])

    # 4. Host QOS fixup (last step)
    if resources["gpus"] == 0:
        resources["qos"] = scheduler["qos"]
    else:
        resources["qos"] = "ng"

    if stage == "predict":
        _apply_predict_time_floor(lane_config, resources)

    return resources
