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
    if len(members_sorted) == 1:
        return str(members_sorted[0])
    if members_sorted == list(range(members_sorted[0], members_sorted[-1] + 1)):
        return f"{members_sorted[0]}/to/{members_sorted[-1]}"
    return "/".join(str(m) for m in members_sorted)


def _steps_to_lead_time(steps: list[int]) -> str:
    """Convert forecast step list to the PrepML lead_time string."""
    if not steps:
        raise ValueError("predict.steps must contain at least one forecast step")
    return f"{max(int(step) for step in steps)}h"


def _num_gpus_per_model(predict: dict) -> int:
    """Return validated GPUs per model for PrepML model-parallel inference."""
    value = predict.get("num_gpus_per_model", 1)
    try:
        num_gpus = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"predict.num_gpus_per_model must be an integer, got {value!r}") from exc
    if num_gpus < 1:
        raise ValueError(f"predict.num_gpus_per_model must be >= 1, got {num_gpus}")
    return num_gpus


def _model_runner_config(
    num_gpus_per_model: int, base_runner: str = "downscaling"
) -> str | dict[str, dict[str, str]]:
    """Build the PrepML/anemoi-inference runner stanza.

    ``base_runner`` selects the anemoi-inference runner the prepml ``anemoi`` task
    instantiates. Default ``"downscaling"`` = the fork DS 2-tensor runner.
    ``"downscaling_unified"`` = the UnifiedDownscalingRunner driving the
    multi-ds-unified dict-batch checkpoints (registered via runner.venv
    sitecustomize). Read from lane ``prepml.base_runner``.
    """
    if num_gpus_per_model == 1:
        return base_runner
    return {"parallel": {"base_runner": base_runner}}


def _gpu_submit_arguments(prepml: dict, num_gpus_per_model: int) -> dict[str, Any]:
    """Build GPU submit arguments, preserving lane overrides."""
    gpu_cfg = prepml["platform"]["gpu"]
    submit_arguments = dict(gpu_cfg.get("submit_arguments") or {})
    submit_arguments["time"] = gpu_cfg["time"]

    if num_gpus_per_model > 1:
        submit_arguments.setdefault("gpus_per_node", num_gpus_per_model)
        raw_pragma = submit_arguments.get("RAW_PRAGMA")
        if raw_pragma is None:
            raw_pragmas: list[str] = []
        elif isinstance(raw_pragma, str):
            raw_pragmas = [raw_pragma]
        else:
            raw_pragmas = list(raw_pragma)
        if not any("gres=gpu" in pragma or "gpus-per-node" in pragma for pragma in raw_pragmas):
            raw_pragmas.insert(0, f"#SBATCH --gres=gpu:{num_gpus_per_model}")
        if not any("--nice" in pragma for pragma in raw_pragmas):
            raw_pragmas.append("#SBATCH --nice=100")
        submit_arguments["RAW_PRAGMA"] = raw_pragmas

    return submit_arguments


def _input_step_request(predict_steps: list[int], prepml_input: dict, time_step: str) -> str | None:
    """Build an optional PrepML input step override.

    Some downscaling checkpoints are trained from coarse forecast fields whose
    first valid input is offset from the forecast base time, e.g. O48 ENFO +6h.
    PrepML passes this as an `--extra step=...` override to anemoi-inference
    retrieve.
    """
    offset = prepml_input.get("step_offset_hours")
    if offset is None:
        return None

    if not predict_steps:
        raise ValueError("predict.steps must contain at least one forecast step")

    try:
        step_hours = int(str(time_step).removesuffix("h"))
    except ValueError as exc:
        raise ValueError(f"prepml.time_step must be an hour duration, got {time_step!r}") from exc

    first_step = int(offset)
    last_step = max(int(step) for step in predict_steps) + first_step
    return f"{first_step}/to/{last_step}/by/{step_hours}"


def _gpu_resources(prepml: dict, num_gpus_per_model: int) -> dict[str, Any]:
    """Build the prepml 0.125+ ``resources.inference.gpu`` block.

    This is the schema current prepml ACTUALLY honors for GPU geometry. The older
    ``platform.flavours.gpu.submit_arguments`` block is IGNORED by prepml for GPU count
    (prepml hardcodes ``gres=gpu:1`` in its platform defaults and drops the config
    override), which silently capped unified multi-GPU inference at 1 GPU -> the parallel
    runner has no rank -> ``global_rank`` crash. ``tasks_per_node == gpus_per_node`` so
    SLURM_NTASKS matches the shard count and the wrap shim srun-launches one rank per GPU
    (world_size == num_gpus_per_model).
    """
    gpu_cfg = prepml.get("platform", {}).get("gpu", {})
    return {
        "total_nodes": 1,
        "tasks_per_node": num_gpus_per_model,
        "gpus_per_node": num_gpus_per_model,
        "cpus_per_task": int(gpu_cfg.get("cpus_per_task", 32)),
        "memory_per_node": str(gpu_cfg.get("memory_per_node", "64G")),
    }


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
    num_gpus_per_model = _num_gpus_per_model(predict)
    lead_time = _steps_to_lead_time(predict["steps"])
    input_step = _input_step_request(predict["steps"], prepml["input"], prepml["time_step"])

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
            "version": "auto",
            "model": prepml.get("runner_model", "aifs-single-mse"),
        },
        "model": {
            "name": "anemoi",
            "runner": _model_runner_config(
                num_gpus_per_model, prepml.get("base_runner", "downscaling")
            ),
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
            "name": prepml.get("platform", {}).get("name", "atos"),
            "flavours": {
                "gpu": {
                    "submit_arguments": _gpu_submit_arguments(prepml, num_gpus_per_model),
                    "late": f"-c +{prepml['platform']['gpu']['time'].split('-', 1)[-1]}",
                },
            },
        },
        "resources": {
            "inference": {"gpu": _gpu_resources(prepml, num_gpus_per_model)},
        },
        "evaluation": False,
    }
    predict_env = dict(predict.get("env") or {})
    if predict_env:
        config["model"]["env"] = predict_env
    if num_gpus_per_model > 1:
        config["model"]["world_size"] = num_gpus_per_model
    if input_step is not None:
        config["input"]["step"] = input_step
    return config


def write_prepml_config(config: dict[str, Any], path: Path) -> Path:
    """Write PrepML config dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return path
