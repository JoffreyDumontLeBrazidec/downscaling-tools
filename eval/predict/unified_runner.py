"""Lazy loader for the production ``downscaling_unified`` inference runner."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from .types import PredictionConfig


def _rank_metadata() -> tuple[int, int, int]:
    """Read launcher rank metadata without importing model/runtime modules."""

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    return global_rank, local_rank, world_size


def _load_raw_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"--runner-config does not exist or is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SystemExit(f"Could not parse --runner-config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"--runner-config must contain a YAML mapping: {path}")
    return raw


def _normalize_runner_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Supply the defaults PrepML normally injects before invoking Anemoi."""

    defaults: dict[str, Any] = {
        "date": None,
        "device": "cuda",
        "precision": "float32",
        "allow_nans": None,
        "verbosity": 0,
        "world_size": 1,
        "use_grib_paramid": False,
        "patch_metadata": {},
        "output_frequency": None,
        "trace_path": None,
        "use_profiler": False,
        "description": None,
        "pre_processors": [],
        "post_processors": None,
        "forcings": None,
        "debugging_info": {},
        "env": {},
    }
    for key, value in defaults.items():
        raw_config.setdefault(key, value)
    return raw_config


def _apply_runner_environment(raw_config: dict[str, Any]) -> None:
    """Apply runner configuration before importing Anemoi/model implementation modules."""

    env = raw_config.get("env", {})
    if not isinstance(env, dict):
        raise SystemExit("runner config field 'env' must be a mapping when present.")
    for key, value in env.items():
        if value is None:
            continue
        os.environ[str(key)] = str(value)


def load_unified_runner(
    config: PredictionConfig,
) -> tuple[object, object, dict, str, object | None, int, int, int]:
    """Create the registered unified runner and adapt it to the bundle-loop contract."""

    if config.runner_config is None:
        raise SystemExit("--inference-backend unified requires --runner-config.")
    runner_config = Path(config.runner_config).expanduser()
    raw_config = _normalize_runner_config(_load_raw_config(runner_config))
    _apply_runner_environment(raw_config)

    # Importing this module registers ``downscaling_unified`` with Anemoi's runner registry.
    importlib.import_module("downscaling_unified_runner")
    from anemoi.inference.runners import create_runner
    from anemoi.utils.config import DotDict

    runner = create_runner(DotDict(raw_config))
    inference_model = runner.model
    data_indices = getattr(inference_model, "data_indices", None)
    if data_indices is None:
        raise SystemExit(
            "Unified runner model does not expose data_indices required for bundle inference."
        )

    development_hacks = raw_config.get("development_hacks", {})
    if not isinstance(development_hacks, dict):
        raise SystemExit("runner config field 'development_hacks' must be a mapping when present.")
    extra_args = development_hacks.get("extra_args", {})
    if not isinstance(extra_args, dict):
        raise SystemExit("runner config field 'development_hacks.extra_args' must be a mapping when present.")

    fallback_global_rank, fallback_local_rank, fallback_world_size = _rank_metadata()
    device = str(getattr(runner, "device", config.device))
    global_rank = int(getattr(runner, "global_rank", fallback_global_rank))
    local_rank = int(getattr(runner, "local_rank", fallback_local_rank))
    world_size = int(getattr(runner, "world_size", fallback_world_size))
    model_comm_group = getattr(runner, "model_comm_group", None)
    # Keep the runner alive for the complete bundle loop.  ParallelRunnerMixin tears
    # down its process group in ``__del__``; returning only ``runner.model`` would
    # leave the interface holding a stale model communication group on first forward.
    datamodule = SimpleNamespace(data_indices=data_indices, _unified_runner=runner)

    print(
        "Unified runner initialized "
        f"config={runner_config} device={device} rank={global_rank}/{world_size}"
    )
    return (
        inference_model,
        datamodule,
        dict(extra_args),
        device,
        model_comm_group,
        global_rank,
        local_rank,
        world_size,
    )
