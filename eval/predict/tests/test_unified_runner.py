from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from . import load_predict_submodule


def _prediction_config(types_mod, *, runner_config: Path | None):
    return types_mod.PredictionConfig(
        checkpoint_path=Path("/checkpoints/last.ckpt"),
        input_dir=Path("/input"),
        output_dir=Path("/output"),
        inference_backend="unified",
        runner_config=runner_config,
    )


def _install_fake_anemoi(monkeypatch, calls: list[object]):
    anemoi = types.ModuleType("anemoi")
    utils = types.ModuleType("anemoi.utils")
    config = types.ModuleType("anemoi.utils.config")
    inference = types.ModuleType("anemoi.inference")
    runners = types.ModuleType("anemoi.inference.runners")

    class DotDict(dict):
        def __init__(self, value):
            calls.append(("dotdict", os.environ.get("UNIT_UNIFIED_ENV")))
            super().__init__(value)

    interface = types.SimpleNamespace(data_indices={"in_lres": object()})
    runner = types.SimpleNamespace(
        model=interface,
        device="cuda:2",
        model_comm_group="model-group",
        global_rank=3,
        local_rank=2,
        world_size=4,
    )

    def create_runner(raw):
        calls.append(("create_runner", raw))
        return runner

    config.DotDict = DotDict
    runners.create_runner = create_runner
    anemoi.utils = utils
    utils.config = config
    anemoi.inference = inference
    inference.runners = runners

    monkeypatch.setitem(sys.modules, "anemoi", anemoi)
    monkeypatch.setitem(sys.modules, "anemoi.utils", utils)
    monkeypatch.setitem(sys.modules, "anemoi.utils.config", config)
    monkeypatch.setitem(sys.modules, "anemoi.inference", inference)
    monkeypatch.setitem(sys.modules, "anemoi.inference.runners", runners)
    monkeypatch.setitem(sys.modules, "downscaling_unified_runner", types.ModuleType("downscaling_unified_runner"))
    return interface


def test_unified_runner_requires_existing_config(monkeypatch, tmp_path: Path):
    types_mod = load_predict_submodule(monkeypatch, "types")
    mod = load_predict_submodule(monkeypatch, "unified_runner")
    config = _prediction_config(types_mod, runner_config=tmp_path / "missing.yaml")

    with pytest.raises(SystemExit, match="--runner-config.*does not exist"):
        mod.load_unified_runner(config)


def test_unified_runner_applies_env_before_constructing_runner(monkeypatch, tmp_path: Path):
    types_mod = load_predict_submodule(monkeypatch, "types")
    mod = load_predict_submodule(monkeypatch, "unified_runner")
    runner_config = tmp_path / "anemoi-config.yaml"
    runner_config.write_text(
        """env:\n  UNIT_UNIFIED_ENV: configured\ndevelopment_hacks:\n  extra_args:\n    num_steps: 30\n    sigma_max: 10000.0\n""",
        encoding="utf-8",
    )
    monkeypatch.delenv("UNIT_UNIFIED_ENV", raising=False)
    calls: list[object] = []
    interface = _install_fake_anemoi(monkeypatch, calls)
    config = _prediction_config(types_mod, runner_config=runner_config)

    (
        inference_model,
        datamodule,
        extra_args,
        device,
        model_comm_group,
        global_rank,
        local_rank,
        world_size,
    ) = mod.load_unified_runner(config)

    assert calls[0] == ("dotdict", "configured")
    assert calls[1][0] == "create_runner"
    assert calls[1][1]["device"] == "cuda"
    assert calls[1][1]["allow_nans"] is None
    assert inference_model is interface
    assert datamodule.data_indices is interface.data_indices
    assert extra_args == {"num_steps": 30, "sigma_max": 10000.0}
    assert (device, model_comm_group, global_rank, local_rank, world_size) == (
        "cuda:2",
        "model-group",
        3,
        2,
        4,
    )
