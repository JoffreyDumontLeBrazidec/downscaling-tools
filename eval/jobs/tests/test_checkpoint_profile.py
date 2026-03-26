from __future__ import annotations

import pytest

from eval.jobs import checkpoint_profile as mod


def _cfg_for_pair(lres: int, hres: int, *, stack: str) -> dict:
    normalizer_target = (
        "anemoi.models.preprocessing.multi_dataset_normalizer.TopNormalizer"
        if stack == "new"
        else "anemoi.models.preprocessing.normalizer.TopNormalizer"
    )
    return {
        "dataloader": {
            "validation": {
                "dataset": {
                    "zip": [
                        {
                            "name": "lres",
                            "dataset": f"/home/mlx/ai-ml/datasets/downscaling-mars-o{lres}.zarr",
                        },
                        {
                            "name": "hres",
                            "dataset": f"/home/mlx/ai-ml/datasets/downscaling-mars-o{hres}-forcings.zarr",
                        },
                        {
                            "name": "out",
                            "dataset": f"/home/mlx/ai-ml/datasets/downscaling-mars-o{hres}.zarr",
                        },
                    ]
                }
            }
        },
        "data": {
            "processors": {
                "normalizer": {
                    "_target_": normalizer_target,
                }
            }
        },
    }


def test_infer_lane_from_config_o48_o96():
    cfg = _cfg_for_pair(48, 96, stack="new")
    assert mod.infer_lane_from_config(cfg) == "o48_o96"


def test_infer_lane_from_config_o96_o320():
    cfg = _cfg_for_pair(96, 320, stack="old")
    assert mod.infer_lane_from_config(cfg) == "o96_o320"


def test_infer_lane_from_config_o320_o1280():
    cfg = _cfg_for_pair(320, 1280, stack="new")
    assert mod.infer_lane_from_config(cfg) == "o320_o1280"


def test_infer_lane_fails_for_unknown_pair():
    cfg = _cfg_for_pair(320, 2560, stack="new")
    with pytest.raises(RuntimeError, match="Unsupported lane resolution pair"):
        mod.infer_lane_from_config(cfg)


def test_infer_stack_new_marker():
    cfg = _cfg_for_pair(320, 1280, stack="new")
    assert mod.infer_stack_from_config(cfg) == "new"


def test_infer_stack_old_marker():
    cfg = _cfg_for_pair(96, 320, stack="old")
    assert mod.infer_stack_from_config(cfg) == "old"


def test_resolve_profile_checks_expected_lane_stack_and_venv(monkeypatch):
    cfg = _cfg_for_pair(320, 1280, stack="new")
    monkeypatch.setattr(mod, "_load_checkpoint_config", lambda _: cfg)
    prof = mod.resolve_profile(
        checkpoint_path="/tmp/ckpt.ckpt",
        source_hpc="ac",
        host_short="ac-login-01",
        expected_lane="o320_o1280",
        expected_stack_flavor="new",
        expected_venv="/home/ecm5702/dev/.ds-dyn/bin/activate",
    )
    assert prof.lane == "o320_o1280"
    assert prof.stack_flavor == "new"
    assert prof.recommended_venv == "/home/ecm5702/dev/.ds-dyn/bin/activate"


def test_resolve_profile_mismatch_fails(monkeypatch):
    cfg = _cfg_for_pair(96, 320, stack="old")
    monkeypatch.setattr(mod, "_load_checkpoint_config", lambda _: cfg)
    with pytest.raises(RuntimeError, match="Stack mismatch"):
        mod.resolve_profile(
            checkpoint_path="/tmp/ckpt.ckpt",
            source_hpc="ag",
            host_short="ag-login-01",
            expected_stack_flavor="new",
        )
