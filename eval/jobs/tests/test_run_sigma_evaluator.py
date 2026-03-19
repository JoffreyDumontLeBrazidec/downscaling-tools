from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str):
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


class _DummyMove:
    def to(self, _device):
        return self


def test_run_sigma_evaluator_forces_full_grid_standalone(tmp_path: Path, monkeypatch):
    mod = _load_module(
        "eval.sigma_evaluator.run_sigma_evaluator",
    )

    created_loaders = []

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                hardware=_ns(num_gpus_per_model=4),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            self.config_for_datamodule = _ns(
                hardware=_ns(num_gpus_per_model=4),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            created_loaders.append(self)

        def load(self):
            self.datamodule = object()
            self.interface = _DummyMove()
            self.downscaler = _DummyMove()

    class _DummySigmaEvaluator:
        def __init__(self, downscaler, datamodule, n_samples):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            return 0.25, {
                "diff_all_var_non_weighted": 0.5,
                "sigma_seen": float(sigma),
                "pure_noise_seen": float(prediction_on_pure_noise),
            }

    checkpoint_config = _ns(
        hardware=_ns(num_gpus_per_model=4),
        dataloader=_ns(
            read_group_size=4,
            validation=_ns(frequency="12h", num_workers=16),
        ),
    )

    monkeypatch.setattr(mod, "ObjectFromCheckpointLoader", _DummyLoader)
    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: ({}, checkpoint_config))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns())
    monkeypatch.setattr(mod, "adapt_config_hpc", lambda config_checkpoint, _config: config_checkpoint)
    monkeypatch.setattr(mod, "_rewrite_dataset_paths_in_place", lambda cfg: cfg)
    monkeypatch.setattr(mod, "SigmaEvaluator", _DummySigmaEvaluator)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
    )

    mod.run_sigma_evaluator(args)

    loader = created_loaders[-1]
    assert loader.config_checkpoint.hardware.num_gpus_per_model == 1
    assert loader.config_checkpoint.dataloader.read_group_size == 1
    assert loader.config_for_datamodule.hardware.num_gpus_per_model == 1
    assert loader.config_for_datamodule.dataloader.read_group_size == 1
    assert loader.config_for_datamodule.dataloader.validation.frequency == "50h"
    assert loader.config_for_datamodule.dataloader.validation.num_workers == 0
    assert out_csv.exists()
