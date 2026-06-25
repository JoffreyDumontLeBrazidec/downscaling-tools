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



def test_run_sigma_evaluator_injects_checkpoint_compat_profile_before_load(
    tmp_path: Path, monkeypatch
):
    mod = _load_module("eval._backends.sigma_evaluator.run_sigma_evaluator")

    created_loaders = []

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                model=_ns(model=_ns()),
                hardware=_ns(num_gpus_per_model=1),
                dataloader=_ns(read_group_size=1, validation=_ns(frequency="6h", num_workers=8)),
            )
            self.config_for_datamodule = _ns(
                model=_ns(model=_ns()),
                hardware=_ns(num_gpus_per_model=1),
                dataloader=_ns(read_group_size=1, validation=_ns(frequency="6h", num_workers=8)),
            )
            created_loaders.append(self)

        def load(self):
            assert self.config_checkpoint.model.model.compatibility_profile == "jupiter_ln_proof_20260622"
            assert self.config_for_datamodule.model.model.compatibility_profile == "jupiter_ln_proof_20260622"
            self.datamodule = object()
            self.interface = _DummyMove()
            self.downscaler = _DummyMove()

    class _DummySigmaEvaluator:
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            return 0.25, {"diff_all_var_non_weighted": 0.5}

    checkpoint_config = _ns(
        model=_ns(model=_ns()),
        hardware=_ns(num_gpus_per_model=1),
        dataloader=_ns(read_group_size=1, validation=_ns(frequency="12h", num_workers=16)),
    )

    monkeypatch.setattr(mod, "ObjectFromCheckpointLoader", _DummyLoader)
    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: ({}, checkpoint_config))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns())
    monkeypatch.setattr(mod, "adapt_config_hpc", lambda config_checkpoint, _config: config_checkpoint)
    monkeypatch.setattr(mod, "_rewrite_dataset_paths_in_place", lambda cfg: cfg)
    monkeypatch.setattr(mod, "SigmaEvaluator", _DummySigmaEvaluator)
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o96_o320")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=1,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
        residual_statistics_fallback="",
        checkpoint_compat_profile="jupiter_ln_proof_20260622",
    )

    mod.run_sigma_evaluator(args)

    assert created_loaders
    assert out_csv.exists()


def test_run_sigma_evaluator_bundle_root_uses_checkpoint_metadata_loader(
    tmp_path: Path, monkeypatch
):
    mod = _load_module("eval._backends.sigma_evaluator.run_sigma_evaluator")

    bundle_root = tmp_path / "bundles"
    bundle_root.mkdir()
    (bundle_root / "eefo_o320_0001_date20230826_time0000_mem01_step024h_input_bundle.nc").write_text(
        "placeholder"
    )

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                model=_ns(model=_ns()),
                hardware=_ns(num_gpus_per_model=1),
                dataloader=_ns(read_group_size=1, validation=_ns(frequency="6h", num_workers=8)),
            )
            self.config_for_datamodule = _ns(
                model=_ns(model=_ns()),
                hardware=_ns(num_gpus_per_model=1),
                dataloader=_ns(read_group_size=1, validation=_ns(frequency="6h", num_workers=8)),
            )

        def load(self):
            raise AssertionError("bundle-root mode must not open the checkpoint datamodule")

    seen = {}

    class _DummySigmaEvaluator:
        def __init__(
            self,
            downscaler=None,
            datamodule=None,
            N_samples=None,
            name_to_index=None,
            *,
            inference_model=None,
            device=None,
            model_comm_group=None,
            bundle_paths=None,
            output_weather_state_mode="all",
            output_weather_states=None,
            precision="fp32",
            sigma_min_floor=0.02,
        ):
            seen["downscaler"] = downscaler
            seen["inference_model"] = inference_model
            seen["bundle_paths"] = bundle_paths
            seen["name_to_index"] = name_to_index
            self.name_to_index = name_to_index

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            # Predict-routing mode: the model wrapper and the bundle paths must be wired,
            # and the training-task downscaler must NOT be used.
            assert seen["inference_model"] is not None
            assert seen["downscaler"] is None
            assert seen["bundle_paths"]
            assert self.name_to_index == {"2t": 0}
            return 0.25, {"diff_all_var_non_weighted": 0.5}

    checkpoint = {
        "hyper_parameters": {
            "data_indices": {"out_hres": _ns(model=_ns(output=_ns(name_to_index={"2t": 0})))},
            "graph_data": {"graph": "data"},
            "metadata": {"meta": "data"},
            "statistics": {"stats": "data"},
            "statistics_tendencies": {"tendencies": "data"},
            "supporting_arrays": {"supporting": "arrays"},
        }
    }
    checkpoint_config = _ns(
        model=_ns(model=_ns()),
        hardware=_ns(num_gpus_per_model=1),
        dataloader=_ns(read_group_size=1, validation=_ns(frequency="12h", num_workers=16)),
    )
    inference_model = _ns(data_indices={"out_hres": _ns(model=_ns(output=_ns(name_to_index={"2t": 0})))})
    datamodule = object()

    def _dummy_load_objects(*, ckpt_path, device, validation_frequency, precision, num_gpus_per_model_override=None):
        seen["load_objects_ckpt"] = ckpt_path
        seen["load_objects_gpus"] = num_gpus_per_model_override
        return inference_model, datamodule, "/dir", "exp"

    monkeypatch.setattr(mod, "ObjectFromCheckpointLoader", _DummyLoader)
    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: (checkpoint, checkpoint_config))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns())
    monkeypatch.setattr(mod, "adapt_config_hpc", lambda config_checkpoint, _config: config_checkpoint)
    monkeypatch.setattr(mod, "_rewrite_dataset_paths_in_place", lambda cfg: cfg)
    monkeypatch.setattr(mod, "SigmaEvaluator", _DummySigmaEvaluator)
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o96_o320")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "_load_objects", _dummy_load_objects, raising=False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=1,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
        residual_statistics_fallback="",
        checkpoint_compat_profile="",
        bundle_root=str(bundle_root),
    )

    mod.run_sigma_evaluator(args)

    assert out_csv.exists()
    # The inference wrapper was loaded once via _load_objects with the model-parallel width.
    assert seen["load_objects_ckpt"] == "/tmp/checkpoints/exp/model.ckpt"
    assert seen["load_objects_gpus"] == 1


def test_resolve_downscaler_cls_prefers_unified_tasks_export(monkeypatch):
    import types

    mod = _load_module("eval._backends.sigma_evaluator.run_sigma_evaluator")

    class _DummyDownscaler:
        pass

    fake_tasks = types.ModuleType("anemoi.training.train.tasks")
    fake_tasks.GraphDiffusionDownscaler = _DummyDownscaler
    monkeypatch.setitem(sys.modules, "anemoi.training.train.tasks", fake_tasks)

    assert mod._resolve_downscaler_cls() is _DummyDownscaler


def test_load_downscaler_from_checkpoint_metadata_uses_trusted_pickle_load(monkeypatch):
    mod = _load_module("eval._backends.sigma_evaluator.run_sigma_evaluator")
    seen = {}

    class _DummyDownscaler:
        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            seen["path"] = path
            seen["kwargs"] = kwargs
            return cls()

    monkeypatch.setattr(mod, "_resolve_downscaler_cls", lambda: _DummyDownscaler)

    checkpoint = {
        "hyper_parameters": {
            "data_indices": {"out_hres": object()},
            "graph_data": {"graph": "data"},
            "metadata": {"meta": "data"},
            "statistics": {"stats": "data"},
        }
    }

    mod._load_downscaler_from_checkpoint_metadata(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        checkpoint=checkpoint,
        config_checkpoint=_ns(model=_ns(model=_ns())),
    )

    assert seen["path"] == "/tmp/checkpoints/exp/model.ckpt"
    assert seen["kwargs"]["strict"] is False
    assert seen["kwargs"]["weights_only"] is False


def test_localize_external_checkpoint_paths_rewrites_existing_inter_mat(
    tmp_path: Path, monkeypatch
):
    mod = _load_module("eval._backends.sigma_evaluator.run_sigma_evaluator")

    inter_mat_dir = tmp_path / "inter_mat"
    inter_mat_dir.mkdir()
    local = inter_mat_dir / "interpol_O320_to_O1280_linear.mat.npz"
    local.write_text("placeholder")
    monkeypatch.setenv("INTER_MAT_DIR", str(inter_mat_dir))

    cfg = _ns(
        model=_ns(
            residual=_ns(
                in_lres=_ns(
                    interpolation_file_path=(
                        "/e/data1/jureap-data/ecmwf/users/jdumont/anemoi/inter_mat/"
                        "interpol_O320_to_O1280_linear.mat.npz"
                    )
                )
            )
        ),
        system=_ns(
            input=_ns(
                truncation=(
                    "/e/data1/jureap-data/ecmwf/users/jdumont/anemoi/inter_mat/"
                    "interpol_O320_to_O1280_linear.mat.npz"
                ),
                truncation_inv=(
                    "/e/data1/jureap-data/ecmwf/users/jdumont/anemoi/inter_mat/"
                    "missing_local_file.mat.npz"
                ),
            )
        ),
    )

    rewrites = mod._localize_external_checkpoint_paths(cfg)

    assert cfg.model.residual.in_lres.interpolation_file_path == str(local)
    assert cfg.system.input.truncation == str(local)
    assert cfg.system.input.truncation_inv.endswith("missing_local_file.mat.npz")
    assert rewrites == [
        (
            "/e/data1/jureap-data/ecmwf/users/jdumont/anemoi/inter_mat/"
            "interpol_O320_to_O1280_linear.mat.npz",
            str(local),
        ),
        (
            "/e/data1/jureap-data/ecmwf/users/jdumont/anemoi/inter_mat/"
            "interpol_O320_to_O1280_linear.mat.npz",
            str(local),
        ),
    ]


def test_run_sigma_evaluator_preserves_four_gpu_model_parallel_for_o1280_family(
    tmp_path: Path, monkeypatch
):
    mod = _load_module(
        "eval._backends.sigma_evaluator.run_sigma_evaluator",
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
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

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
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o320_o1280")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 4))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=4,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
        residual_statistics_fallback="",
    )

    mod.run_sigma_evaluator(args)

    loader = created_loaders[-1]
    assert loader.config_checkpoint.hardware.num_gpus_per_model == 4
    assert loader.config_checkpoint.dataloader.read_group_size == 4
    assert loader.config_for_datamodule.hardware.num_gpus_per_model == 4
    assert loader.config_for_datamodule.dataloader.read_group_size == 4
    assert loader.config_for_datamodule.dataloader.validation.frequency == "50h"
    assert loader.config_for_datamodule.dataloader.validation.num_workers == 0
    assert out_csv.exists()


def test_run_sigma_evaluator_defaults_to_single_gpu_for_lower_res_lanes(
    tmp_path: Path, monkeypatch
):
    mod = _load_module(
        "eval._backends.sigma_evaluator.run_sigma_evaluator",
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
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

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
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o96_o320")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=1,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
        residual_statistics_fallback="",
    )

    mod.run_sigma_evaluator(args)

    loader = created_loaders[-1]
    assert loader.config_checkpoint.hardware.num_gpus_per_model == 1
    assert loader.config_checkpoint.dataloader.read_group_size == 1
    assert loader.config_for_datamodule.hardware.num_gpus_per_model == 1
    assert loader.config_for_datamodule.dataloader.read_group_size == 1
    assert out_csv.exists()


def test_run_sigma_evaluator_applies_o1280_o2560_residual_stats_fallback(
    tmp_path: Path, monkeypatch
):
    mod = _load_module(
        "eval._backends.sigma_evaluator.run_sigma_evaluator",
    )

    created_loaders = []
    residual_dir = tmp_path / "residuals"
    residual_dir.mkdir()
    (residual_dir / "o2560_dict_6_72.npy").write_text("placeholder")
    missing_name = "o2560_dict_6_72_destine_recomputed_4fields.npy"

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                hardware=_ns(
                    num_gpus_per_model=4,
                    paths=_ns(residual_statistics=str(residual_dir)),
                    files=_ns(residual_statistics=missing_name),
                ),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            self.config_for_datamodule = _ns(
                hardware=_ns(
                    num_gpus_per_model=4,
                    paths=_ns(residual_statistics=str(residual_dir)),
                    files=_ns(residual_statistics=missing_name),
                ),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            created_loaders.append(self)

        def load(self):
            assert (
                self.config_checkpoint.hardware.files.residual_statistics
                == "o2560_dict_6_72.npy"
            )
            assert (
                self.config_for_datamodule.hardware.files.residual_statistics
                == "o2560_dict_6_72.npy"
            )
            self.datamodule = object()
            self.interface = _DummyMove()
            self.downscaler = _DummyMove()

    class _DummySigmaEvaluator:
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            return 0.25, {
                "diff_all_var_non_weighted": 0.5,
                "sigma_seen": float(sigma),
                "pure_noise_seen": float(prediction_on_pure_noise),
            }

    checkpoint_config = _ns(
        hardware=_ns(
            num_gpus_per_model=4,
            paths=_ns(residual_statistics=str(residual_dir)),
            files=_ns(residual_statistics=missing_name),
        ),
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
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o1280_o2560")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 4))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=0,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
        residual_statistics_fallback="o2560_dict_6_72.npy",
    )

    mod.run_sigma_evaluator(args)

    loader = created_loaders[-1]
    assert loader.config_checkpoint.hardware.files.residual_statistics == "o2560_dict_6_72.npy"
    assert loader.config_for_datamodule.hardware.files.residual_statistics == "o2560_dict_6_72.npy"
    assert out_csv.exists()


def test_sigma_evaluator_single_variable_model():
    """1-channel output: only the matching field gets a real MSE; others are NaN."""
    import math

    import torch

    se_mod = _load_module("eval._backends.sigma_evaluator.sigma_evaluator")
    SigmaEvaluator = se_mod.SigmaEvaluator

    name_to_index = {"2t": 0}  # single-variable model
    evaluator = SigmaEvaluator.__new__(SigmaEvaluator)
    evaluator.name_to_index = name_to_index
    evaluator._warned_fields = set()

    # Simulate what evaluate_batch_with_sigma does for the per-field block:
    diff = torch.randn(100, 1)  # 1 output channel
    metrics = {}
    metrics["diff_all_var_non_weighted"] = torch.sqrt(torch.mean(diff**2))

    num_output_vars = diff.shape[-1]
    for name in SigmaEvaluator.STANDARD_FIELDS:
        idx = evaluator.name_to_index.get(name)
        if idx is not None and idx < num_output_vars:
            metrics[f"mse_{name}_non_weighted"] = torch.mean(diff[..., idx] ** 2)
        else:
            metrics[f"mse_{name}_non_weighted"] = float("nan")

    assert not math.isnan(float(metrics["mse_2t_non_weighted"]))
    for name in SigmaEvaluator.STANDARD_FIELDS:
        if name != "2t":
            assert math.isnan(float(metrics[f"mse_{name}_non_weighted"]))
    assert not math.isnan(float(metrics["diff_all_var_non_weighted"]))


def test_sigma_evaluator_missing_data_indices():
    """When data_indices is unavailable, all per-field metrics are NaN."""
    import math

    import torch

    se_mod = _load_module("eval._backends.sigma_evaluator.sigma_evaluator")
    SigmaEvaluator = se_mod.SigmaEvaluator

    evaluator = SigmaEvaluator.__new__(SigmaEvaluator)
    evaluator.name_to_index = {}  # empty — no data_indices available
    evaluator._warned_fields = set()

    diff = torch.randn(100, 5)
    metrics = {}
    metrics["diff_all_var_non_weighted"] = torch.sqrt(torch.mean(diff**2))

    num_output_vars = diff.shape[-1]
    for name in SigmaEvaluator.STANDARD_FIELDS:
        idx = evaluator.name_to_index.get(name)
        if idx is not None and idx < num_output_vars:
            metrics[f"mse_{name}_non_weighted"] = torch.mean(diff[..., idx] ** 2)
        else:
            metrics[f"mse_{name}_non_weighted"] = float("nan")

    for name in SigmaEvaluator.STANDARD_FIELDS:
        assert math.isnan(float(metrics[f"mse_{name}_non_weighted"]))
    assert not math.isnan(float(metrics["diff_all_var_non_weighted"]))


def test_adapt_config_hpc_missing_hardware_key(tmp_path: Path, monkeypatch):
    """When adapt_config_hpc raises due to missing hardware key, fallback is used."""
    mod = _load_module("eval._backends.sigma_evaluator.run_sigma_evaluator")

    created_loaders = []
    inject_called = []

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                dataloader=_ns(
                    read_group_size=1,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            # No hardware key — adapt_config_hpc will fail
            self.config_for_datamodule = _ns(
                dataloader=_ns(
                    read_group_size=1,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            created_loaders.append(self)

        def load(self):
            self.datamodule = object()
            self.interface = _DummyMove()
            self.downscaler = _DummyMove()

    class _DummySigmaEvaluator:
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            return 0.25, {
                "diff_all_var_non_weighted": 0.5,
            }

    def _failing_adapt(config_checkpoint, _config):
        raise AttributeError("Missing key hardware")

    original_inject = mod._inject_minimal_hardware_config

    def _tracking_inject(config_checkpoint, host_config):
        inject_called.append(True)
        original_inject(config_checkpoint, host_config)

    checkpoint_config = _ns(
        dataloader=_ns(
            read_group_size=1,
            validation=_ns(frequency="12h", num_workers=16),
        ),
    )

    monkeypatch.setattr(mod, "ObjectFromCheckpointLoader", _DummyLoader)
    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: ({}, checkpoint_config))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns(hardware=_ns(paths=_ns(data="/data"))))
    monkeypatch.setattr(mod, "adapt_config_hpc", _failing_adapt)
    monkeypatch.setattr(mod, "_inject_minimal_hardware_config", _tracking_inject)
    monkeypatch.setattr(mod, "_rewrite_dataset_paths_in_place", lambda cfg: cfg)
    monkeypatch.setattr(mod, "SigmaEvaluator", _DummySigmaEvaluator)
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o48_o96")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=1,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
        residual_statistics_fallback="",
    )

    mod.run_sigma_evaluator(args)

    assert len(inject_called) == 1, "_inject_minimal_hardware_config should have been called"
    assert out_csv.exists()
