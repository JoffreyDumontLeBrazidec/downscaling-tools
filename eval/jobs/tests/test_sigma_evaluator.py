from __future__ import annotations

from types import SimpleNamespace

import torch

from eval._backends.sigma_evaluator.sigma_evaluator import SigmaEvaluator
from eval._backends.sigma_evaluator.sigma_evaluator import _disable_first_run_checks_for_nan_free_sigma_eval
from eval._backends.sigma_evaluator.sigma_evaluator import _localize_data_index_tensors
from eval._backends.sigma_evaluator.sigma_evaluator import _use_spatial_sigma_sharding


class _IdentityProcessor:
    def __call__(self, tensor, **_kwargs):
        return tensor


def test_disable_first_run_checks_for_nan_free_sigma_eval():
    processor = SimpleNamespace(first_run=True)

    _disable_first_run_checks_for_nan_free_sigma_eval(processor, None)

    assert processor.first_run is False


def test_spatial_sigma_sharding_requires_checkpoint_opt_in():
    class _Group:
        def size(self):
            return 4

    downscaler = SimpleNamespace(model_comm_group=_Group(), keep_batch_sharded=False)
    assert _use_spatial_sigma_sharding(downscaler) is False

    downscaler.keep_batch_sharded = True
    assert _use_spatial_sigma_sharding(downscaler) is True


def test_localize_data_index_tensors_accepts_unified_index_shape():
    tensor_index = SimpleNamespace(
        prognostic=torch.tensor([0]),
        diagnostic=torch.tensor([], dtype=torch.long),
        forcing=torch.tensor([], dtype=torch.long),
        target=torch.tensor([], dtype=torch.long),
        full=torch.tensor([0]),
    )
    collection = SimpleNamespace(
        data=SimpleNamespace(input=tensor_index, output=tensor_index),
        model=SimpleNamespace(input=tensor_index, output=tensor_index),
    )

    assert _localize_data_index_tensors({"out_hres": collection}, "cpu") == 0
    assert tensor_index.full.device.type == "cpu"


class _DummyInnerModel:
    def __init__(self):
        self._decoder_datasets = ["out_hres"]
        self._residual_pairs = {"out_hres": "in_lres"}
        self.sigma_data = 1.0
        output = SimpleNamespace(
            full=torch.arange(2),
            prognostic=torch.arange(2),
            diagnostic=torch.tensor([], dtype=torch.long),
        )
        self.data_indices = {
            "out_hres": SimpleNamespace(
                data=SimpleNamespace(output=output),
                model=SimpleNamespace(output=output),
            )
        }
        self.interp_calls = []
        self.residual_interp_calls = []
        self.residual_calls = []
        self.residual = {"in_lres": self._residual_interp}

    def _residual_interp(self, x, grid_shard_sizes=None, model_comm_group=None):
        self.residual_interp_calls.append(
            {
                "shape": tuple(x.shape),
                "grid_shard_sizes": grid_shard_sizes,
                "model_comm_group": model_comm_group,
            }
        )
        return torch.ones((x.shape[0], x.shape[2], 4, x.shape[-1]), device=x.device)

    def _before_sampling(self, batch, pre_processors, n_step_input, model_comm_group, **_kwargs):
        assert n_step_input == 1
        assert model_comm_group is None
        return (batch["in_lres"], batch["in_hres"]), {
            "in_lres": None,
            "in_hres": None,
            "out_hres": None,
        }

    def apply_interpolate_to_high_res(self, x, grid_shard_sizes=None, model_comm_group=None):
        self.interp_calls.append(
            {
                "shape": tuple(x.shape),
                "grid_shard_sizes": grid_shard_sizes,
                "model_comm_group": model_comm_group,
            }
        )
        return torch.ones((x.shape[0], 1, 1, 4, x.shape[-1]), device=x.device)

    def get_matching_channel_indices(self, target_dataset):
        assert target_dataset == "out_hres"
        return torch.arange(2)

    def compute_residuals(
        self,
        *,
        y,
        x_interp,
        pre_processors_state,
        pre_processors_tendencies,
        target_dataset,
        skip_imputation,
    ):
        assert target_dataset == "out_hres"
        assert skip_imputation is True
        assert tuple(x_interp.shape) == tuple(y.shape)
        self.residual_calls.append(tuple(x_interp.shape))
        return y - x_interp


class _DummyDownscaler:
    def __init__(self):
        inner = _DummyInnerModel()
        self.model = SimpleNamespace(
            model=inner,
            pre_processors={"in_lres": _IdentityProcessor(), "in_hres": _IdentityProcessor(), "out_hres": _IdentityProcessor()},
            post_processors={"out_hres": _IdentityProcessor()},
            pre_processors_tendencies={"out_hres": _IdentityProcessor()},
            post_processors_tendencies={"out_hres": _IdentityProcessor()},
        )
        self.device = "cpu"
        self.model_comm_group = None
        self.target_dataset_names = ["out_hres"]
        self.dataset_names = ["in_lres", "in_hres", "out_hres"]
        self.forward_calls = []

    def eval(self):
        return self

    def _noise_target(self, target, sigma):
        return {name: value for name, value in target.items()}

    def __call__(self, x, y_noised, sigma):
        assert set(x) == {"in_lres", "in_hres"}
        assert set(y_noised) == {"out_hres"}
        assert set(sigma) == {"out_hres"}
        self.forward_calls.append((x, y_noised, sigma))
        return {"out_hres": torch.zeros_like(y_noised["out_hres"])}

    def compute_loss_metrics(self, y_pred, y, validation_mode, weights, **_kwargs):
        assert validation_mode is True
        assert set(y_pred) == {"out_hres"}
        assert set(y) == {"out_hres"}
        assert set(weights) == {"out_hres"}
        return torch.tensor(1.0), {}, y_pred


class _OneBatchDataModule:
    def __init__(self, batch):
        self.batch = batch

    def val_dataloader(self):
        return [self.batch]


def test_unified_sigma_evaluator_uses_dict_api_without_legacy_lres_shard_attr():
    downscaler = _DummyDownscaler()
    batch = (
        torch.zeros((1, 1, 1, 3, 2)),
        torch.zeros((1, 1, 1, 4, 1)),
        torch.full((1, 1, 1, 4, 2), 2.0),
    )

    evaluator = SigmaEvaluator(
        downscaler,
        _OneBatchDataModule(batch),
        N_samples=1,
        name_to_index={"2t": 0},
    )

    loss, metrics = evaluator.evaluate_batch_with_sigma(1.0, batch, prediction_on_pure_noise=False)

    assert loss.item() == 1.0
    assert downscaler.model.model.interp_calls == []
    assert downscaler.model.model.residual_interp_calls == [
        {"shape": (1, 1, 1, 3, 2), "grid_shard_sizes": None, "model_comm_group": None}
    ]
    assert downscaler.model.model.residual_calls == [(1, 1, 1, 4, 2)]
    assert downscaler.forward_calls
    assert "diff_all_var_non_weighted" in metrics
    assert "mse_2t_non_weighted" in metrics
