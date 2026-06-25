from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from eval._backends.sigma_evaluator.sigma_evaluator import SigmaEvaluator
from eval._backends.sigma_evaluator.sigma_evaluator import _disable_first_run_checks_for_nan_free_sigma_eval
from eval._backends.sigma_evaluator.sigma_evaluator import _localize_data_index_tensors
from eval._backends.sigma_evaluator.sigma_evaluator import _use_spatial_sigma_sharding


class _IdentityProcessor:
    first_run = False

    def __call__(self, tensor, **_kwargs):
        return tensor


def test_disable_first_run_checks_for_nan_free_sigma_eval():
    processor = SimpleNamespace(first_run=True)

    _disable_first_run_checks_for_nan_free_sigma_eval(processor, None)

    assert processor.first_run is False


def test_spatial_sigma_sharding_follows_model_parallel_inference_contract():
    class _Group:
        def size(self):
            return 4

    downscaler = SimpleNamespace(model_comm_group=_Group(), keep_batch_sharded=False)
    assert _use_spatial_sigma_sharding(downscaler) is True

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


# ---------------------------------------------------------------------------
# Predict-routing (unified) mode: the sigma diagnostic calls the model's REAL
# predict_step (1-step sampler forced at sigma) instead of reconstructing the
# diffusion forward inside the harness. These tests cover the per-sigma kwargs,
# the per-field metric reduction, and the predict_step routing in evaluate_sigma.
# ---------------------------------------------------------------------------


def _predict_routing_evaluator(name_to_index, bundle_paths, inference_model):
    return SigmaEvaluator(
        downscaler=None,
        datamodule=object(),
        N_samples=len(bundle_paths),
        name_to_index=name_to_index,
        inference_model=inference_model,
        device="cpu",
        model_comm_group=None,
        bundle_paths=bundle_paths,
    )


def test_one_step_extra_args_force_single_denoise_at_sigma():
    evaluator = _predict_routing_evaluator({"2t": 0}, ["b0"], inference_model=object())
    args = evaluator._build_one_step_extra_args(3.5)

    assert args["num_steps"] == 1
    assert args["sigma_max"] == 3.5
    # sigma_min floors below sigma so a single Heun step denoises from sigma.
    assert args["sigma_min"] <= 3.5
    assert args["schedule_type"] == "karras"
    assert args["S_churn"] == 0.0


def test_one_step_extra_args_floor_never_exceeds_small_sigma():
    evaluator = _predict_routing_evaluator({"2t": 0}, ["b0"], inference_model=object())
    args = evaluator._build_one_step_extra_args(0.005)
    # For tiny sigma, sigma_min must not exceed sigma_max.
    assert args["sigma_min"] <= args["sigma_max"] == 0.005


def test_per_field_metrics_from_numpy_masks_nan_and_indexes_channels():
    evaluator = SigmaEvaluator.__new__(SigmaEvaluator)
    evaluator.name_to_index = {"2t": 0, "msl": 1}
    evaluator.STANDARD_FIELDS = SigmaEvaluator.STANDARD_FIELDS
    evaluator._warned_fields = set()

    # shape (batch, ens, grid, vars) with 2 channels.
    pred = np.zeros((1, 1, 4, 2), dtype=np.float32)
    truth = np.zeros((1, 1, 4, 2), dtype=np.float32)
    pred[..., 0] = 2.0  # 2t error == 2 -> mse 4
    truth[..., 1] = 1.0  # msl error == -1 -> mse 1

    metrics = evaluator._per_field_metrics_from_numpy(pred, truth)

    assert abs(metrics["mse_2t_non_weighted"] - 4.0) < 1e-6
    assert abs(metrics["mse_msl_non_weighted"] - 1.0) < 1e-6
    # Fields not in the model output are NaN.
    assert np.isnan(metrics["mse_10u_non_weighted"])
    # All-field RMSE = sqrt(mean([4_each, 1_each])) = sqrt(2.5).
    assert abs(metrics["diff_all_var_non_weighted"] - np.sqrt(2.5)) < 1e-6


def test_per_field_metrics_handles_missing_truth():
    evaluator = SigmaEvaluator.__new__(SigmaEvaluator)
    evaluator.name_to_index = {"2t": 0}
    evaluator.STANDARD_FIELDS = SigmaEvaluator.STANDARD_FIELDS
    evaluator._warned_fields = set()

    metrics = evaluator._per_field_metrics_from_numpy(np.zeros((1, 1, 4, 1)), None)
    assert np.isnan(metrics["diff_all_var_non_weighted"])
    assert np.isnan(metrics["mse_2t_non_weighted"])


def test_evaluate_sigma_routes_through_predict_from_bundle(monkeypatch):
    """evaluate_sigma calls the model's predict_step (via _predict_from_bundle) per bundle,
    forcing a 1-step sampler at sigma, and averages per-field MSE vs the bundle truth y."""
    import eval._backends.sigma_evaluator.sigma_evaluator as se_mod

    calls = []

    def _fake_predict_from_bundle(
        *, inference_model, datamodule, device, bundle_nc, member_index,
        extra_args, precision, model_comm_group, output_weather_state_mode, output_weather_states,
    ):
        calls.append({"bundle": bundle_nc, "extra_args": extra_args})
        # x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, dates
        y = np.zeros((1, 1, 4, 1), dtype=np.float32)
        y_pred = np.full((1, 1, 4, 1), 2.0, dtype=np.float32)  # error 2 -> mse 4
        return (None, y, y_pred, None, None, None, None, ["2t"], None)

    # _predict_from_bundle is imported inside the method; patch it at its source module.
    import manual_inference.prediction.predict as predict_mod
    monkeypatch.setattr(predict_mod, "_predict_from_bundle", _fake_predict_from_bundle, raising=False)

    inference_model = SimpleNamespace()
    evaluator = _predict_routing_evaluator({"2t": 0}, ["b0", "b1"], inference_model)
    evaluator.N_samples = 2

    loss, metrics = evaluator.evaluate_sigma(1.5, prediction_on_pure_noise=False)

    assert len(calls) == 2
    assert calls[0]["extra_args"]["num_steps"] == 1
    assert calls[0]["extra_args"]["sigma_max"] == 1.5
    assert abs(metrics["mse_2t_non_weighted"] - 4.0) < 1e-6
    assert abs(loss - 2.0) < 1e-6  # RMSE of all-field diff = sqrt(4) = 2
