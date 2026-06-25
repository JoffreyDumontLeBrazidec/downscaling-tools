import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _comm_size(model_comm_group) -> int:
    if model_comm_group is None:
        return 1
    try:
        return int(model_comm_group.size())
    except TypeError:
        return int(model_comm_group.size)


def _mapping_get(mapping, key, default=None):
    if mapping is None:
        return default
    try:
        if key in mapping:
            return mapping[key]
    except TypeError:
        pass
    try:
        return mapping.get(key, default)
    except AttributeError:
        return default


def _call_processor(processor, tensor, **kwargs):
    if processor is None:
        return tensor
    try:
        return processor(tensor, in_place=False, **kwargs)
    except TypeError:
        return processor(tensor)


def _distributed_mean_square(tensor: torch.Tensor) -> torch.Tensor:
    total = torch.sum(tensor.float() ** 2)
    count = torch.tensor(float(tensor.numel()), device=tensor.device, dtype=total.dtype)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return total / count.clamp_min(1.0)


def _is_unified_dict_api(downscaler) -> bool:
    inner = getattr(getattr(downscaler, "model", None), "model", None)
    return (
        hasattr(inner, "_before_sampling")
        and hasattr(inner, "compute_residuals")
        and hasattr(downscaler, "target_dataset_names")
    )


def _target_dataset_name(downscaler) -> str:
    inner = getattr(getattr(downscaler, "model", None), "model", None)
    decoder_datasets = getattr(inner, "_decoder_datasets", None)
    if decoder_datasets:
        return decoder_datasets[0]
    target_names = getattr(downscaler, "target_dataset_names", None)
    if target_names:
        return target_names[0]
    return "out_hres"


def _use_spatial_sigma_sharding(downscaler) -> bool:
    """Match the unified model's inference path, not its dataloader policy.

    ``keep_batch_sharded`` only controls whether the training task gathers reader
    shards before ``_step``.  The unified model's ``_before_sampling`` spatially
    shards its already-complete inference batch whenever model parallelism is
    active, including for checkpoints whose training config sets that flag false.
    """
    return _comm_size(getattr(downscaler, "model_comm_group", None)) > 1


def _apply_native_grid_shard_state(downscaler, grid_shard_sizes) -> None:
    """Mirror native grid shard state onto the task instance.

    ``_before_sampling`` is the single source of truth for the per-dataset grid
    shard sizes (its hidden-mesh all-to-all is only consistent with the sizes it
    derives itself).  The diffusion loss reduction reads ``grid_shard_slice`` off
    the task, so we replicate ONLY the slices implied by the model's own sizes —
    we never recompute the sizes ourselves.
    """
    dataset_names = list(getattr(downscaler, "dataset_names", ()) or ())
    # No real sharding (single rank): _before_sampling returns either None or an
    # all-None dict. Avoid importing the partition helper in that case.
    if grid_shard_sizes is None or all(v is None for v in grid_shard_sizes.values()):
        names = dataset_names or list(grid_shard_sizes or ())
        downscaler.grid_shard_sizes = {name: None for name in names}
        downscaler.grid_shard_slice = {name: None for name in names}
        return

    from anemoi.models.distributed.balanced_partition import get_partition_range

    rank = getattr(downscaler, "reader_group_rank", None)
    if rank is None:
        rank = getattr(downscaler, "model_comm_group_rank", 0)

    downscaler.grid_shard_sizes = dict(grid_shard_sizes)
    downscaler.grid_shard_slice = {}
    for dataset_name, shard_sizes in downscaler.grid_shard_sizes.items():
        if shard_sizes is None:
            downscaler.grid_shard_slice[dataset_name] = None
            continue
        start, end = get_partition_range(shard_sizes, int(rank))
        downscaler.grid_shard_slice[dataset_name] = slice(start, end)


def _raw_upsampled_interp_with_native_sizes(inner_model, x_in_lres, grid_shard_sizes, model_comm_group):
    """Replicate the RAW (non-preprocessed) lres upsample that ``_before_sampling`` does internally.

    ``_before_sampling`` returns the PREPROCESSED upsampled conditioning, but the
    residual target needs the RAW upsample on the same grid.  We reproduce exactly
    the model's own ``residual["in_lres"]`` call.

    Note ``grid_shard_sizes["in_lres"]`` is the POST-upsample (hres) size — after
    upsampling the lres input lives on the hres grid, so ``_before_sampling`` stores
    the hres sizes under that key.  The RAW upsample, however, consumes the lres grid
    and must be sharded with the LRES shard sizes, which ``_before_sampling`` derives
    internally via ``get_shard_sizes`` and does not return.  We recompute them the
    same way here.  Non-distributed falls through to the full-grid path.
    """
    sharded = grid_shard_sizes is not None and model_comm_group is not None and _comm_size(model_comm_group) > 1
    lres_shard_sizes = None
    if sharded:
        from anemoi.models.distributed.graph import shard_tensor
        from anemoi.models.distributed.shapes import get_shard_sizes

        lres_shard_sizes = get_shard_sizes(x_in_lres, -2, model_comm_group=model_comm_group)
        x_in_lres = shard_tensor(x_in_lres, -2, lres_shard_sizes, model_comm_group)
    x_interp_raw = inner_model.residual["in_lres"](
        x_in_lres,
        grid_shard_sizes=lres_shard_sizes,
        model_comm_group=model_comm_group,
    )[:, :, None, :, :]
    return x_interp_raw


def _residual_pre_processor(downscaler, target_dataset: str):
    residual_pre = getattr(downscaler, "_residual_pre_processors", None)
    processor = _mapping_get(residual_pre, target_dataset)
    if processor is not None:
        return processor
    return _mapping_get(getattr(downscaler.model, "pre_processors_tendencies", None), target_dataset)


def _residual_post_processor(downscaler, target_dataset: str):
    processor = _mapping_get(getattr(downscaler.model, "post_processors_tendencies", None), target_dataset)
    if processor is not None:
        return processor
    return _mapping_get(getattr(downscaler.model, "post_processors", None), target_dataset)


def _disable_first_run_checks_for_nan_free_sigma_eval(*processors) -> None:
    for processor in processors:
        if processor is not None and getattr(processor, "first_run", False):
            processor.first_run = False
            logger.info("Skipped redundant first-run processor NaN check for verified finite sigma bundles")


def _localize_data_index_tensors(data_indices, device) -> int:
    moved = 0
    collections = data_indices.values() if isinstance(data_indices, dict) else (data_indices,)
    for collection in collections:
        for index_kind in ("data", "model"):
            index = getattr(collection, index_kind, None)
            for tensor_index_kind in ("input", "output"):
                tensor_index = getattr(index, tensor_index_kind, None)
                for field in ("prognostic", "diagnostic", "forcing", "target", "full"):
                    tensor = getattr(tensor_index, field, None)
                    if torch.is_tensor(tensor) and tensor.device != torch.device(device):
                        setattr(tensor_index, field, tensor.to(device))
                        moved += 1
    return moved


class SigmaEvaluator:
    STANDARD_FIELDS = (
        "10u", "10v", "2d", "2t", "msl",
        "skt", "sp", "tcw", "z_500", "u_850", "v_850",
    )

    def __init__(
        self,
        downscaler,
        datamodule,
        N_samples,
        name_to_index=None,
        *,
        inference_model=None,
        device=None,
        model_comm_group=None,
        bundle_paths=None,
        output_weather_state_mode: str = "all",
        output_weather_states=None,
        precision: str = "fp32",
        sigma_min_floor: float = 0.02,
    ):
        self.downscaler = downscaler
        self.datamodule = datamodule
        self.N_samples = N_samples
        self.name_to_index = name_to_index or {}
        self._warned_fields: set = set()

        # Predict-routing (unified) mode. When ``inference_model`` is provided we do
        # NOT reconstruct the diffusion forward inside the harness; we call the model's
        # own ``predict_step`` end-to-end (the same GREEN path as ``eval.cli predict``),
        # with a degenerate 1-step sampler forced at the target sigma. The result is the
        # one-step CEILING (capacity-at-sigma) prediction, scored vs the bundle truth y.
        # This is the H3 instrument: NOT the EDM denoising loss, but a per-sigma skill
        # ceiling that uses the only forward path known not to deadlock on the uneven
        # hidden-mesh all-to-all split.
        self.inference_model = inference_model
        self.device = device
        self.model_comm_group = model_comm_group
        self.bundle_paths = list(bundle_paths) if bundle_paths is not None else None
        self.output_weather_state_mode = output_weather_state_mode
        self.output_weather_states = output_weather_states
        self.precision = precision
        self.sigma_min_floor = float(sigma_min_floor)

    def _is_predict_routing_mode(self) -> bool:
        return self.inference_model is not None and self.bundle_paths is not None

    def evaluate_sigma(self, sigma, prediction_on_pure_noise):
        if self._is_predict_routing_mode():
            return self._evaluate_sigma_predict(sigma, prediction_on_pure_noise)

        self.downscaler.eval()
        total_loss = 0.0
        total_metrics = {}
        n_batches = 0

        dataloader = self.datamodule.val_dataloader()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.N_samples:
                break
            loss, metrics = self.evaluate_batch_with_sigma(
                sigma, batch, prediction_on_pure_noise
            )
            total_loss += loss.item()

            for key, value in metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().float().cpu().item()
                total_metrics[key] = total_metrics.get(key, 0.0) + float(value)

            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_metrics = {key: value / n_batches for key, value in total_metrics.items()}

        return avg_loss, avg_metrics

    def evaluate_batch_with_sigma(self, sigma, batch, prediction_on_pure_noise):
        # Predict-routing mode never reaches here (evaluate_sigma short-circuits).
        # This dispatch only serves the legacy (non-unified, dataloader-batch) path.
        return self._evaluate_batch_with_sigma_legacy(sigma, batch, prediction_on_pure_noise)

    # ------------------------------------------------------------------
    # Predict-routing (unified) mode: call the model's REAL predict_step.
    # ------------------------------------------------------------------
    def _build_one_step_extra_args(self, sigma: float) -> dict:
        """Force a degenerate 1-step denoise at ``sigma`` via predict_step kwargs.

        ``num_steps=1`` with ``sigma_max==sigma`` and a tiny ``sigma_min`` floor makes
        the sampler take a single denoising step starting from ``sigma``, so predict_step
        returns the model's one-step teacher/ceiling prediction at that noise level.
        ``schedule_type`` is forced to a single-segment scheduler (``karras``) so the
        experimental piecewise default does not split the (1-step) schedule.
        """
        sigma_value = float(sigma)
        sigma_min = min(self.sigma_min_floor, sigma_value)
        return {
            "schedule_type": "karras",
            "num_steps": 1,
            "sigma_max": sigma_value,
            "sigma_min": sigma_min,
            "rho": 7.0,
            "sampler": "heun",
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": float("inf"),
            "S_noise": 1.0,
        }

    def _evaluate_sigma_predict(self, sigma, prediction_on_pure_noise):
        """Average one-step CEILING skill over the bundle set at a fixed sigma.

        For each bundle we call the inference model's own ``predict_step`` (via
        ``_predict_from_bundle``) with the degenerate 1-step sampler forced at
        ``sigma``, then compute per-field MSE between the prediction and the bundle
        truth ``y``. ``prediction_on_pure_noise`` is accepted for CSV-schema parity but
        is a no-op here: predict_step already initialises its own latent from noise at
        ``sigma_max``, so there is no separate pure-noise target to seed.
        """
        from manual_inference.prediction.predict import _predict_from_bundle

        if prediction_on_pure_noise:
            logger.info(
                "predict-routing sigma mode ignores prediction_on_pure_noise "
                "(predict_step seeds its own latent from noise at sigma_max=%.4g)",
                float(sigma),
            )

        extra_args = self._build_one_step_extra_args(sigma)
        bundle_paths = self.bundle_paths[: self.N_samples] if self.N_samples > 0 else self.bundle_paths

        total_metrics: dict = {}
        n_batches = 0
        for bundle_path in bundle_paths:
            with torch.inference_mode():
                (
                    _x,
                    y_np,
                    y_pred_np,
                    *_grids_states,
                ) = _predict_from_bundle(
                    inference_model=self.inference_model,
                    datamodule=self.datamodule,
                    device=self.device,
                    bundle_nc=str(bundle_path),
                    member_index=0,
                    extra_args=extra_args,
                    precision=self.precision,
                    model_comm_group=self.model_comm_group,
                    output_weather_state_mode=self.output_weather_state_mode,
                    output_weather_states=self.output_weather_states,
                )
            metrics = self._per_field_metrics_from_numpy(y_pred_np, y_np)
            for key, value in metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().float().cpu().item()
                total_metrics[key] = total_metrics.get(key, 0.0) + float(value)
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No bundles evaluated in predict-routing sigma mode.")

        avg_metrics = {key: value / n_batches for key, value in total_metrics.items()}
        # No EDM denoising loss in this instrument; report the all-field RMSE as ``loss``
        # so the CSV ``loss`` column stays populated and monotone-comparable across sigma.
        avg_loss = float(avg_metrics.get("diff_all_var_non_weighted", float("nan")))
        return avg_loss, avg_metrics

    def _per_field_metrics_from_numpy(self, y_pred_np, y_np) -> dict:
        """Per-field MSE between prediction and bundle truth, plus all-field RMSE.

        ``y_pred_np`` and ``y_np`` come out of ``_predict_from_bundle`` in the model's
        output-weather-state channel order (last axis), so ``self.name_to_index`` maps
        field names to the channel axis directly. NaNs in truth (missing channels) are
        masked out per field.
        """
        import numpy as np

        metrics: dict = {}
        if y_np is None:
            logger.warning("Bundle has no truth y; per-field sigma metrics will be NaN")
            for name in self.STANDARD_FIELDS:
                metrics[f"mse_{name}_non_weighted"] = float("nan")
            metrics["diff_all_var_non_weighted"] = float("nan")
            return metrics

        pred = np.asarray(y_pred_np, dtype=np.float64)
        truth = np.asarray(y_np, dtype=np.float64)
        # Collapse leading singleton axes so the channel axis is last and shapes match.
        pred = np.squeeze(pred)
        truth = np.squeeze(truth)
        if pred.shape != truth.shape:
            # Fall back to broadcasting on the shared trailing dims (channel axis last).
            min_ndim = min(pred.ndim, truth.ndim)
            pred = pred.reshape((-1,) + pred.shape[-min_ndim:]) if pred.ndim > min_ndim else pred
            truth = truth.reshape((-1,) + truth.shape[-min_ndim:]) if truth.ndim > min_ndim else truth

        diff = pred - truth
        finite = np.isfinite(diff)
        all_sq = diff[finite] ** 2 if finite.any() else np.array([np.nan])
        metrics["diff_all_var_non_weighted"] = float(np.sqrt(np.nanmean(all_sq)))

        num_output_vars = diff.shape[-1]
        for name in self.STANDARD_FIELDS:
            idx = self.name_to_index.get(name)
            if idx is not None and idx < num_output_vars:
                field_diff = diff[..., idx]
                field_finite = np.isfinite(field_diff)
                if field_finite.any():
                    metrics[f"mse_{name}_non_weighted"] = float(
                        np.mean(field_diff[field_finite] ** 2)
                    )
                else:
                    metrics[f"mse_{name}_non_weighted"] = float("nan")
            else:
                metrics[f"mse_{name}_non_weighted"] = float("nan")
                if name not in self._warned_fields:
                    self._warned_fields.add(name)
                    logger.warning(
                        "Field %r unavailable (idx=%s, n_out=%d) — writing NaN",
                        name, idx, num_output_vars,
                    )
        return metrics

    def _add_residual_diff_metrics(self, metrics_next, diff: torch.Tensor) -> None:
        metrics_next["diff_all_var_non_weighted"] = torch.sqrt(_distributed_mean_square(diff))

        num_output_vars = diff.shape[-1]
        for name in self.STANDARD_FIELDS:
            idx = self.name_to_index.get(name)
            if idx is not None and idx < num_output_vars:
                metrics_next[f"mse_{name}_non_weighted"] = _distributed_mean_square(diff[..., idx])
            else:
                metrics_next[f"mse_{name}_non_weighted"] = torch.tensor(float("nan"), device=diff.device)
                if name not in self._warned_fields:
                    self._warned_fields.add(name)
                    logger.warning(
                        "Field %r unavailable (idx=%s, n_out=%d) — writing NaN",
                        name, idx, num_output_vars,
                    )

    def _evaluate_batch_with_sigma_legacy(self, sigma, batch, prediction_on_pure_noise):
        with torch.inference_mode():
            sigma = torch.Tensor([sigma]).to(self.downscaler.device)
            sigma_view = sigma.view(-1, 1, 1, 1)
            batch = [x.to(self.downscaler.device) for x in batch]
            x_in, x_in_hres, y = batch

            x_in_interp_to_hres = (
                self.downscaler.model.model.apply_interpolate_to_high_res(
                    x_in[:, 0, ...],
                    grid_shard_shapes=self.downscaler.lres_grid_shard_shapes,
                    model_comm_group=self.downscaler.model_comm_group,
                )[:, None, ...]
            )

            residuals_target = self.downscaler.model.model.compute_residuals(
                y,
                x_in_interp_to_hres,
                direct_prediction_indices=getattr(self.downscaler, "direct_prediction_indices", None),
            )

            x_in_interp_to_hres = self.downscaler.model.pre_processors(
                x_in_interp_to_hres, dataset="input_lres"
            )
            x_in_hres = self.downscaler.model.pre_processors(
                x_in_hres, dataset="input_hres"
            )
            residuals_target = self.downscaler.model.pre_processors(
                residuals_target, dataset="output"
            )

            sigma_data = 1
            noise_weights = (sigma_view**2 + sigma_data**2) / (
                sigma_view * sigma_data
            ) ** 2
            if prediction_on_pure_noise:
                residuals_target_noised = (
                    torch.randn(
                        residuals_target.shape,
                        device=residuals_target.device,
                    )
                    * sigma
                )
            else:
                residuals_target_noised = self.downscaler._noise_target(
                    residuals_target, sigma
                )

            y_pred = self.downscaler(
                x_in_interp_to_hres,
                x_in_hres,
                residuals_target_noised,
                sigma_view,
            )
            loss, metrics_next = self.downscaler.compute_loss_metrics(
                y_pred=y_pred[:, 0, ...],
                y=residuals_target[:, 0, ...],
                rollout_step=0,
                training_mode=True,
                validation_mode=True,
                weights=noise_weights,
                use_reentrant=False,
            )
            denorm_pred_residuals = self.downscaler.model.post_processors(
                y_pred, dataset="output", in_place=False
            )
            denorm_truth_residuals = self.downscaler.model.post_processors(
                residuals_target[:, 0, ...], dataset="output", in_place=False
            )

            diff = denorm_pred_residuals - denorm_truth_residuals
            self._add_residual_diff_metrics(metrics_next, diff)

            del y_pred, residuals_target_noised, x_in, x_in_hres, residuals_target
        return loss, metrics_next
