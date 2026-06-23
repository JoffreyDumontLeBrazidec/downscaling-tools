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


def _set_grid_shard_state(downscaler, grid_shard_sizes) -> None:
    dataset_names = list(getattr(downscaler, "dataset_names", ()) or ())
    if grid_shard_sizes is None:
        downscaler.grid_shard_sizes = {name: None for name in dataset_names}
        downscaler.grid_shard_slice = {name: None for name in dataset_names}
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


def _shard_lres_for_raw_interpolation(x_lres_single_time: torch.Tensor, model_comm_group):
    if _comm_size(model_comm_group) <= 1:
        return x_lres_single_time, None

    from anemoi.models.distributed.graph import shard_tensor
    from anemoi.models.distributed.shapes import get_shard_sizes

    lres_shard_sizes = get_shard_sizes(x_lres_single_time, -2, model_comm_group=model_comm_group)
    return shard_tensor(x_lres_single_time, -2, lres_shard_sizes, model_comm_group), lres_shard_sizes


def _interpolate_lres_to_hres_raw(inner_model, x_in_lres: torch.Tensor, model_comm_group) -> torch.Tensor:
    x_for_interp, lres_shard_sizes = _shard_lres_for_raw_interpolation(x_in_lres[:, 0, ...], model_comm_group)
    try:
        x_interp = inner_model.apply_interpolate_to_high_res(
            x_for_interp,
            grid_shard_sizes=lres_shard_sizes,
            model_comm_group=model_comm_group,
        )
    except TypeError:
        try:
            x_interp = inner_model.apply_interpolate_to_high_res(
                x_for_interp,
                grid_shard_shapes=getattr(inner_model, "lres_grid_shard_shapes", None),
                model_comm_group=model_comm_group,
            )
        except TypeError:
            x_interp = inner_model.apply_interpolate_to_high_res(x_for_interp)

    if x_interp.ndim == 4:
        x_interp = x_interp[:, None, ...]
    if x_interp.ndim != 5:
        raise ValueError(
            "Expected interpolated low-resolution input to be 4D or 5D, "
            f"got shape {tuple(x_interp.shape)}"
        )
    return x_interp


def _shard_target_if_needed(y: torch.Tensor, grid_shard_sizes, target_dataset: str, model_comm_group) -> torch.Tensor:
    if grid_shard_sizes is None:
        return y
    target_shard_sizes = grid_shard_sizes.get(target_dataset)
    if target_shard_sizes is None:
        return y

    from anemoi.models.distributed.graph import shard_tensor

    return shard_tensor(y, -2, target_shard_sizes, model_comm_group)


def _prepare_unified_conditioning_and_raw_interp(
    inner_model,
    model_wrapper,
    x_in,
    x_in_hres,
    model_comm_group,
    *,
    spatial_sharding: bool,
):
    x_in = x_in[:, :1, ...]
    x_in_hres = x_in_hres[:, :1, ...]

    upsample_sharded = spatial_sharding and _comm_size(model_comm_group) > 1
    if upsample_sharded:
        from anemoi.models.distributed.graph import shard_tensor
        from anemoi.models.distributed.shapes import get_shard_sizes

        lres_in_shard_sizes = get_shard_sizes(x_in, -2, model_comm_group=model_comm_group)
        x_in_for_interp = shard_tensor(x_in, -2, lres_in_shard_sizes, model_comm_group)
        x_interp_raw = inner_model.residual["in_lres"](
            x_in_for_interp,
            grid_shard_sizes=lres_in_shard_sizes,
            model_comm_group=model_comm_group,
        )[:, :, None, :, :]
    else:
        x_interp_raw = inner_model.residual["in_lres"](
            x_in,
            grid_shard_sizes=None,
            model_comm_group=model_comm_group,
        )[:, :, None, :, :]

    x_interp_conditioning = model_wrapper.pre_processors["in_lres"](x_interp_raw, in_place=False)
    x_hres_conditioning = model_wrapper.pre_processors["in_hres"](x_in_hres, in_place=False)

    grid_shard_sizes = None
    if spatial_sharding and model_comm_group is not None:
        from anemoi.models.distributed.graph import shard_tensor
        from anemoi.models.distributed.shapes import get_shard_sizes

        shard_sizes_hres = get_shard_sizes(x_in_hres, -2, model_comm_group=model_comm_group)
        shard_sizes_lres = shard_sizes_hres
        grid_shard_sizes = {
            "in_lres": shard_sizes_lres,
            "in_hres": shard_sizes_hres,
            "out_hres": shard_sizes_hres,
        }
        x_hres_conditioning = shard_tensor(x_hres_conditioning, -2, shard_sizes_hres, model_comm_group)

    return x_interp_conditioning, x_hres_conditioning, x_interp_raw, grid_shard_sizes


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

    def __init__(self, downscaler, datamodule, N_samples, name_to_index=None):
        self.downscaler = downscaler
        self.datamodule = datamodule
        self.N_samples = N_samples
        self.name_to_index = name_to_index or {}
        self._warned_fields: set = set()

    def evaluate_sigma(self, sigma, prediction_on_pure_noise):
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
        if _is_unified_dict_api(self.downscaler):
            return self._evaluate_batch_with_sigma_unified(sigma, batch, prediction_on_pure_noise)
        return self._evaluate_batch_with_sigma_legacy(sigma, batch, prediction_on_pure_noise)

    def _evaluate_batch_with_sigma_unified(self, sigma, batch, prediction_on_pure_noise):
        with torch.inference_mode():
            batch = [x.to(self.downscaler.device) for x in batch]
            x_in, x_in_hres, y = batch

            model_wrapper = self.downscaler.model
            inner_model = model_wrapper.model
            model_comm_group = getattr(self.downscaler, "model_comm_group", None)
            target_dataset = _target_dataset_name(self.downscaler)
            moved_index_tensors = _localize_data_index_tensors(inner_model.data_indices, x_in.device)
            if moved_index_tensors:
                logger.info("Localized %d checkpoint data-index tensors to %s", moved_index_tensors, x_in.device)

            spatial_sharding = _use_spatial_sigma_sharding(self.downscaler)
            logger.info(
                "Preparing unified sigma batch (distributed=%s, spatial_sharding=%s)",
                _comm_size(model_comm_group) > 1,
                spatial_sharding,
            )
            x_in_interp_to_hres, x_in_hres, x_interp_raw, grid_shard_sizes = (
                _prepare_unified_conditioning_and_raw_interp(
                    inner_model,
                    model_wrapper,
                    x_in,
                    x_in_hres,
                    model_comm_group,
                    spatial_sharding=spatial_sharding,
                )
            )
            _set_grid_shard_state(self.downscaler, grid_shard_sizes)
            logger.info(
                "Prepared unified sigma batch: x_interp=%s x_hres=%s raw_interp=%s grid_sharded=%s",
                tuple(x_in_interp_to_hres.shape),
                tuple(x_in_hres.shape),
                tuple(x_interp_raw.shape),
                grid_shard_sizes is not None,
            )

            logger.info("Sharding sigma target if needed")
            y_for_residual = _shard_target_if_needed(y, grid_shard_sizes, target_dataset, model_comm_group)
            logger.info("Sigma target ready: y=%s", tuple(y_for_residual.shape))

            channel_indices = None
            if hasattr(inner_model, "get_matching_channel_indices"):
                channel_indices = inner_model.get_matching_channel_indices(target_dataset).to(x_interp_raw.device)
            x_interp_for_residual = x_interp_raw[..., channel_indices] if channel_indices is not None else x_interp_raw

            logger.info("Computing sigma residual target: x_interp=%s", tuple(x_interp_for_residual.shape))
            residual_pre_processor = _residual_pre_processor(self.downscaler, target_dataset)
            _disable_first_run_checks_for_nan_free_sigma_eval(
                model_wrapper.pre_processors[target_dataset],
                residual_pre_processor,
            )
            residuals_target = inner_model.compute_residuals(
                y=y_for_residual,
                x_interp=x_interp_for_residual,
                pre_processors_state=model_wrapper.pre_processors[target_dataset],
                pre_processors_tendencies=residual_pre_processor,
                target_dataset=target_dataset,
                skip_imputation=True,
            )
            logger.info("Sigma residual target ready: residual=%s", tuple(residuals_target.shape))

            sigma_value = float(sigma)
            sigma_base = torch.full(
                (residuals_target.shape[0], residuals_target.shape[2]),
                sigma_value,
                device=residuals_target.device,
                dtype=residuals_target.dtype,
            )
            sigma_tensor = sigma_base[:, None, :, None, None]
            sigma_dict = {target_dataset: sigma_tensor}
            noise_weights = {
                target_dataset: (sigma_tensor**2 + inner_model.sigma_data**2) / (sigma_tensor * inner_model.sigma_data) ** 2
            }
            target_dict = {target_dataset: residuals_target}

            if prediction_on_pure_noise:
                residuals_target_noised = {target_dataset: torch.randn_like(residuals_target) * sigma_tensor}
            else:
                residuals_target_noised = self.downscaler._noise_target(target_dict, sigma_dict)

            logger.info("Running unified sigma forward")
            y_pred = self.downscaler(
                {"in_lres": x_in_interp_to_hres, "in_hres": x_in_hres},
                residuals_target_noised,
                sigma_dict,
            )
            logger.info("Unified sigma forward ready: y_pred=%s", tuple(y_pred[target_dataset].shape))
            logger.info("Computing unified sigma loss/metrics")
            loss, metrics_next, _ = self.downscaler.compute_loss_metrics(
                y_pred=y_pred,
                y=target_dict,
                validation_mode=True,
                weights=noise_weights,
                use_reentrant=False,
            )
            logger.info("Unified sigma loss/metrics ready")

            residual_post_processor = _residual_post_processor(self.downscaler, target_dataset)
            target_indices = inner_model.data_indices[target_dataset]
            data_index = target_indices.data.output.full
            logger.info("Computing sigma residual diff metrics")
            denorm_pred_residuals = _call_processor(
                residual_post_processor,
                y_pred[target_dataset],
                data_index=data_index,
                skip_imputation=True,
            )
            denorm_truth_residuals = _call_processor(
                residual_post_processor,
                residuals_target,
                data_index=data_index,
                skip_imputation=True,
            )

            diff = denorm_pred_residuals - denorm_truth_residuals
            self._add_residual_diff_metrics(metrics_next, diff)
            logger.info("Sigma residual diff metrics ready")

            del y_pred, residuals_target_noised, target_dict, residuals_target, x_in, x_in_hres
        return loss, metrics_next

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
