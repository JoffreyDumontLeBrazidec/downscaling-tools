from __future__ import annotations

import argparse
import inspect
import json
import os
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from manual_inference.checkpoints import (
    adapt_config_hpc,
    get_checkpoint,
    get_datamodule,
    to_omegaconf,
)
from manual_inference.config import DATASET_PATH_REWRITE_PREFIXES
from manual_inference.config import DEFAULT_CKPT_ROOT
from manual_inference.config import DEFAULT_EXPERIMENTS_DIR
from manual_inference.config import DEFAULT_EXTRA_ARGS_JSON

# Re-export for backward compatibility — external callers import these from here.
__all__ = ["DEFAULT_EXTRA_ARGS_JSON"]
from manual_inference.input_data_construction.bundle import extract_target_from_bundle_dataset
from manual_inference.input_data_construction.bundle import check_input_distribution
from manual_inference.input_data_construction.bundle import load_inputs_from_bundle_numpy
from manual_inference.input_data_construction.bundle import previous_step_bundle_path
from manual_inference.input_data_construction.bundle import open_bundle_dataset
from manual_inference.input_data_construction.bundle import parse_channel_subset_csv as _parse_channel_subset_csv
from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.dataset import OUTPUT_WEATHER_STATE_MODE_CHOICES
from manual_inference.prediction.dataset import resolve_output_weather_states
from manual_inference.prediction.utils import extract_filtered_input_from_output

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Matches any Jupiter .runtime_datasets/<run_id>/ prefix (e.g. o1280_406905).
# These are ephemeral symlink farms whose zarr filenames mirror the AG local mirror.
_JUPITER_RUNTIME_RE = re.compile(
    r"^/e/home/jusers/dumontlebrazidec1/jupiter/dev/\.runtime_datasets/[^/]+/"
)
_JUPITER_RUNTIME_LOCAL = "/home/mlx/ai-ml/datasets/"

# Unified multi-ds predict_step kwarg routing.
_NOISE_SCHEDULER_KEYS = {"num_steps", "sigma_max", "sigma_min", "rho", "schedule_type"}
_SAMPLER_KEYS = {"sampler", "S_churn", "S_min", "S_max", "S_noise"}


def _model_takes_lead_hours(inference_model) -> bool:
    """True iff the checkpoint carries a lead-conditioned hres_branch (fine-scale epic, 2026-09-06)."""
    inner = getattr(inference_model, "model", inference_model)
    branch = getattr(inner, "hres_branch", None)
    return branch is not None and getattr(branch, "lead_embed", None) is not None


_BUNDLE_LEAD_RE = re.compile(r"_step(\d{3})h_input_bundle\.nc$")


def _bundle_lead_hours(bundle_nc) -> float | None:
    m = _BUNDLE_LEAD_RE.search(str(bundle_nc)) if bundle_nc is not None else None
    return float(int(m.group(1))) if m else None


def _predict_with_compatible_kwargs(*, inference_model, batch, model_comm_group, extra_args: dict):
    """Call predict_step with the unified dict-batch convention.

    Unified multi-ds models take ``predict_step(batch={"in_lres":..., "in_hres":...}, ...)``.
    Sampling overrides are routed by inspecting the predict_step signature: either a
    single ``extra_args`` dict, or split ``noise_scheduler_params``/``sampler_params``.
    """
    predict_params = inspect.signature(inference_model.predict_step).parameters
    predict_kwargs = {"model_comm_group": model_comm_group}

    if "extra_args" in predict_params:
        predict_kwargs["extra_args"] = extra_args
    else:
        noise_scheduler_params = {k: v for k, v in extra_args.items() if k in _NOISE_SCHEDULER_KEYS}
        sampler_params = {k: v for k, v in extra_args.items() if k in _SAMPLER_KEYS}
        remaining_params = {
            k: v for k, v in extra_args.items() if k not in _NOISE_SCHEDULER_KEYS | _SAMPLER_KEYS
        }
        if "noise_scheduler_params" in predict_params and noise_scheduler_params:
            predict_kwargs["noise_scheduler_params"] = noise_scheduler_params
        else:
            predict_kwargs.update(noise_scheduler_params)
        if "sampler_params" in predict_params and sampler_params:
            predict_kwargs["sampler_params"] = sampler_params
        else:
            predict_kwargs.update(sampler_params)
        predict_kwargs.update(remaining_params)

    return inference_model.predict_step(batch, **predict_kwargs)


def _rewrite_dataset_paths_in_place(node):
    if OmegaConf.is_config(node):
        container = OmegaConf.to_container(node, resolve=False)
        rewritten = _rewrite_dataset_paths_in_place(container)
        return OmegaConf.create(rewritten)
    if isinstance(node, dict):
        for k, v in list(node.items()):
            node[k] = _rewrite_dataset_paths_in_place(v)
        return node
    if isinstance(node, list):
        return [_rewrite_dataset_paths_in_place(v) for v in node]
    if isinstance(node, tuple):
        return tuple(_rewrite_dataset_paths_in_place(v) for v in node)
    if isinstance(node, str):
        # Imported checkpoints may carry absolute dataset roots from multiple remote
        # sites. Rewrite those to the canonical local mirror when the target exists.
        for remote_prefix, local_prefix in DATASET_PATH_REWRITE_PREFIXES:
            if node.startswith(remote_prefix):
                candidate = node.replace(remote_prefix, local_prefix, 1)
                if os.path.exists(candidate):
                    return candidate
        # Fallback: any Jupiter .runtime_datasets/<run_id>/ → local AG mirror.
        # Prevents breakage when new Jupiter training runs use a different run ID.
        m = _JUPITER_RUNTIME_RE.match(node)
        if m:
            candidate = _JUPITER_RUNTIME_LOCAL + node[m.end():]
            if os.path.exists(candidate):
                return candidate
        return node
    return node


def _split_ckpt_path(path: str) -> tuple[str, str, str, str]:
    ckpt_path = os.path.abspath(os.path.expanduser(path))
    name_ckpt = os.path.basename(ckpt_path)
    name_exp = os.path.basename(os.path.dirname(ckpt_path))
    dir_exp = os.path.dirname(os.path.dirname(ckpt_path))
    return ckpt_path, dir_exp, name_exp, name_ckpt


def _get_parallel_info() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    return global_rank, local_rank, world_size


def _resolve_device(requested_device: str, local_rank: int) -> str:
    if requested_device == "cuda" and torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return requested_device


def _init_model_comm_group(device: str, global_rank: int, world_size: int):
    if world_size <= 1:
        return None
    backend = "nccl" if str(device).startswith("cuda") else "gloo"
    if dist.is_initialized():
        return dist.new_group(list(range(world_size)))
    if os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"):
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=global_rank,
        )
        return dist.new_group(list(range(world_size)))

    # Slurm fallback, mirrors distributed/utils.py behavior.
    from distributed.utils import init_parallel

    return init_parallel(device, global_rank, world_size)


def _resolve_ckpt_path(
    name_ckpt: str,
    ckpt_root: str,
    *,
    allow_inference_companion: bool = False,
) -> str:
    raw = os.path.expanduser(name_ckpt)
    raw_name = os.path.basename(raw)
    if raw_name.startswith("inference-") and raw_name.endswith(".ckpt"):
        if allow_inference_companion:
            return raw if os.path.isabs(raw) else os.path.join(os.path.expanduser(ckpt_root), raw)
        raise ValueError(
            "Pass the base checkpoint path, not the inference companion. "
            f"Got {raw_name}; expected the matching non-inference .ckpt file."
        )
    if os.path.isabs(raw):
        return raw
    root = os.path.expanduser(ckpt_root)
    if raw.endswith(".ckpt"):
        return os.path.join(root, raw)
    run_dir = os.path.join(root, raw)
    last_ckpt = os.path.join(run_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    ckpt_candidates = sorted(Path(run_dir).glob("*.ckpt"))
    primary_candidates = [p for p in ckpt_candidates if not p.name.startswith("inference-")]
    if len(primary_candidates) == 1:
        return str(primary_candidates[0])
    if not ckpt_candidates:
        raise FileNotFoundError(
            f"No checkpoint file found under {run_dir}. Expected last.ckpt or one explicit *.ckpt file."
        )
    if not primary_candidates:
        names = ", ".join(p.name for p in ckpt_candidates[:5])
        raise FileNotFoundError(
            f"Only inference companion checkpoint(s) found under {run_dir} ({names}). "
            "Pass the matching base .ckpt path explicitly or restore the base checkpoint file."
        )
    names = ", ".join(p.name for p in primary_candidates[:5])
    raise FileNotFoundError(
        f"Multiple base checkpoint files found under {run_dir} ({names}). Pass an explicit --name-ckpt path."
    )




def load_objects(
    *,
    ckpt_path: str,
    device: str,
    validation_frequency: str,
    precision: str,
    num_gpus_per_model_override: int | None = None,
):
    ckpt_path, dir_exp, name_exp, name_ckpt = _split_ckpt_path(ckpt_path)
    checkpoint, config_checkpoint = get_checkpoint(dir_exp, name_exp, name_ckpt)
    # Unified multi-ds: config is checkpoint-native. Inject local paths from env vars
    # (DATA_DIR/GRID_DIR/RESIDUAL_STATISTICS_DIR/INTER_MAT_DIR are resolved via the
    # checkpoint's system.input ${oc.env:...} entries; the runtime must export them).
    config_checkpoint = adapt_config_hpc(config_checkpoint)
    config_for_datamodule = to_omegaconf(config_checkpoint)
    config_for_datamodule = _rewrite_dataset_paths_in_place(config_for_datamodule)
    config_for_datamodule.dataloader.validation.frequency = validation_frequency
    if hasattr(config_for_datamodule.dataloader.validation, "num_workers"):
        config_for_datamodule.dataloader.validation.num_workers = 0
    # Bundle-based inference needs template tensors on the full grid. Some checkpoints
    # carry a multi-GPU-per-model training setting that shards template grids.
    if num_gpus_per_model_override is not None and hasattr(config_for_datamodule, "hardware"):
        config_for_datamodule.hardware.num_gpus_per_model = int(num_gpus_per_model_override)
    if num_gpus_per_model_override is not None and hasattr(config_for_datamodule.dataloader, "read_group_size"):
        config_for_datamodule.dataloader.read_group_size = int(num_gpus_per_model_override)

    inference_model = torch.load(
        os.path.join(dir_exp, name_exp, "inference-" + name_ckpt),
        map_location=torch.device(device),
        weights_only=False,
    ).to(device)
    # Halo branch: processor blocks cache halo_info/partition as pickled attrs. Under
    # model-parallel inference each rank would reuse rank-0's cache and OOB its per-rank
    # tensor; invalidate so the cache rebuilds on first forward.
    for _mod in inference_model.modules():
        if hasattr(_mod, "_cached_halo_info"):
            _mod._cached_halo_info = None
        if hasattr(_mod, "_cached_partition"):
            _mod._cached_partition = None
    if str(device).startswith("cuda"):
        if precision == "fp16":
            inference_model = inference_model.half()
        elif precision == "bf16":
            inference_model = inference_model.bfloat16()
    # Memory lever MI_BF16_SAMPLER=1 (2026-07-13): cast the sampler conditioning (upsampled
    # residual + hres forcings returned by _before_sampling) to bf16 AFTER the fp32 sparse
    # residual has run. Drops the full-grid diffusion sampler state fp32->bf16 (~7 GB/rank),
    # which is what OOMs 1-GPU global-o1280 pristine inference on 96 GB GH200 (holds ~84 GB
    # fp32 + a 12.6 GB fp32 layernorm transient). Does NOT feed bf16 to torch.sparse.mm
    # (unsupported: addmm_sparse_cuda not implemented for BFloat16). Keeps residual fp32 +
    # autocast-bf16 compute = same precision class as the accepted b785-375k eval route.
    if precision == "bf16" and os.environ.get("MI_BF16_SAMPLER", "0") == "1":
        _core = getattr(inference_model, "model", None)
        if _core is not None and hasattr(_core, "_before_sampling"):
            _orig_before_sampling = _core._before_sampling
            def _before_sampling_bf16(*a, __orig=_orig_before_sampling, **k):
                bsd, gss = __orig(*a, **k)
                bsd = tuple(
                    t.bfloat16() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                    for t in bsd
                )
                return bsd, gss
            _core._before_sampling = _before_sampling_bf16
    graph_data = inference_model.graph_data
    datamodule = get_datamodule(config_for_datamodule, graph_data)
    return inference_model, datamodule, dir_exp, name_exp


# Backward-compatible alias.
_load_objects = load_objects


def _parse_members(value: str, max_members: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(max_members))
    members = [int(v.strip()) for v in value.split(",") if v.strip()]
    invalid = [m for m in members if m < 0 or m >= max_members]
    if invalid:
        raise ValueError(
            f"Requested member(s) {invalid} out of range for available members [0, {max_members - 1}]."
        )
    return members


def _parse_output_weather_states(value: str) -> list[str] | None:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    return requested or None


def _predict_from_dataloader(
    *,
    inference_model,
    datamodule,
    device: str,
    idx: int,
    n_samples: int,
    members: Sequence[int],
    extra_args: dict,
    precision: str,
    model_comm_group,
    output_weather_state_mode: str = "all",
    output_weather_states: Sequence[str] | None = None,
    split: str = "valid",
):
    ds = datamodule.ds_train if split == "train" else datamodule.ds_valid
    data = ds.data
    x_in = np.asarray(data["in_lres"][idx : idx + n_samples])  # [sample, vars, ens, grid]
    x_in_hres = np.asarray(data["in_hres"][idx : idx + n_samples])
    y = np.asarray(data["out_hres"][idx : idx + n_samples])

    x_in = np.transpose(x_in, (0, 2, 3, 1))  # [sample, ens, grid, vars]
    x_in_hres = np.transpose(x_in_hres, (0, 2, 3, 1))
    y = np.transpose(y, (0, 2, 3, 1))

    name_to_idx_in = datamodule.data_indices["in_lres"].data.input.name_to_index
    name_to_idx_out = datamodule.data_indices["out_hres"].model.output.name_to_index

    # Downscaling models expect full lres input (including forcings) — the model's
    # normalizer has parameters for all input channels.  Only filter the *output
    # reference* arrays (x_out, y_out) later; keep x_in unfiltered for predict_step.
    x_in_full = x_in  # keep full-channel copy for the model

    lon_lres = np.asarray(datamodule.supporting_arrays["in_lres"]["longitudes"])
    lat_lres = np.asarray(datamodule.supporting_arrays["in_lres"]["latitudes"])
    lon_hres = np.asarray(datamodule.supporting_arrays["out_hres"]["longitudes"])
    lat_hres = np.asarray(datamodule.supporting_arrays["out_hres"]["latitudes"])
    dates = np.asarray(ds.datasets["out_hres"].dates[idx : idx + n_samples])
    full_weather_states = list(name_to_idx_out.keys())
    weather_states, selected_indices = resolve_output_weather_states(
        weather_states=full_weather_states,
        mode=output_weather_state_mode,
        explicit_weather_states=output_weather_states,
    )

    y_pred = np.zeros(
        (n_samples, len(members), lon_hres.shape[0], len(weather_states)),
        dtype=np.float32,
    )

    amp_enabled = str(device).startswith("cuda") and precision in {"fp16", "bf16"}
    if os.environ.get("MI_BF16_NATIVE", "0") == "1" and precision == "bf16":
        amp_enabled = False  # native bf16 layernorm (~6.3GB) vs autocast fp32 transient (~12.6GB); OOM fix for 1-GPU global-o1280
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    if _model_takes_lead_hours(inference_model):
        raise NotImplementedError("from-dataloader does not carry the forecast lead; use from-bundle for a lead-conditioned model")
    for i_sample in range(n_samples):
        for j, m in enumerate(members):
            x_l = torch.from_numpy(x_in_full[i_sample, m]).to(device)[None, None, None, ...]
            x_h = torch.from_numpy(x_in_hres[i_sample, m]).to(device)[None, None, None, ...]
            batch = {"in_lres": x_l, "in_hres": x_h}
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    pred = _predict_with_compatible_kwargs(
                        inference_model=inference_model,
                        batch=batch,
                        model_comm_group=model_comm_group,
                        extra_args=extra_args,
                    )
            pred_tensor = pred["out_hres"] if isinstance(pred, dict) else pred
            y_pred[i_sample, j] = (
                pred_tensor[0, 0, 0][..., selected_indices].detach().cpu().numpy().astype(np.float32)
            )

    if not members:
        raise ValueError("No members selected. Pass at least one member id.")
    # For output comparison, filter x_in to output-matching variables
    x_in_filtered, _ = extract_filtered_input_from_output(
        x_in, name_to_idx_in, name_to_idx_out
    )
    x_out = x_in_filtered[:, members, :, :][..., selected_indices]
    y_out = y[:, members, :, :][..., selected_indices]
    return (
        x_out,
        y_out,
        y_pred,
        lon_lres,
        lat_lres,
        lon_hres,
        lat_hres,
        weather_states,
        dates,
    )


def predict_from_bundle(
    *,
    inference_model,
    datamodule,
    device: str,
    bundle_nc: str,
    member_index: int,
    extra_args: dict,
    precision: str,
    model_comm_group,
    output_weather_state_mode: str = "all",
    output_weather_states: Sequence[str] | None = None,
):
    if member_index != 0:
        raise ValueError(
            "Bundle inputs are single-member bundles. Use member_index=0 and select the desired "
            "ensemble member while building the bundle."
        )

    name_to_idx_lres = datamodule.data_indices["in_lres"].data.input.name_to_index
    name_to_idx_hres = datamodule.data_indices["in_hres"].data.input.name_to_index
    name_to_idx_out = datamodule.data_indices["out_hres"].model.output.name_to_index

    bundle = open_bundle_dataset(bundle_nc)
    try:
        (
            x_lres_np,
            x_hres_np,
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
        ) = load_inputs_from_bundle_numpy(
            bundle,
            name_to_idx_lres,
            name_to_idx_hres,
            # De-accumulation needs the previous step's bundle, a sibling file;
            # `bundle` is already an open Dataset, so resolve it from the path.
            prev_bundle=previous_step_bundle_path(bundle_nc),
        )
        # Sanity-check the inputs against the training distribution before spending
        # a GPU-hour on them. Advisory only: it logs and never refuses.
        check_input_distribution(
            inference_model,
            x_lres_np,
            name_to_idx_lres,
            label=Path(str(bundle_nc)).name if bundle_nc is not None else "",
        )
        x_in = torch.from_numpy(x_lres_np).to(device)[None, None, None, ...]
        x_in_hres = torch.from_numpy(x_hres_np).to(device)[None, None, None, ...]
        # Opt-in memory lever (MI_BF16_INPUTS=1): cast inputs so diffusion sampler state
        # (full-grid, per-rank, NOT sharded) is bf16 instead of fp32 — halves the ~14 GB/rank
        # sampler footprint that OOMs global-o1280 pristine inference on 40 GB A100s.
        # Normalized-space bf16; same precision class as the b785-375k unified eval route.
        if (
            str(device).startswith("cuda")
            and precision == "bf16"
            and os.environ.get("MI_BF16_INPUTS", "0") == "1"
        ):
            x_in = x_in.bfloat16()
            x_in_hres = x_in_hres.bfloat16()

        amp_enabled = str(device).startswith("cuda") and precision in {"fp16", "bf16"}
        if os.environ.get("MI_BF16_NATIVE", "0") == "1" and precision == "bf16":
            amp_enabled = False  # native bf16 layernorm (~6.3GB) vs autocast fp32 transient (~12.6GB); OOM fix for 1-GPU global-o1280
        amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        # Unified multi-ds: predict_step takes a dict batch.
        batch = {"in_lres": x_in, "in_hres": x_in_hres}
        # lead-conditioned hres_branch (2026-09-06): the lead comes from the bundle file name and
        # travels as a plain predict_step kwarg; models without the option never see it.
        if _model_takes_lead_hours(inference_model):
            lead_hours = _bundle_lead_hours(bundle_nc)
            if lead_hours is None:
                raise ValueError(f"lead-conditioned model but no _stepNNNh in bundle name: {bundle_nc}")
            extra_args = {**extra_args, "lead_hours": lead_hours}
            logging.getLogger(__name__).info("lead_hours=%s passed to the lead-conditioned branch (%s)", lead_hours, Path(str(bundle_nc)).name)
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                pred = _predict_with_compatible_kwargs(
                    inference_model=inference_model,
                    batch=batch,
                    model_comm_group=model_comm_group,
                    extra_args=extra_args,
                )
        pred_tensor = pred["out_hres"] if isinstance(pred, dict) else pred

        weather_states_full = list(name_to_idx_out.keys())
        weather_states, selected_indices = resolve_output_weather_states(
            weather_states=weather_states_full,
            mode=output_weather_state_mode,
            explicit_weather_states=output_weather_states,
        )

        x_np = x_in[0, 0, 0].detach().cpu().numpy().astype(np.float32)
        pred_np = pred_tensor[0, 0, 0][..., selected_indices].detach().cpu().numpy().astype(np.float32)
        dates = None
        # Free the sampler forward's cached GPU memory now that the output is on CPU, before
        # the unsharded full-grid x_interp export (~9 GB on global o1280) reallocates on GPU.
        # Needed on aarch64 GH200 (expandable_segments inert) to avoid a rank0 gather OOM.
        import gc as _gc
        _gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        x_np, _ = extract_filtered_input_from_output(
            x_np, name_to_idx_lres, name_to_idx_out
        )
        x_np = x_np[..., selected_indices]

        local_graph_mask = getattr(inference_model, "_local_graph_cut_data_mask", None)
        if local_graph_mask is not None:
            local_graph_mask = local_graph_mask.detach().cpu().numpy().astype(bool)
            if pred_np.shape[-2] != int(local_graph_mask.sum()):
                raise ValueError(
                    "Local cut-graph prediction size mismatch: "
                    f"model returned {pred_np.shape[-2]} hres nodes but mask selects {int(local_graph_mask.sum())}."
                )
            lon_hres = np.asarray(lon_hres)[local_graph_mask]
            lat_hres = np.asarray(lat_hres)[local_graph_mask]

        y_np = None
        target_np, found_target_channels = extract_target_from_bundle_dataset(bundle, weather_states)
        if target_np is not None:
            if local_graph_mask is not None:
                target_np = target_np[local_graph_mask, :]
            y_np = target_np[None, None, ...]
            if found_target_channels < len(weather_states):
                print(
                    f"Bundle target coverage: {found_target_channels}/{len(weather_states)} weather states "
                    f"(missing channels will be NaN in y)."
                )

        return (
            x_np[None, ...],
            y_np,
            pred_np[None, None, ...],
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
            weather_states,
            dates,
        )
    finally:
        try:
            bundle.close()
        except Exception:
            pass


# Backward-compatible alias.
_predict_from_bundle = predict_from_bundle


def _compute_x_interp_for_export(
    *,
    inference_model,
    x: np.ndarray,
    device: str,
    model_comm_group,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 3:
        x_arr = x_arr[:, None, ...]
    elif x_arr.ndim != 4:
        raise ValueError(f"Unsupported x shape for x_interp export: {x_arr.shape}")

    x_tensor = torch.from_numpy(x_arr).to(device)
    with torch.inference_mode():
        if hasattr(inference_model, "interpolate_down"):
            member_interp = []
            for member_idx in range(x_tensor.shape[1]):
                member_x = x_tensor[:, member_idx, ...]
                try:
                    interp = inference_model.interpolate_down(member_x, grad_checkpoint=False)
                except TypeError:
                    interp = inference_model.interpolate_down(member_x)
                if interp.ndim != 3:
                    raise ValueError(
                        f"Unexpected interpolate_down output shape {tuple(interp.shape)} for member {member_idx}"
                    )
                member_interp.append(interp)
            x_interp = torch.stack(member_interp, dim=1)
        else:
            model = getattr(inference_model, "model", inference_model)
            if not hasattr(model, "apply_interpolate_to_high_res"):
                raise RuntimeError(
                    "Inference model cannot export x_interp: missing interpolate_down and apply_interpolate_to_high_res."
                )
            # Unified multi-ds: apply_interpolate_to_high_res accepts exactly one ensemble
            # member — input (batch, 1, grid_lres, vars), output (batch, 1, 1, grid_hres,
            # vars). Loop per member (as the per-member dev/multi runs did) and stack to the
            # [batch, member, grid_hres, vars] layout the interpolate_down branch yields.
            member_interp = []
            for member_idx in range(x_tensor.shape[1]):
                member_x = x_tensor[:, member_idx : member_idx + 1, ...]
                try:
                    mi = model.apply_interpolate_to_high_res(
                        member_x,
                        grid_shard_sizes=None,
                        model_comm_group=model_comm_group,
                    )
                except TypeError:
                    mi = model.apply_interpolate_to_high_res(member_x)
                # Collapse the size-1 time/ensemble axes -> (batch, grid_hres, vars),
                # matching the interpolate_down branch's per-member ndim-3 output.
                mi = mi.reshape(mi.shape[0], mi.shape[-2], mi.shape[-1])
                member_interp.append(mi)
            x_interp = torch.stack(member_interp, dim=1)

    return x_interp.detach().cpu().numpy().astype(np.float32)



def _parse_json(value: str) -> dict:
    if not value:
        return {}
    return json.loads(value)


def _fail_if_missing_truth(*, y: np.ndarray | None, context: str) -> None:
    if y is None:
        raise SystemExit(
            "Missing target truth `y` in predictions output. "
            f"Context={context}. Rebuild bundles with complete target_hres_* fields."
        )


def _coerce_missing_truth_to_nan(
    *,
    y: np.ndarray | None,
    y_pred: np.ndarray,
    context: str,
    allow_missing_target_unsafe: bool,
) -> tuple[np.ndarray, bool]:
    if y is not None:
        return y, False
    if not allow_missing_target_unsafe:
        _fail_if_missing_truth(y=y, context=context)
    print(
        "WARNING: missing target truth `y` in predictions output. "
        f"Context={context}. Writing all-NaN y because --allow-missing-target-unsafe was set. "
        "Treat this artifact as prediction-only and non-canonical for truth-aware evaluation."
    )
    return np.full_like(y_pred, np.nan, dtype=np.float32), True


def _validate_output_path(
    *,
    out_path: Path,
    allow_existing_output_dir: bool,
) -> None:
    resolved = out_path.expanduser().resolve()
    parent = resolved.parent

    # Guard against accidental nested run layouts like <old_run>/<new_run>/...
    parent_name = parent.name
    grandparent_name = parent.parent.name if parent.parent != parent else ""
    if _RUN_NAME_RE.fullmatch(parent_name) and _RUN_NAME_RE.fullmatch(grandparent_name):
        if parent.parent.exists() and (parent.parent / "logs").is_dir():
            raise SystemExit(
                f"Unsafe nested output path detected: {resolved}. "
                "Refusing to place a new run folder under an existing run folder."
            )

    if parent.exists():
        if not allow_existing_output_dir and any(parent.iterdir()):
            raise SystemExit(
                f"Output directory already exists and is not empty: {parent}. "
                "Use a fresh run folder or pass --allow-existing-output-dir explicitly."
            )
    else:
        parent.mkdir(parents=True, exist_ok=False)

    if resolved.exists():
        raise SystemExit(
            f"Refusing to overwrite existing output file: {resolved}. "
            "Use a fresh run folder or rename the output file."
        )


def main() -> None:
    ckpt_root_default = os.environ.get(
        "AIFS_CKPT_ROOT", DEFAULT_CKPT_ROOT
    )
    parser = argparse.ArgumentParser(description="Generate predictions.nc from a checkpoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("from-dataloader", help="Use dataloader inputs.")
    p_dl.add_argument("--name-ckpt", required=True)
    p_dl.add_argument("--ckpt-root", default=ckpt_root_default)
    p_dl.add_argument("--device", default="cuda")
    p_dl.add_argument("--idx", type=int, default=0)
    p_dl.add_argument("--n-samples", type=int, default=1)
    p_dl.add_argument("--members", default="0")
    p_dl.add_argument("--validation-frequency", default="50h")
    p_dl.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    p_dl.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p_dl.add_argument(
        "--output-weather-state-mode",
        choices=OUTPUT_WEATHER_STATE_MODE_CHOICES,
        default="all",
        help="Subset saved variables. 'surface-plus-core-pl' keeps all surface outputs plus z_500 and t_850.",
    )
    p_dl.add_argument(
        "--output-weather-states",
        default="",
        help="Explicit CSV override for saved weather states. Overrides --output-weather-state-mode when set.",
    )
    p_dl.add_argument(
        "--slim-output",
        action="store_true",
        help="Write only canonical x/y/y_pred ensemble arrays and skip duplicate x_*/y_*/y_pred_* member views.",
    )
    p_dl.add_argument("--out", default="")
    p_dl.add_argument(
        "--split",
        choices=["valid", "train"],
        default="valid",
        help="Which data split to sample from (default: valid).",
    )
    p_dl.add_argument(
        "--debug-from-dataloader",
        action="store_true",
        help="Required safety switch: from-dataloader is debug-only.",
    )
    p_dl.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Allow writing into a pre-existing non-empty output directory.",
    )

    p_bundle = sub.add_parser("from-bundle", help="Use a prebuilt input bundle NetCDF.")
    p_bundle.add_argument("--name-ckpt", required=True)
    p_bundle.add_argument("--ckpt-root", default=ckpt_root_default)
    p_bundle.add_argument("--device", default="cuda")
    p_bundle.add_argument("--bundle-nc", required=True)
    p_bundle.add_argument("--member-index", type=int, default=0)
    p_bundle.add_argument("--validation-frequency", default="50h")
    p_bundle.add_argument("--extra-args-json", default=DEFAULT_EXTRA_ARGS_JSON)
    p_bundle.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p_bundle.add_argument(
        "--output-weather-state-mode",
        choices=OUTPUT_WEATHER_STATE_MODE_CHOICES,
        default="all",
        help="Subset saved variables. 'surface-plus-core-pl' keeps all surface outputs plus z_500 and t_850.",
    )
    p_bundle.add_argument(
        "--output-weather-states",
        default="",
        help="Explicit CSV override for saved weather states. Overrides --output-weather-state-mode when set.",
    )
    p_bundle.add_argument(
        "--slim-output",
        action="store_true",
        help="Write only canonical x/y/y_pred ensemble arrays and skip duplicate x_*/y_*/y_pred_* member views.",
    )
    p_bundle.add_argument("--out", default="")
    p_bundle.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Allow writing into a pre-existing non-empty output directory.",
    )
    p_bundle.add_argument(
        "--allow-missing-target-unsafe",
        action="store_true",
        help=(
            "Explicitly allow missing target_hres_* truth in bundle predictions by writing "
            "all-NaN y. Unsafe: output is prediction-only and non-canonical for truth-aware evaluation."
        ),
    )

    p_bundle_build = sub.add_parser("build-bundle", help="Create input bundle from GRIB.")
    p_bundle_build.add_argument("--lres-sfc-grib", required=True)
    p_bundle_build.add_argument(
        "--lres-sfc-extra-grib",
        action="append",
        default=[],
        help="Optional extra low-resolution surface GRIB to merge before bundle creation.",
    )
    p_bundle_build.add_argument("--lres-pl-grib", required=True)
    p_bundle_build.add_argument("--hres-grib", required=True)
    p_bundle_build.add_argument(
        "--hres-static-grib",
        default="",
        help="Optional GRIB file used only for high-resolution static fields such as z and lsm.",
    )
    p_bundle_build.add_argument("--target-sfc-grib", default="")
    p_bundle_build.add_argument(
        "--target-sfc-extra-grib",
        action="append",
        default=[],
        help="Optional extra target surface GRIB to merge before bundle creation.",
    )
    p_bundle_build.add_argument("--target-pl-grib", default="")
    p_bundle_build.add_argument(
        "--lres-sfc-channels",
        default="",
        help="Optional CSV override for low-resolution surface bundle channels.",
    )
    p_bundle_build.add_argument(
        "--lres-pl-channels",
        default="",
        help="Optional CSV override for low-resolution pressure-level bundle channels.",
    )
    p_bundle_build.add_argument(
        "--target-sfc-channels",
        default="",
        help="Optional CSV override for target high-resolution surface bundle channels.",
    )
    p_bundle_build.add_argument(
        "--target-pl-channels",
        default="",
        help="Optional CSV override for target high-resolution pressure-level bundle channels.",
    )
    p_bundle_build.add_argument("--allow-missing-target", action="store_true")
    p_bundle_build.add_argument(
        "--allow-missing-target-unsafe",
        action="store_true",
        help=(
            "Explicitly allow creating bundle without target_hres_* fields. "
            "Unsafe: output is prediction-only and non-canonical for truth-aware evaluation."
        ),
    )
    p_bundle_build.add_argument("--out", required=True)
    p_bundle_build.add_argument("--step-hours", type=int, default=None)
    p_bundle_build.add_argument("--member", type=int, default=None)

    args = parser.parse_args()
    global_rank, local_rank, world_size = _get_parallel_info()

    if args.cmd == "build-bundle":
        from manual_inference.input_data_construction.bundle import (
            build_input_bundle_from_grib,
        )
        if args.allow_missing_target:
            raise SystemExit(
                "--allow-missing-target is deprecated. "
                "Use --allow-missing-target-unsafe for an explicit prediction-only escape hatch."
            )

        out = build_input_bundle_from_grib(
            lres_sfc_grib=args.lres_sfc_grib,
            lres_sfc_extra_gribs=args.lres_sfc_extra_grib,
            lres_pl_grib=args.lres_pl_grib,
            hres_grib=args.hres_grib,
            hres_static_grib=args.hres_static_grib or None,
            out_nc=args.out,
            step_hours=args.step_hours,
            member=args.member,
            target_sfc_grib=args.target_sfc_grib or None,
            target_sfc_extra_gribs=args.target_sfc_extra_grib,
            target_pl_grib=args.target_pl_grib or None,
            require_target_fields=not args.allow_missing_target_unsafe,
            lres_sfc_channels=_parse_channel_subset_csv(args.lres_sfc_channels),
            lres_pl_channels=_parse_channel_subset_csv(args.lres_pl_channels),
            target_sfc_channels=_parse_channel_subset_csv(args.target_sfc_channels),
            target_pl_channels=_parse_channel_subset_csv(args.target_pl_channels),
        )
        print(f"Saved bundle: {out}")
        return

    resolved_ckpt = _resolve_ckpt_path(args.name_ckpt, args.ckpt_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "Requested --device cuda, but CUDA is not available on this host. "
            "Refusing to fall back to CPU for diffusion sampling."
        )
    args.device = _resolve_device(args.device, local_rank)
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(int(str(args.device).split(":")[1]))
    model_comm_group = _init_model_comm_group(args.device, global_rank, world_size)

    inference_model, datamodule, dir_exp, name_exp = _load_objects(
        ckpt_path=resolved_ckpt,
        device=args.device,
        validation_frequency=args.validation_frequency,
        precision=args.precision,
    )

    extra_args = _parse_json(args.extra_args_json)
    output_weather_states = _parse_output_weather_states(getattr(args, "output_weather_states", ""))
    if args.cmd == "from-dataloader":
        if not args.debug_from_dataloader:
            raise SystemExit(
                "from-dataloader is debug-only in the new stack. "
                "Pass --debug-from-dataloader to run it intentionally."
            )
        data = (datamodule.ds_train if args.split == "train" else datamodule.ds_valid).data
        _probe = data["in_lres"] if hasattr(data, "keys") else data  # multi-ds: data is dict-keyed {in_lres,in_hres,out_hres}
        max_members = int(np.asarray(_probe[args.idx : args.idx + 1][0]).shape[2])
        members = _parse_members(args.members, max_members)
        (
            x,
            y,
            y_pred,
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
            weather_states,
            dates,
        ) = _predict_from_dataloader(
            inference_model=inference_model,
            datamodule=datamodule,
            device=args.device,
            idx=args.idx,
            n_samples=args.n_samples,
            members=members,
            extra_args=extra_args,
            precision=args.precision,
            model_comm_group=model_comm_group,
            output_weather_state_mode=args.output_weather_state_mode,
            output_weather_states=output_weather_states,
            split=args.split,
        )
        member_ids = members
        y, used_missing_target_unsafe = _coerce_missing_truth_to_nan(
            y=y,
            y_pred=y_pred,
            context="from-dataloader",
            allow_missing_target_unsafe=bool(getattr(args, "allow_missing_target_unsafe", False)),
        )
    elif args.cmd == "from-bundle":
        (
            x,
            y,
            y_pred,
            lon_lres,
            lat_lres,
            lon_hres,
            lat_hres,
            weather_states,
            dates,
        ) = _predict_from_bundle(
            inference_model=inference_model,
            datamodule=datamodule,
            device=args.device,
            bundle_nc=args.bundle_nc,
            member_index=args.member_index,
            extra_args=extra_args,
            precision=args.precision,
            model_comm_group=model_comm_group,
            output_weather_state_mode=args.output_weather_state_mode,
            output_weather_states=output_weather_states,
        )
        member_ids = [args.member_index]
        y, used_missing_target_unsafe = _coerce_missing_truth_to_nan(
            y=y,
            y_pred=y_pred,
            context="from-bundle",
            allow_missing_target_unsafe=bool(getattr(args, "allow_missing_target_unsafe", False)),
        )
    else:
        raise SystemExit("Unknown command")

    import gc as _gc2
    _gc2.collect()
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    x_interp = _compute_x_interp_for_export(
        inference_model=inference_model,
        x=x,
        device=args.device,
        model_comm_group=model_comm_group,
    )

    ds = build_predictions_dataset(
        x=x,
        y=y,
        y_pred=y_pred,
        lon_lres=lon_lres,
        lat_lres=lat_lres,
        lon_hres=lon_hres,
        lat_hres=lat_hres,
        weather_states=weather_states,
        dates=dates,
        member_ids=member_ids,
        x_interp=x_interp,
        include_member_views=not getattr(args, "slim_output", False),
    )
    if used_missing_target_unsafe:
        ds.attrs["missing_target_policy"] = "all_nan_due_to_allow_missing_target_unsafe"
    ds.attrs["checkpoint_path"] = resolved_ckpt
    ds.attrs["sampling_config_json"] = args.extra_args_json
    ds.attrs["validation_frequency"] = args.validation_frequency
    ds.attrs["x_interp_exported"] = 1
    ds.attrs["output_weather_state_mode"] = args.output_weather_state_mode
    ds.attrs["output_weather_states"] = ",".join(weather_states)
    ds.attrs["slim_output"] = int(bool(getattr(args, "slim_output", False)))

    out_path = args.out
    if not out_path:
        experiments_dir = os.environ.get("AIFS_EXPERIMENTS_DIR", DEFAULT_EXPERIMENTS_DIR)
        out_path = os.path.join(
            experiments_dir, name_exp, "predictions.nc"
        )
    out_path = Path(out_path)
    # Output-path validation and write are rank-0 only. In a sharded multi-rank run,
    # rank 0 writes predictions.nc first; if non-zero ranks then re-validated the path
    # they would see the just-written file and abort, failing the whole job even though
    # the output is correct. The dist.barrier() below keeps the ranks in sync.
    if global_rank == 0:
        _validate_output_path(
            out_path=out_path,
            allow_existing_output_dir=bool(getattr(args, "allow_existing_output_dir", False)),
        )
        ds.to_netcdf(out_path)
        print(f"Saved predictions: {out_path}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
