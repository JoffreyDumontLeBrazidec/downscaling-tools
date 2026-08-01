"""Thin wrappers around manual inference model-loading utilities."""

from __future__ import annotations

import json

import torch

from ._mi import predict as _mi_predict
_get_parallel_info = _mi_predict._get_parallel_info
_init_model_comm_group = _mi_predict._init_model_comm_group
_load_objects = _mi_predict._load_objects
_resolve_device = _mi_predict._resolve_device

from .graph_cut import activate_local_graph_cut
from .types import PredictionConfig

import logging
from pathlib import Path as _Path

LOG = logging.getLogger(__name__)


def _activate_autoguidance(inference_model, extra_args: dict, config: PredictionConfig,
                           device) -> dict | None:
    """Karras autoguidance in the deployed predict path: D' = w*D - (w-1)*D_weak.

    Reads (and REMOVES) autoguide_* keys from extra_args so they never reach
    predict_step. Loads D_bad with the same loader/runtime, applies the same
    local_scope graph cut, and wraps fwd_with_preconditioning on the strong
    inner model (instance attribute shadows the bound method; predict_step
    passes self.fwd_with_preconditioning to the sampler, covering both Heun
    evals). Returns an info dict (or None when off)."""
    ckpt = extra_args.pop("autoguide_checkpoint", None)
    w = extra_args.pop("autoguide_weight", None)
    sig_lo = float(extra_args.pop("autoguide_sigma_lo", 0.0))
    sig_hi = float(extra_args.pop("autoguide_sigma_hi", 1.0e9))
    if w in (None, 0, 0.0):
        return None
    if not ckpt:
        raise SystemExit("autoguide_weight set but autoguide_checkpoint missing")
    w = float(w)
    # load_objects() takes the BASE (training-format) path and loads "inference-"+name
    # itself; the main model path is normalized upstream, so normalize D_bad here too
    # (accept either format, as a user would expect).
    ckpt_p = _Path(str(ckpt)).expanduser()
    if ckpt_p.name.startswith("inference-"):
        base = ckpt_p.with_name(ckpt_p.name[len("inference-"):])
        if not base.exists():
            raise SystemExit(
                "autoguide_checkpoint given in inference- form but the base checkpoint "
                f"{base} is missing; pass the training-format path")
        ckpt = str(base)
    LOG.info("autoguidance: loading D_bad %s (w=%.3g, band [%.3g, %.3g])", ckpt, w, sig_lo, sig_hi)
    weak_model, _, _, _ = _load_objects(
        ckpt_path=str(ckpt),
        device=device,
        validation_frequency=config.validation_frequency,
        precision=config.precision,
        num_gpus_per_model_override=config.num_gpus_per_model,
    )
    if config.local_scope_json:
        activate_local_graph_cut(weak_model, config.local_scope_json)
    strong_inner = getattr(inference_model, "model", inference_model)
    weak_inner = getattr(weak_model, "model", weak_model)
    base_fn = strong_inner.fwd_with_preconditioning
    weak_fn = weak_inner.fwd_with_preconditioning

    def _autoguided(*args, **kwargs):
        D = base_fn(*args, **kwargs)
        try:
            if args and isinstance(args[0], dict):   # unified dict-API: (x, y, sigma_dict, ...)
                sig = float(next(iter(args[2].values())).reshape(-1)[0].item())
            else:                                    # ds-API: (x_interp, x_hres, y_noised, sigma, ...)
                sig = float(args[3].reshape(-1)[0].item())
            if not (sig_lo <= sig <= sig_hi):
                return D
            Dw = weak_fn(*args, **kwargs)
            if isinstance(D, dict):
                return {k: D[k] + (w - 1.0) * (D[k] - Dw[k].to(D[k].dtype)) for k in D}
            Dwt = next(iter(Dw.values())) if isinstance(Dw, dict) else Dw
            return D + (w - 1.0) * (D - Dwt.to(D.dtype))
        except Exception:
            LOG.exception("autoguidance wrap failed; returning unguided D")
            return D

    strong_inner.fwd_with_preconditioning = _autoguided
    # Keep D_bad alive for the process lifetime (the wrapper closes over weak_fn).
    inference_model._autoguide_weak_model = weak_model
    info = {"autoguide_checkpoint": str(ckpt), "autoguide_weight": w,
            "autoguide_sigma_lo": sig_lo, "autoguide_sigma_hi": sig_hi}
    inference_model._autoguide_info = info
    return info


def setup_distributed(config: PredictionConfig) -> tuple[str, object | None, int, int, int]:
    """Resolve device and initialize the model communication group."""

    global_rank, local_rank, world_size = _get_parallel_info()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "Requested --device cuda, but CUDA is not available on this host. "
            "Refusing to fall back to CPU for diffusion sampling."
        )

    device = _resolve_device(config.device, local_rank)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(int(str(device).split(":")[1]))

    if config.num_gpus_per_model > 1 and world_size != config.num_gpus_per_model:
        raise SystemExit(
            f"Expected world_size={config.num_gpus_per_model} for model-parallel inference, "
            f"got {world_size}. Launch with matching srun --ntasks."
        )

    model_comm_group = _init_model_comm_group(device, global_rank, world_size)
    return device, model_comm_group, global_rank, local_rank, world_size


def load_inference_model(
    config: PredictionConfig,
) -> tuple[object, object, dict, str, object | None, int, int, int]:
    """Load the inference model, datamodule, parsed sampling args, and runtime metadata."""

    device, model_comm_group, global_rank, local_rank, world_size = setup_distributed(config)
    extra_args = json.loads(config.extra_args_json) if config.extra_args_json else {}
    inference_model, datamodule, _, _ = _load_objects(
        ckpt_path=str(config.checkpoint_path),
        device=device,
        validation_frequency=config.validation_frequency,
        precision=config.precision,
        num_gpus_per_model_override=config.num_gpus_per_model,
    )
    if config.local_scope_json:
        activate_local_graph_cut(inference_model, config.local_scope_json)
    ag_info = _activate_autoguidance(inference_model, extra_args, config, device)
    if ag_info:
        LOG.info("autoguidance ACTIVE: %s", ag_info)
    return (
        inference_model,
        datamodule,
        extra_args,
        device,
        model_comm_group,
        global_rank,
        local_rank,
        world_size,
    )
