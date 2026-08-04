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

import atexit as _atexit
import logging
import sys as _sys
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
    w_map = extra_args.pop("autoguide_weight_map", None)
    sig_lo = float(extra_args.pop("autoguide_sigma_lo", 0.0))
    sig_hi = float(extra_args.pop("autoguide_sigma_hi", 1.0e9))
    if w in (None, 0, 0.0) and not w_map:
        return None
    if not ckpt:
        raise SystemExit("autoguide_weight(_map) set but autoguide_checkpoint missing")
    if w_map is not None and not isinstance(w_map, dict):
        raise SystemExit("autoguide_weight_map must be a {variable: weight} mapping")
    w = 1.0 if w is None else float(w)
    # load_objects() takes the BASE (training-format) path and loads "inference-"+name
    # itself; the main model path is normalized upstream, so normalize D_bad here too
    # (accept either format, as a user would expect).
    ckpt_p = _Path(str(ckpt)).expanduser()
    inference_only = None
    if ckpt_p.name.startswith("inference-"):
        base = ckpt_p.with_name(ckpt_p.name[len("inference-"):])
        if base.exists():
            ckpt = str(base)
        else:
            inference_only = ckpt_p
    elif not ckpt_p.exists():
        companion = ckpt_p.with_name("inference-" + ckpt_p.name)
        if not companion.exists():
            raise SystemExit("autoguide_checkpoint not found: %s (nor %s)" % (ckpt_p, companion))
        inference_only = companion
    print("[autoguidance] loading D_bad %s (w=%.3g, band [%.3g, %.3g])"
          % (ckpt, w, sig_lo, sig_hi), file=_sys.stderr, flush=True)
    if inference_only is not None:
        # inference-only D_bad: the serialized interface already carries the weights;
        # config/datamodule are irrelevant for a denoiser-only role.
        import torch as _torch
        print("[autoguidance] inference-only D_bad: %s" % inference_only.name,
              file=_sys.stderr, flush=True)
        weak_model = _torch.load(str(inference_only), map_location=device,
                                 weights_only=False)
        try:
            weak_model = weak_model.to(device)
        except Exception:
            pass
    else:
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

    # Per-output-variable weights: gvecs[dataset_key] = (w_vec - 1) over the LAST (vars)
    # dim of that dataset's output tensor, built from data_indices name_to_index.
    gvecs = None
    if w_map:
        di = getattr(strong_inner, "data_indices", None)
        if di is None:
            raise SystemExit("autoguide_weight_map: model exposes no data_indices "
                             "(ds-API models are unsupported; use scalar autoguide_weight)")
        try:
            di_items = dict(di).items()
        except Exception:
            raise SystemExit("autoguide_weight_map: cannot iterate data_indices (%r)" % type(di))
        gvecs, seen = {}, set()
        for dkey, idx in di_items:
            nti = getattr(getattr(getattr(idx, "model", None), "output", None),
                          "name_to_index", None)
            if not nti:
                continue
            vec = torch.full((max(nti.values()) + 1,), w, dtype=torch.float32, device=device)
            for name, wv in w_map.items():
                if name in nti:
                    vec[nti[name]] = float(wv)
                    seen.add(name)
            gvecs[dkey] = vec - 1.0
        missing = set(w_map) - seen
        if not gvecs:
            raise SystemExit("autoguide_weight_map: no dataset in data_indices has "
                             "model.output.name_to_index")
        if missing:
            names = sorted({n for _, idx in di_items for n in getattr(getattr(getattr(
                idx, "model", None), "output", None), "name_to_index", {})})
            raise SystemExit("autoguide_weight_map: unknown variables %s (available: %s)"
                             % (sorted(missing), names))

    calls = {"n": 0, "in_band": 0}

    def _autoguided(*args, **kwargs):
        D = base_fn(*args, **kwargs)
        calls["n"] += 1
        try:
            if args and isinstance(args[0], dict):   # unified dict-API: (x, y, sigma_dict, ...)
                sig = float(next(iter(args[2].values())).reshape(-1)[0].item())
            else:                                    # ds-API: (x_interp, x_hres, y_noised, sigma, ...)
                sig = float(args[3].reshape(-1)[0].item())
            if not (sig_lo <= sig <= sig_hi):
                return D
            calls["in_band"] += 1
            Dw = weak_fn(*args, **kwargs)
            if isinstance(D, dict):
                if gvecs is not None:
                    out = {}
                    for k in D:
                        g = gvecs.get(k)
                        if g is None:
                            raise SystemExit("autoguide_weight_map: no output index for "
                                             "dataset %r (have %s)" % (k, sorted(gvecs)))
                        if D[k].shape[-1] != g.numel():
                            raise SystemExit("autoguide_weight_map: %r has %d channels, "
                                             "weight vector has %d"
                                             % (k, D[k].shape[-1], g.numel()))
                        out[k] = D[k] + g.to(D[k].dtype) * (D[k] - Dw[k].to(D[k].dtype))
                    return out
                return {k: D[k] + (w - 1.0) * (D[k] - Dw[k].to(D[k].dtype)) for k in D}
            if gvecs is not None:
                raise SystemExit("autoguide_weight_map is dict-API only; this model returned "
                                 "a bare tensor (ds-API)")
            Dwt = next(iter(Dw.values())) if isinstance(Dw, dict) else Dw
            return D + (w - 1.0) * (D - Dwt.to(D.dtype))
        except Exception:
            LOG.exception("autoguidance wrap failed; returning unguided D")
            return D

    strong_inner.fwd_with_preconditioning = _autoguided
    # Keep D_bad alive for the process lifetime (the wrapper closes over weak_fn).
    inference_model._autoguide_weak_model = weak_model
    info = {"autoguide_checkpoint": str(ckpt), "autoguide_weight": w,
            "autoguide_weight_map": w_map,
            "autoguide_sigma_lo": sig_lo, "autoguide_sigma_hi": sig_hi}
    inference_model._autoguide_info = info
    inference_model._autoguide_calls = calls
    print("[autoguidance] AUTOGUIDANCE ACTIVE w=%.3g%s D_bad=%s"
          % (w, "" if not w_map else " map=%s" % json.dumps(w_map, sort_keys=True),
             _Path(str(ckpt)).name), file=_sys.stderr, flush=True)

    def _report():
        print("[autoguidance] denoiser calls total=%d guided_in_band=%d"
              % (calls["n"], calls["in_band"]), file=_sys.stderr, flush=True)

    _atexit.register(_report)
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
