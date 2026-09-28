"""Model loading and forward helpers for interpretability experiments.

Thin wrapper around manual_inference model loading. Provides a clean
InterpModelBundle plus the three forward entry points every tool uses:

  - denoise_at_sigma       : single denoiser call at fixed sigma (no grad)
  - denoise_at_sigma_grad  : same, grad-enabled, for Integrated Gradients
  - sample_full            : full Heun sampling trajectory (end-to-end output)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

# Environment variables required by the Hydra/OmegaConf config resolution.
_ENV_DEFAULTS = {
    "DATA_DIR": "",
    "DEV": "/home/ecm5702/dev/",
    "OUTPUT": "/ec/res4/scratch/ecm5702/aifs",
    "GRID_DIR": "/home/mlx/ai-ml/grids/",
    "INTER_MAT_DIR": "/home/ecm5702/hpcperm/data/inter_mat",
    "RESIDUAL_STATISTICS_DIR": "/home/ecm5702/hpcperm/data/residuals_statistics/",
    "ANEMOI_BASE_SEED": "756",
}
for k, v in _ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

from manual_inference.prediction.predict import load_objects as _load_objects

LOGGER = logging.getLogger(__name__)

SURFACE_TARGETS = ["10u", "10v", "2t", "msl", "tp", "cp"]


@dataclass
class InterpModelBundle:
    """Everything needed for interpretability experiments."""

    model: object  # AnemoiModelInterface
    datamodule: object
    device: str

    @property
    def inner_model(self):
        """The underlying AnemoiDownscalingModelEncProcDec."""
        return self.model.model

    @property
    def pre_processors(self):
        return self.model.pre_processors

    @property
    def post_processors(self):
        return self.model.post_processors

    @property
    def data_indices(self):
        return self.model.data_indices

    @property
    def graph_data(self):
        return self.model.graph_data


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    precision: str = "fp32",
    validation_frequency: str = "6h",
    num_gpus_per_model: int = 1,
) -> InterpModelBundle:
    """Load an AIFSDD checkpoint for interpretability analysis.

    Parameters
    ----------
    checkpoint_path : str
        Path to training checkpoint (not the inference- companion).
    device : str
        'cuda' or 'cpu'.
    precision : str
        'fp32', 'fp16', or 'bf16'.
    validation_frequency : str
        Validation frequency for datamodule (controls date sampling).
    num_gpus_per_model : int
        Grid-sharding degree. 1 = single-GPU (default). >1 shards the model's
        grid templates across that many ranks (for model-parallel inference of
        large lanes like O1280; the caller must set up the model_comm_group).
    """
    inference_model, datamodule, _, _ = _load_objects(
        ckpt_path=str(checkpoint_path),
        device=device,
        validation_frequency=validation_frequency,
        precision=precision,
        num_gpus_per_model_override=num_gpus_per_model,
    )
    # Disable torch.compile on the interpolation function — it caches shapes
    # and breaks with dynamic batch sizes in interpretability experiments. The
    # method is named _interpolate_to_high_res on the ds model and
    # apply_interpolate_to_high_res on the unified model; patch whichever exists.
    import types
    for _meth in ("_interpolate_to_high_res", "apply_interpolate_to_high_res"):
        f = getattr(inference_model.model, _meth, None)
        if f is None:
            continue
        # Only un-compile if it is actually a torch.compile wrapper. getattr returns an
        # already-BOUND method; re-binding an uncompiled bound method (unified model)
        # double-binds self and breaks the call. The ds method carries __wrapped__.
        if hasattr(f, "__wrapped__"):
            orig_func = f
            while hasattr(orig_func, "__wrapped__"):
                orig_func = orig_func.__wrapped__
            setattr(inference_model.model, _meth,
                    types.MethodType(orig_func, inference_model.model))
        break
    # Force the memory-efficient 'edges' grid-sharding for model-parallel inference.
    # Old unified checkpoints (e.g. b785) bake shard_strategy='heads' on the encoder/
    # decoder mappers + processor, whose all_to_all_transpose materialises full QKV over
    # the entire (e.g. O1280 6.6M-node) grid on EVERY rank -> OOM on A100-40GB regardless
    # of rank count. 'edges' shards+chunks the graph locally -- exactly what every unified
    # eval lane uses (the retired CODEX_UNIFIED_SHARD_STRATEGY=edges). Numerically
    # equivalent; only the work partition changes. Only meaningful when actually sharded.
    if num_gpus_per_model > 1:
        _n_over = 0
        for _m in inference_model.modules():
            if getattr(_m, "shard_strategy", None) == "heads":
                _m.shard_strategy = "edges"
                _n_over += 1
        if _n_over:
            LOGGER.info(
                "load_model: forced shard_strategy heads->edges on %d modules "
                "for O1280-safe model-parallel inference", _n_over)
    inference_model.eval()

    return InterpModelBundle(
        model=inference_model,
        datamodule=datamodule,
        device=device,
    )


def get_variable_names(bundle: InterpModelBundle) -> dict[str, dict[int, str]]:
    """Extract input/output variable names from checkpoint data_indices.

    Returns dict with keys 'input_lres', 'input_hres', 'output', each mapping
    index -> variable name.
    """
    di = bundle.data_indices
    result = {}
    if isinstance(di, dict):
        # Unified (multi-ds): one IndexCollection per dataset; use its data-space names.
        for key, ds in (("input_lres", "in_lres"), ("input_hres", "in_hres"), ("output", "out_hres")):
            n2i = getattr(di.get(ds), "name_to_index", None) if ds in di else None
            if n2i is not None:
                result[key] = {v: k for k, v in n2i.items()}
        return result
    if hasattr(di, "name_to_index_input_lres"):
        result["input_lres"] = {v: k for k, v in di.name_to_index_input_lres.items()}
    if hasattr(di, "name_to_index_input_hres"):
        result["input_hres"] = {v: k for k, v in di.name_to_index_input_hres.items()}
    if hasattr(di, "name_to_index_output"):
        result["output"] = {v: k for k, v in di.name_to_index_output.items()}
    return result


def get_surface_target_indices(bundle) -> dict:
    """Return ordered dict of surface target name -> output channel index,
    for the surface targets that exist in this checkpoint's output schema.

    `tp` is only in the o48->o96 checkpoint; o96->o320 has 10u/10v/2t/msl only.
    """
    # Unified (multi-ds) data_indices is a dict keyed by dataset; the ds model is a
    # single collection. Pick the out_hres collection either way.
    di = bundle.data_indices
    d = di["out_hres"] if isinstance(di, dict) else di
    out_idx = getattr(d, "name_to_index_output", None)
    if out_idx is None:
        out_idx = d.model.output.name_to_index
    return {name: out_idx[name] for name in SURFACE_TARGETS if name in out_idx}


def prepare_batch(
    bundle: InterpModelBundle,
    x_lres: torch.Tensor,
    x_hres: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    """Prepare a raw batch for interpretability experiments.

    Mirrors the model's _before_sampling() data flow:
    1. Ensure 5D tensors (batch, time, ensemble, grid, vars)
    2. Interpolate lres to hres grid (raw)
    3. Preprocess interpolated lres and hres for conditioning
    4. Compute residuals using raw interpolated data

    Returns dict with all tensors needed for denoiser calls.
    """
    device = bundle.device
    inner = bundle.inner_model

    x_lres = x_lres.to(device)
    x_hres = x_hres.to(device)
    y = y.to(device)

    # Ensure 5D: (batch, time, ensemble, grid, vars)
    if x_lres.dim() == 4:
        x_lres = x_lres[:, None, ...]
    if x_hres.dim() == 4:
        x_hres = x_hres[:, None, ...]
    if y.dim() == 4:
        y = y[:, None, ...]

    # Interpolate lres to hres grid (raw, unprocessed)
    x_interp_raw = inner.apply_interpolate_to_high_res(
        x_lres[:, 0, ...], None, None
    )[:, None, ...]

    # Preprocess for conditioning
    x_interp = bundle.pre_processors(x_interp_raw, dataset="input_lres", in_place=False)
    x_hres_prep = bundle.pre_processors(x_hres, dataset="input_hres", in_place=False)

    # Compute residuals using RAW interpolated data (not preprocessed)
    y_residual = inner.compute_residuals(y[:, 0, ...], x_interp_raw[:, 0, ...])
    y_residual = y_residual[:, None, ...]

    return {
        "x_interp": x_interp,
        "x_hres": x_hres_prep,
        "x_interp_raw": x_interp_raw,
        "y_residual": y_residual,
    }


def is_dict_api(inner) -> bool:
    """True for the unified (multi-ds) downscaler, whose forward/sampler use a
    per-dataset DICT API ({"in_lres","in_hres"}/{"out_hres"}); False for the legacy
    `ds` tensor-API downscaler."""
    return "diffusiondownscaler" in type(inner).__module__.lower()


def denoise_at_sigma(
    bundle: InterpModelBundle,
    x_interp: torch.Tensor,
    x_hres: torch.Tensor,
    y_residual: torch.Tensor,
    sigma: float,
    noise: torch.Tensor | None = None,
    model_comm_group=None,
    grid_shard_shapes=None,
) -> torch.Tensor:
    """Run the denoiser at a fixed sigma level (no grad).

    Returns the denoised output D(x; sigma). When ``model_comm_group`` /
    ``grid_shard_shapes`` are given, the inputs are per-rank grid SHARDS and the
    forward runs model-parallel (the output is this rank's shard).
    """
    inner = bundle.inner_model
    device = bundle.device

    if noise is None:
        noise = torch.randn_like(y_residual)

    batch_size = x_interp.shape[0]
    ensemble_size = x_interp.shape[2]
    sigma_t = torch.tensor(sigma, device=device, dtype=x_interp.dtype)
    y_noised = y_residual.to(x_interp.dtype) + sigma_t * noise.to(x_interp.dtype)
    y_noised = y_noised.to(x_interp.dtype)

    # Per-sample (batch-1) forwards. The anemoi-core-ref downscaler mis-assembles
    # the encoder input for batch_size > 1 (the batch dim leaks into the (time vars)
    # feature group, giving 151 + (B-1)*68 features), so a single batched forward
    # crashes emb_nodes_src. The model is correct at batch 1, and the tools apply
    # the denoiser independently per sample, so looping batch-1 is exact.
    dict_api = is_dict_api(inner)
    # ds tensor API: sigma (1,ens,1,1); unified dict API: sigma (1,time,ens,1,1).
    sigma_1 = sigma_t.view(1, 1, 1, 1).expand(1, ensemble_size, 1, 1)
    sigma_5 = sigma_t.view(1, 1, 1, 1, 1).expand(1, 1, ensemble_size, 1, 1)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    outs = []
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=x_interp.dtype):
            for i in range(batch_size):
                if dict_api:
                    D = inner.fwd_with_preconditioning(
                        {"in_lres": x_interp[i:i + 1], "in_hres": x_hres[i:i + 1]},
                        {"out_hres": y_noised[i:i + 1]}, {"out_hres": sigma_5},
                        model_comm_group, grid_shard_shapes)["out_hres"]
                else:
                    D = inner.fwd_with_preconditioning(
                        x_interp[i:i + 1], x_hres[i:i + 1], y_noised[i:i + 1], sigma_1,
                        model_comm_group, grid_shard_shapes)
                outs.append(D)
    return torch.cat(outs, dim=0)


def denoise_at_sigma_grad(bundle, x_interp, x_hres, y_residual, sigma, noise):
    """Grad-enabled denoise at fixed sigma (for Integrated Gradients).

    Copies denoise_at_sigma's exact calling pattern but WITHOUT torch.no_grad /
    autocast so IG can backprop into x_interp / x_hres. NOTE: uses a single
    batched forward, so it inherits the encoder batch>1 assembly bug — run IG
    with batch_size 1 (one bundle).
    """
    inner = bundle.inner_model
    device = bundle.device
    b, e = x_interp.shape[0], x_interp.shape[2]
    sigma_t = torch.tensor(sigma, device=device, dtype=x_interp.dtype)
    sigma_4d = sigma_t.view(1, 1, 1, 1).expand(b, e, 1, 1)
    y_noised = y_residual.to(x_interp.dtype) + sigma_t * noise.to(x_interp.dtype)
    return inner.fwd_with_preconditioning(x_interp, x_hres, y_noised, sigma_4d)


def sample_full(bundle, x_interp, x_hres, num_steps: int, seed: int,
                model_comm_group=None, grid_shard_shapes=None,
                sigma_min: float | None = None,
                noise_scheduler_params: dict | None = None,
                sampler_params: dict | None = None) -> torch.Tensor:
    """Run the full Heun sampling trajectory on (x_interp, x_hres).

    Uses the model's own sample() entry point with a fixed seed so baseline
    and permuted runs share the same y_init noise. When ``model_comm_group`` /
    ``grid_shard_shapes`` are given, (x_interp, x_hres) are per-rank grid SHARDS
    and the sampler runs model-parallel (returns this rank's output shard).

    ``sigma_min`` overrides the schedule floor. Some checkpoints (the unified
    o96->o320) ship inference_defaults.sigma_min=0.0, which makes the Karras
    ladder end at 0 and the Heun step divide by ~0 -> NaN; pass a small positive
    value (e.g. 0.03, the ds default) to keep the sampler finite.

    ``noise_scheduler_params`` / ``sampler_params`` override further scheduler
    fields (e.g. sigma_max, sigma_transition) and sampler knobs (e.g. S_churn)
    on top of the checkpoint's inference_defaults — pass exactly the production
    eval-config sampler to reproduce a scored run's schedule.
    """
    inner = bundle.inner_model
    nsp = dict(noise_scheduler_params or {})
    nsp["num_steps"] = num_steps
    if sigma_min is not None:
        nsp["sigma_min"] = float(sigma_min)
    with torch.no_grad():
        _shard_kw = ({"grid_shard_sizes": grid_shard_shapes} if is_dict_api(inner)
                     else {"grid_shard_shapes": grid_shard_shapes})
        out = inner.sample(
            x_interp,
            x_hres,
            model_comm_group=model_comm_group,
            noise_scheduler_params=nsp,
            sampler_params=(dict(sampler_params) if sampler_params else None),
            seed=seed,
            **_shard_kw,
        )
    return out


def per_target_mse(pred, target, target_indices: dict) -> dict:
    """Per-surface-target MSE between two tensors with shape (..., V_out).

    Returns a plain Python dict mapping surface var name -> float MSE.
    """
    out = {}
    p = pred.float()
    t = target.float()
    for name, idx in target_indices.items():
        diff = (p[..., idx] - t[..., idx]) ** 2
        out[name] = float(diff.mean().item())
    return out
