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

SURFACE_TARGETS = ["10u", "10v", "2t", "msl", "tp"]


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

    @property
    def pre_processors_tendencies(self):
        """Residual/tendency normalizers (dict keyed by dataset), unified only."""
        return getattr(self.model, "pre_processors_tendencies", None)

    @property
    def unified(self) -> bool:
        """True for multi-ds-unified checkpoints (dict data_indices keyed
        in_lres/in_hres/out_hres + dict pre/post-processors), False for the flat
        single-DS format."""
        di = self.data_indices
        return not hasattr(di, "name_to_index_input_lres")


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
    # and breaks with dynamic batch sizes in interpretability experiments.
    # (Single-DS only; the unified downscaler has no _interpolate_to_high_res.)
    if hasattr(inference_model.model, "_interpolate_to_high_res"):
        orig_func = inference_model.model._interpolate_to_high_res
        while hasattr(orig_func, "__wrapped__"):
            orig_func = orig_func.__wrapped__
        import types
        inference_model.model._interpolate_to_high_res = types.MethodType(
            orig_func, inference_model.model
        )
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
    if not bundle.unified:
        if hasattr(di, "name_to_index_input_lres"):
            result["input_lres"] = {v: k for k, v in di.name_to_index_input_lres.items()}
        if hasattr(di, "name_to_index_input_hres"):
            result["input_hres"] = {v: k for k, v in di.name_to_index_input_hres.items()}
        if hasattr(di, "name_to_index_output"):
            result["output"] = {v: k for k, v in di.name_to_index_output.items()}
    else:
        # Unified: data_indices is a dict keyed in_lres/in_hres/out_hres; names
        # live at <ds>.data.input.name_to_index and <ds>.model.output.name_to_index.
        result["input_lres"] = {v: k for k, v in
                                di["in_lres"].data.input.name_to_index.items()}
        result["input_hres"] = {v: k for k, v in
                                di["in_hres"].data.input.name_to_index.items()}
        result["output"] = {v: k for k, v in
                            di["out_hres"].model.output.name_to_index.items()}
    return result


def get_surface_target_indices(bundle) -> dict:
    """Return ordered dict of surface target name -> output channel index,
    for the surface targets that exist in this checkpoint's output schema.

    `tp` is only in the o48->o96 checkpoint; o96->o320 has 10u/10v/2t/msl only.
    """
    di = bundle.data_indices
    if bundle.unified:
        out_idx = di["out_hres"].model.output.name_to_index
    else:
        out_idx = getattr(di, "name_to_index_output", None)
        if out_idx is None:
            out_idx = di.model.output.name_to_index
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

    if bundle.unified:
        return _prepare_batch_unified(bundle, x_lres, x_hres, y)

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


def _prepare_batch_unified(bundle, x_lres, x_hres, y) -> dict:
    """Unified (multi-ds) data prep — mirrors the training task's residual flow
    (anemoi training diffusiondownscaler.py:~245): upsample in_lres via the
    model's residual interpolator, build the raw residual TARGET with
    compute_residuals (channel-matched, tendency-normalized), then normalize the
    conditioning inputs. dict-keyed pre_processors; keys in_lres/in_hres/out_hres.
    """
    inner = bundle.inner_model          # AnemoiD2ModelEncProcDec
    pp = bundle.pre_processors          # ModuleDict keyed in_lres/in_hres/out_hres
    tgt = "out_hres"

    # Upsample in_lres -> hres (residual interpolator consumes the ensemble axis;
    # re-add it). x_lres is 5D (b, time, ens=1, grid, vars).
    x_up = inner.residual["in_lres"](
        x_lres, grid_shard_sizes=None, model_comm_group=None)[:, :, None, :, :]

    # compute_residuals indexes y in the FULL out_hres DATA space (here 79
    # channels incl. interleaved forcings), but our `y` carries only the 68
    # model-output channels (what the tools index). Scatter them into a
    # data-space tensor at their data-output positions so compute_residuals'
    # data-index reads are valid; forcing slots stay zero (never read).
    out_di = bundle.data_indices[tgt]
    data_full = out_di.data.output.full.to(y.device)        # 68 data indices
    n_data = len(out_di.data.input.name_to_index)           # 79
    y_data = y.new_zeros(*y.shape[:-1], n_data)
    y_data[..., data_full] = y

    # Raw residual target (before input normalization), as in training.
    ch = inner.get_matching_channel_indices(tgt).to(x_up.device)
    ten = bundle.pre_processors_tendencies
    ten_tgt = ten[tgt] if (ten is not None and tgt in ten) else None
    y_residual = inner.compute_residuals(
        y=y_data,
        x_interp=x_up[..., ch],
        pre_processors_state=pp[tgt],
        pre_processors_tendencies=ten_tgt,
        target_dataset=tgt,
    )

    # Normalize the conditioning inputs (after residual, which needs raw data).
    x_interp = pp["in_lres"](x_up, in_place=False)
    x_hres_prep = pp["in_hres"](x_hres, in_place=False)

    return {
        "x_interp": x_interp,
        "x_hres": x_hres_prep,
        "x_interp_raw": x_up,
        "y_residual": y_residual,
    }


def _fwd_unified(inner, x_interp, x_hres, y_noised, sigma):
    """Unified fwd_with_preconditioning: inputs/outputs are dataset-keyed dicts."""
    out = inner.fwd_with_preconditioning(
        {"in_lres": x_interp, "in_hres": x_hres},
        {"out_hres": y_noised},
        {"out_hres": sigma},
    )
    return out["out_hres"]


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
    sigma_1 = sigma_t.view(1, 1, 1, 1).expand(1, ensemble_size, 1, 1)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    outs = []
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=x_interp.dtype):
            for i in range(batch_size):
                if bundle.unified:
                    sig_u = sigma_t.view(1, 1, 1, 1, 1).expand(
                        1, x_interp.shape[1], ensemble_size, 1, 1)
                    outs.append(_fwd_unified(
                        inner, x_interp[i:i + 1], x_hres[i:i + 1],
                        y_noised[i:i + 1], sig_u))
                else:
                    outs.append(inner.fwd_with_preconditioning(
                        x_interp[i:i + 1], x_hres[i:i + 1], y_noised[i:i + 1], sigma_1,
                        model_comm_group, grid_shard_shapes,
                    ))
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
    y_noised = y_residual.to(x_interp.dtype) + sigma_t * noise.to(x_interp.dtype)
    if bundle.unified:
        sig_u = sigma_t.view(1, 1, 1, 1, 1).expand(b, x_interp.shape[1], e, 1, 1)
        return _fwd_unified(inner, x_interp, x_hres, y_noised, sig_u)
    sigma_4d = sigma_t.view(1, 1, 1, 1).expand(b, e, 1, 1)
    return inner.fwd_with_preconditioning(x_interp, x_hres, y_noised, sigma_4d)


def sample_full(bundle, x_interp, x_hres, num_steps: int, seed: int,
                model_comm_group=None, grid_shard_shapes=None) -> torch.Tensor:
    """Run the full Heun sampling trajectory on (x_interp, x_hres).

    Uses the model's own sample() entry point with a fixed seed so baseline
    and permuted runs share the same y_init noise. When ``model_comm_group`` /
    ``grid_shard_shapes`` are given, (x_interp, x_hres) are per-rank grid SHARDS
    and the sampler runs model-parallel (returns this rank's output shard).
    """
    inner = bundle.inner_model
    with torch.no_grad():
        if bundle.unified:
            # Unified sample(): kwarg is grid_shard_sizes (not _shapes), no seed.
            out = inner.sample(
                x_interp, x_hres,
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_shapes,
                noise_scheduler_params={"num_steps": num_steps},
                sampler_params=None,
            )
        else:
            out = inner.sample(
                x_interp, x_hres,
                model_comm_group=model_comm_group,
                grid_shard_shapes=grid_shard_shapes,
                noise_scheduler_params={"num_steps": num_steps},
                sampler_params=None,
                seed=seed,
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
