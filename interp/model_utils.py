"""Model loading and data utilities for interpretability experiments.

Thin wrapper around manual_inference model loading. Provides a clean
InterpModelBundle for all interpretability tools.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_DT_ROOT = Path(__file__).resolve().parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

import os

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

    Returns
    -------
    InterpModelBundle
    """
    inference_model, datamodule, _, _ = _load_objects(
        ckpt_path=str(checkpoint_path),
        device=device,
        validation_frequency=validation_frequency,
        precision=precision,
        num_gpus_per_model_override=1,
    )
    # Disable torch.compile on the interpolation function — it caches shapes
    # and breaks with dynamic batch sizes in interpretability experiments.
    # Get the original uncompiled function
    orig_func = inference_model.model._interpolate_to_high_res
    while hasattr(orig_func, "__wrapped__"):
        orig_func = orig_func.__wrapped__
    # Rebind as an instance method
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
    if hasattr(di, "name_to_index_input_lres"):
        result["input_lres"] = {v: k for k, v in di.name_to_index_input_lres.items()}
    if hasattr(di, "name_to_index_input_hres"):
        result["input_hres"] = {v: k for k, v in di.name_to_index_input_hres.items()}
    if hasattr(di, "name_to_index_output"):
        result["output"] = {v: k for k, v in di.name_to_index_output.items()}
    return result


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


def denoise_at_sigma(
    bundle: InterpModelBundle,
    x_interp: torch.Tensor,
    x_hres: torch.Tensor,
    y_residual: torch.Tensor,
    sigma: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the denoiser at a fixed sigma level.

    Returns the denoised output D(x; sigma).
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
    # crashes emb_nodes_src. The model is correct at batch 1, and the Tier-1 tools
    # apply the denoiser independently per sample, so looping batch-1 is exact.
    sigma_1 = sigma_t.view(1, 1, 1, 1).expand(1, ensemble_size, 1, 1)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    outs = []
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=x_interp.dtype):
            for i in range(batch_size):
                outs.append(inner.fwd_with_preconditioning(
                    x_interp[i:i + 1], x_hres[i:i + 1], y_noised[i:i + 1], sigma_1
                ))
    return torch.cat(outs, dim=0)


# === surface target helpers ===
SURFACE_TARGETS = ["10u", "10v", "2t", "msl", "tp"]


def get_surface_target_indices(bundle) -> dict:
    """Return ordered dict of surface target name -> output channel index,
    for the surface targets that exist in this checkpoint's output schema.

    `tp` is only in the o48->o96 checkpoint; o96->o320 has 10u/10v/2t/msl only.
    """
    di = bundle.data_indices
    out_idx = getattr(di, "name_to_index_output", None)
    if out_idx is None:
        # Fallback for other index collection layouts
        out_idx = di.model.output.name_to_index
    return {name: out_idx[name] for name in SURFACE_TARGETS if name in out_idx}


def per_target_mse(pred, target, target_indices: dict) -> dict:
    """Per-surface-target MSE between two tensors with shape (..., V_out).

    Both tensors must already be 4D-or-5D with V as the last axis. Returns a
    plain Python dict mapping surface var name -> float MSE.
    """
    import torch as _torch
    out = {}
    p = pred.float()
    t = target.float()
    for name, idx in target_indices.items():
        diff = (p[..., idx] - t[..., idx]) ** 2
        out[name] = float(diff.mean().item())
    return out


def collect_event_bundles(bundle, bundle_dir, dates, members, steps):
    """Build (x_lres, x_hres, y) from real event .nc bundles.

    This is the same input path the IG/patching tools use. It replaces
    ``datamodule.val_dataloader()`` for the Tier-1 tools: after the 2026-06-11
    env reorg the o96->o320 val dataloader yields lres on the O320 (hres) grid,
    which makes the assembled encoder input 219-wide vs the trained 151 and
    crashes ``emb_nodes_src``. Real bundles carry O96 lres, so prepare_batch
    interpolates O96->O320 correctly. Each (date, member, step) becomes one
    batch element, so pass several to get batch_size > 1.
    """
    import glob as _glob
    from manual_inference.input_data_construction.bundle import (
        load_inputs_from_bundle_numpy as _load_bundle_np,
        extract_target_from_bundle as _extract_target,
    )

    vn = get_variable_names(bundle)
    n2i_lres = {name: idx for idx, name in vn["input_lres"].items()}
    n2i_hres = {name: idx for idx, name in vn["input_hres"].items()}
    out_states = [vn["output"][i] for i in sorted(vn["output"])]

    paths = []
    for d in dates:
        for m in members:
            for s in steps:
                pat = f"*date{d}*mem{m}*step{s}h*input_bundle.nc"
                hits = sorted(_glob.glob(str(Path(bundle_dir) / pat)))
                if not hits:
                    raise FileNotFoundError(f"No bundle matching {pat} in {bundle_dir}")
                paths.append(hits[0])

    xl, xh, ys = [], [], []
    for p in paths:
        x_lres_np, x_hres_np, *_ = _load_bundle_np(p, n2i_lres, n2i_hres)
        target_np, _found = _extract_target(p, out_states)
        if target_np is None:
            raise SystemExit(f"Could not extract target from bundle {p}")
        xl.append(torch.from_numpy(x_lres_np)[None, None, None, ...])
        xh.append(torch.from_numpy(x_hres_np)[None, None, None, ...])
        ys.append(torch.from_numpy(target_np)[None, None, None, ...])
    x_lres, x_hres, y = torch.cat(xl), torch.cat(xh), torch.cat(ys)
    LOGGER.info("collect_event_bundles: %d bundle(s) from %s x_lres=%s x_hres=%s y=%s",
                len(paths), bundle_dir, tuple(x_lres.shape), tuple(x_hres.shape), tuple(y.shape))
    return x_lres, x_hres, y
