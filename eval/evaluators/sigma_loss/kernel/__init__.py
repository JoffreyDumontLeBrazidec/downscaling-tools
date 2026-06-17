"""sigma_loss kernel — fixed-sigma sweep + F-space per-variable loss readout.

Domain logic only. No I/O, no plotting, no eval.cli coupling.

Method (mirrors the validated Jupiter probe ``probe_a_per_sigma.py`` and the
unified/legacy ``GraphDiffusionDownscaler._step``):

  * ``install_fixed_sigma`` monkeypatches ``_get_noise_level`` so the diffusion
    task uses a single deterministic sigma instead of lognormal sampling. The
    returned loss weight is ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``
    == ``1/c_out^2`` (the EDM preconditioning output scale), exactly as the task
    computes it.
  * For each (sigma x fixed batch) we run the task ``_step`` (compute_residuals
    -> add FIXED-SEED noise -> fwd_with_preconditioning -> WeightedMSELoss with
    weights=1/c_out^2). That total is the F-space (network-output) loss.
  * Per-variable loss is read out by re-running the *same* WeightedMSELoss on the
    captured ``(pred, target, weights)`` with ``squash=False`` (keeps the
    variable dim). Because the task squashes with ``squash_mode=avg``, the mean
    of the per-variable vector equals the task total — this is the M0 sanity
    contract.

Two model APIs are supported transparently:
  * ``tensor`` — legacy single-ds DS branch: batch is ``[x_in, x_in_hres, y]``,
    ``_step(batch, batch_idx, ...)``; loss is a single ``WeightedMSELoss``.
  * ``dict``   — unified multi-ds branch: batch is ``{in_lres, in_hres, out_hres}``,
    ``_step(batch, validation_mode=...)``; loss is a ``ModuleDict`` keyed by the
    target dataset name.
"""

from .domain import (
    install_fixed_sigma,
    fspace_loss_for_batch,
    capture_fixed_batches,
    detect_api,
    output_variable_names,
    log_spaced_grid,
)
from .sweep import run_sigma_sweep

__all__ = [
    "install_fixed_sigma",
    "fspace_loss_for_batch",
    "capture_fixed_batches",
    "detect_api",
    "output_variable_names",
    "log_spaced_grid",
    "run_sigma_sweep",
]
