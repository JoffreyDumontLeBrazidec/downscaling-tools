"""sigma_loss kernel — the sigma sweep.

K fixed validation batches x sigma grid -> per-sigma per-variable F-space loss,
each from a SINGLE forward pass (no diffusion sampling). Fixed noise seed +
fixed batches make the profile comparable across sigma and (later) checkpoints.
"""
from __future__ import annotations

import gc
import logging
from typing import Any

import torch

from .domain import (
    capture_fixed_batches,
    fspace_loss_for_batch,
    install_fixed_sigma,
)

LOG = logging.getLogger(__name__)


def run_sigma_sweep(
    loaded: Any,
    sigma_grid: list[float],
    *,
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the fixed-sigma sweep.

    Returns a list of row dicts, one per (sigma, variable) plus one per
    (sigma, "__total__"):
        {sigma, variable, fspace_loss, n_batches}

    The fixed batches are captured once and reused for every sigma.
    """
    downscaler = loaded.downscaler
    api = loaded.api
    variables = loaded.variables

    batches = capture_fixed_batches(loaded.datamodule, k, loaded.device, api)
    LOG.info("Captured %d fixed validation batches (api=%s)", len(batches), api)

    rows: list[dict[str, Any]] = []
    for sigma in sigma_grid:
        restore = install_fixed_sigma(downscaler, sigma, api)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            total_acc = 0.0
            per_var_acc: torch.Tensor | None = None
            n = 0
            for b_idx, batch in enumerate(batches):
                # Same seed for every (sigma, batch_idx) pair -> identical noise
                # draw across the whole sweep for a given batch.
                batch_seed = seed + b_idx
                total, per_var = fspace_loss_for_batch(
                    downscaler, batch, api, seed=batch_seed
                )
                total_acc += total
                per_var_acc = per_var if per_var_acc is None else per_var_acc + per_var
                n += 1
            mean_total = total_acc / n
            mean_per_var = (per_var_acc / n) if per_var_acc is not None else None

            rows.append(
                {"sigma": float(sigma), "variable": "__total__",
                 "fspace_loss": float(mean_total), "n_batches": n}
            )
            if mean_per_var is not None:
                nv = mean_per_var.numel()
                for i in range(nv):
                    var_name = variables[i] if i < len(variables) else f"var_{i}"
                    rows.append(
                        {"sigma": float(sigma), "variable": var_name,
                         "fspace_loss": float(mean_per_var[i]), "n_batches": n}
                    )
            peak = (
                float(torch.cuda.max_memory_allocated() / 1e9)
                if torch.cuda.is_available() else 0.0
            )
            LOG.info("sigma=%.4g total_fspace_loss=%.6g (n=%d, peak=%.2fGB)",
                     sigma, mean_total, n, peak)
        finally:
            restore()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return rows
