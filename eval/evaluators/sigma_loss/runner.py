"""sigma_loss evaluator — runner (orchestration).

Resolves a checkpoint, loads model+datamodule via the manual_inference loader
(``eval.evaluators.sigma_loss.kernel.loader.load_model_for_lane``, which wraps
``manual_inference.checkpoints.ObjectFromCheckpointLoader``), captures K fixed
validation batches, sweeps the sigma grid with SINGLE forward passes, and writes:

  data/sigma_loss/per_sigma.csv   (run_id, ckpt_step, sigma, variable,
                                   fspace_loss, n_batches, seed)
  data/sigma_loss/meta.json       (sigma_grid, k, seed, checkpoint, lane,
                                   variables, sigma_data, api)

Writes ONLY under <results_dir>/data/sigma_loss/. Imports only from
eval.* / manual_inference / its own kernel / stdlib.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from .kernel import log_spaced_grid, run_sigma_sweep
from .kernel.loader import load_model_for_lane

LOG = logging.getLogger(__name__)

DATA_SUBDIR = ("data", "sigma_loss")


def _data_dir(output_dir: Path) -> Path:
    d = output_dir.joinpath(*DATA_SUBDIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_sigma_grid(eval_config: dict) -> list[float]:
    grid = eval_config.get("sigma_grid")
    if isinstance(grid, list) and grid:
        return [float(s) for s in grid]
    # log-spaced fallback through the lane extreme band
    lo = float(eval_config.get("sigma_min", 0.02))
    hi = float(eval_config.get("sigma_max", 500.0))
    n = int(eval_config.get("sigma_n", 16))
    return log_spaced_grid(lo, hi, n)


def run(
    predictions_dir: str | Path | None,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    checkpoint: str | None = None,
    **kwargs: Any,
) -> Path:
    """Run the per-sigma F-space loss sweep for a single checkpoint (M0).

    ``predictions_dir`` is unused (this evaluator runs the model, like sigma /
    mechanistic). ``checkpoint`` is required.

    M1 STUB: ``eval_config["checkpoints"]`` (a selector of multiple checkpoints)
    is accepted but only the single ``checkpoint`` arg is processed in M0. See
    the multi-checkpoint TODO below.
    """
    if not checkpoint:
        raise ValueError(
            "sigma_loss requires --checkpoint (it runs the model, not predictions). "
            "Pass it via `python -m eval.cli evaluate --checkpoint <path> --only sigma_loss`."
        )

    # ---- M1 STUB: multi-checkpoint sweep -----------------------------------
    # TODO(M1): if eval_config.get("checkpoints") is set, resolve each via
    # eval.discovery and loop run_sigma_sweep over them, tagging rows by
    # run_id/ckpt_step so the scorer/plotter can produce per-checkpoint curves.
    if eval_config.get("checkpoints"):
        LOG.warning(
            "sigma_loss: eval_config['checkpoints'] is set but multi-checkpoint "
            "sweep is M1 (stub). Using single --checkpoint=%s for M0.", checkpoint,
        )

    output_dir = Path(output_dir) if output_dir else Path(checkpoint).parent / "evaluators" / "sigma_loss"
    data_dir = _data_dir(output_dir)
    csv_path = data_dir / "per_sigma.csv"
    meta_path = data_dir / "meta.json"
    if csv_path.exists() and not overwrite:
        LOG.info("sigma_loss per_sigma.csv already exists, skipping: %s", csv_path)
        return output_dir

    seed = int(eval_config.get("seed", 1234))
    k = int(eval_config.get("k", 8))
    validation_frequency = eval_config.get("validation_frequency")  # None = native
    sigma_grid = _resolve_sigma_grid(eval_config)

    LOG.info(
        "sigma_loss: checkpoint=%s k=%d seed=%d sigma_grid(n=%d)=%s",
        checkpoint, k, seed, len(sigma_grid), sigma_grid,
    )

    loaded = load_model_for_lane(checkpoint, validation_frequency=validation_frequency)
    LOG.info(
        "sigma_loss: loaded run_id=%s step=%s api=%s sigma_data=%g vars=%d",
        loaded.run_id, loaded.ckpt_step, loaded.api, loaded.sigma_data, len(loaded.variables),
    )

    rows = run_sigma_sweep(loaded, sigma_grid, k=k, seed=seed)

    # ---- write per_sigma.csv ----
    fieldnames = ["run_id", "ckpt_step", "sigma", "variable", "fspace_loss", "n_batches", "seed"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "run_id": loaded.run_id,
                "ckpt_step": loaded.ckpt_step if loaded.ckpt_step is not None else "",
                "sigma": r["sigma"],
                "variable": r["variable"],
                "fspace_loss": r["fspace_loss"],
                "n_batches": r["n_batches"],
                "seed": seed,
            })
    LOG.info("sigma_loss: wrote %d rows -> %s", len(rows), csv_path)

    # ---- write meta.json ----
    meta = {
        "lane": lane_config.get("__lane_name__") or eval_config.get("lane") or "",
        "checkpoint": loaded.checkpoint_path,
        "run_id": loaded.run_id,
        "ckpt_step": loaded.ckpt_step,
        "api": loaded.api,
        "sigma_data": loaded.sigma_data,
        "sigma_grid": sigma_grid,
        "k": k,
        "seed": seed,
        "validation_frequency": validation_frequency,
        "variables": loaded.variables,
        # ---- STUB flags (not implemented in M0) ----
        "data_space_loss": False,  # TODO(M0+): physical-unit (denorm) loss option
        "multi_checkpoint": False,  # TODO(M1)
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    LOG.info("sigma_loss: wrote meta -> %s", meta_path)

    return output_dir
