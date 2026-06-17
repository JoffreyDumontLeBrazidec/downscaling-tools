"""sigma_loss kernel — model + datamodule loader.

Reuses the canonical manual_inference loader entry point
``manual_inference.checkpoints.ObjectFromCheckpointLoader`` — the exact same
class the existing ``sigma`` evaluator backend uses to obtain a loaded model +
datamodule from a checkpoint for a lane. We add only the thin glue M0 needs:
single-rank shard-attr neutralisation, validation-frequency / worker capping,
device placement, and Leonardo->local dataset path rewriting (via
``manual_inference.prediction.predict._rewrite_dataset_paths_in_place``).

No new checkpoint-loading logic is duplicated here.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from manual_inference.checkpoints import ObjectFromCheckpointLoader
from manual_inference.prediction.predict import _rewrite_dataset_paths_in_place

from .domain import detect_api, model_sigma_data, output_variable_names

LOG = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    downscaler: Any
    datamodule: Any
    device: str
    api: str
    sigma_data: float
    variables: list[str]
    run_id: str
    ckpt_step: int | None
    checkpoint_path: str


def _split_ckpt(path: str) -> tuple[str, str, str]:
    p = Path(os.path.abspath(os.path.expanduser(path)))
    return str(p.parent.parent), p.parent.name, p.name


def _ckpt_step_from_name(name: str) -> int | None:
    import re
    m = re.search(r"step[_-]?(\d+)", name)
    return int(m.group(1)) if m else None


def _neutralise_single_rank(downscaler: Any) -> None:
    """Make a model loaded for multi-GPU run on a single rank.

    Sets shard shapes / comm group / shard slices to None so the forward path
    takes the unsharded branch. Covers both API shapes.
    """
    for attr in (
        "lres_grid_shard_shapes",
        "hres_grid_shard_shapes",
        "model_comm_group",
        "grid_shard_shapes",
    ):
        setattr(downscaler, attr, None)
    # unified task: dict-valued shard sizes/slices keyed by dataset name
    for attr in ("grid_shard_sizes", "grid_shard_slice"):
        cur = getattr(downscaler, attr, None)
        if isinstance(cur, dict):
            setattr(downscaler, attr, {k: None for k in cur})
    if hasattr(downscaler, "model_comm_group_size"):
        downscaler.model_comm_group_size = 1


def load_model_for_lane(
    checkpoint: str,
    *,
    validation_frequency: str | None = None,
    device: str | None = None,
) -> LoadedModel:
    """Load downscaler + datamodule for a checkpoint via the manual_inference loader.

    Parameters
    ----------
    checkpoint : str
        Path to the base (non-inference) checkpoint .ckpt.
    validation_frequency : str | None
        Optional override on the validation dataloader frequency. Default None =
        use the checkpoint-native validation config (the fake_hindcasts subset
        does not support arbitrary frequency strides, so overriding can empty
        the split). The native val order is deterministic for a fixed base seed.
    device : str | None
        "cuda" / "cpu". Defaults to cuda when available.
    """
    # ANEMOI_BASE_SEED is required by the anemoi dataloader worker_init; make the
    # batch capture deterministic and worker-safe.
    os.environ.setdefault("ANEMOI_BASE_SEED", "1234")

    dir_exp, name_exp, name_ckpt = _split_ckpt(checkpoint)
    loader = ObjectFromCheckpointLoader(dir_exp, name_exp, name_ckpt)

    # Rewrite remote (Leonardo / Jupiter) dataset roots to local mirrors.
    loader.config_for_datamodule = _rewrite_dataset_paths_in_place(loader.config_for_datamodule)

    # Optional fixed validation slice (default: native config).
    if validation_frequency:
        try:
            loader.config_for_datamodule.dataloader.validation.frequency = validation_frequency
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not set validation.frequency: %s", exc)

    # Cap dataloader workers at the dataloader level (NOT the dataset spec, which
    # would be mis-parsed). Use 1 not 0: the datamodule sets a fixed prefetch_factor
    # that requires num_workers>0; ANEMOI_BASE_SEED makes worker init deterministic.
    # (original note)
    # would be mis-parsed as a dataset open() argument).
    dl_cfg = getattr(loader.config_for_datamodule, "dataloader", None)
    nw = getattr(dl_cfg, "num_workers", None) if dl_cfg is not None else None
    if nw is not None and hasattr(nw, "validation"):
        nw.validation = 1
    elif dl_cfg is not None and isinstance(getattr(dl_cfg, "num_workers", None), int):
        dl_cfg.num_workers = 1

    # Force single-rank: the checkpoint was trained with model-parallel grid
    # sharding (read_group_size / num_gpus_per_model > 1), which makes the
    # datamodule pre-shard each batch grid by that factor. On one rank that
    # yields a fractional grid that the (unsharded) interpolation matrix
    # rejects. Pin both to 1 so the batch carries the full grid. Mirrors
    # eval._backends.sigma_evaluator.run_sigma_evaluator.
    for cfg in (loader.config_checkpoint, loader.config_for_datamodule):
        hw = getattr(cfg, "hardware", None)
        if hw is not None and hasattr(hw, "num_gpus_per_model"):
            hw.num_gpus_per_model = 1
        dl = getattr(cfg, "dataloader", None)
        if dl is not None and hasattr(dl, "read_group_size"):
            dl.read_group_size = 1

    loader.load()

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    downscaler = loader.downscaler.to(resolved_device)
    downscaler.eval()
    _neutralise_single_rank(downscaler)

    api = detect_api(downscaler)
    sigma_data = model_sigma_data(downscaler)
    variables = output_variable_names(downscaler, api)

    return LoadedModel(
        downscaler=downscaler,
        datamodule=loader.datamodule,
        device=resolved_device,
        api=api,
        sigma_data=sigma_data,
        variables=variables,
        run_id=name_exp,
        ckpt_step=_ckpt_step_from_name(name_ckpt),
        checkpoint_path=os.path.abspath(os.path.expanduser(checkpoint)),
    )
