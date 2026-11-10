"""Inputs from a split of the checkpoint's own datamodule (``predict.input_source`` type zarr_split).

The E2 Europe box checkpoints were never given eval-format bundles: every E2 read took its
inputs from the E2 zarr TEST split through the model's own datamodule (Jupiter kit
predictor ``20261102-2t-students-30k-sl/kit/predict_student_sl.py``). This module does the
same data access inside ``eval.predict``, so the rest of the bundle path (sampler routing,
rule-31 seeds, EnsemblePrediction, write_predictions_file, validation) is shared:

* (init date YYYYMMDD, lead h) is mapped to a row of the split through the fake-hindcast
  mapping of the out_hres reader (``ZarrSplitSource.find_row``);
* the row is read once with ``MultiDataset.get_sample`` (in_lres, in_hres, out_hres) and
  every member draws from the same row (members are independent draws, rule 31);
* x is the low-res input on the low-res grid for the saved weather states (NaN where the
  in_lres reader has no such variable), y the truth from out_hres.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Sequence

import numpy as np
import torch

from ._mi import USING_LEGACY_DS, dataset as _mi_dataset, predict as _mi_predict
from .types import PredictionResult

LOG = logging.getLogger(__name__)

_predict_with_compatible_kwargs = _mi_predict._predict_with_compatible_kwargs
_model_takes_lead_hours = _mi_predict._model_takes_lead_hours
resolve_output_weather_states = _mi_dataset.resolve_output_weather_states

# split name of the lane key -> datamodule property (and the dataloader config block of that name)
_SPLIT_DATASETS = {"training": "ds_train", "validation": "ds_valid", "test": "ds_test"}
_LEAD_DATASET = "out_hres"


def _fake_hindcast_owner(reader):
    """The object under ``reader.data`` that owns the fake-hindcast mapping (walk the forward chain)."""
    ds = reader.data
    for _ in range(32):
        try:
            ds.fake_hindcasts
            return ds
        except AttributeError:
            ds = ds.forward
    raise ValueError(f"no fake_hindcasts mapping under dataset {reader.data!r}")


def _store_paths(datamodule, split: str) -> dict:
    """The dataset entry of each reader of the split, as the checkpoint's config names it."""
    try:
        from omegaconf import OmegaConf

        datasets = OmegaConf.to_container(datamodule.config.dataloader[split].datasets, resolve=True)
        return {name: cfg.get("dataset") for name, cfg in datasets.items() if isinstance(cfg, dict)}
    except Exception as exc:  # provenance only; never block a prediction on it
        LOG.warning("zarr_split: could not read the store paths of split %s (%s)", split, exc)
        return {}


def _squeeze_to_grid_vars(array, name: str) -> np.ndarray:
    """(time, ensemble, grid, vars) of one sample -> (grid, vars)."""
    arr = np.asarray(array.detach().cpu() if torch.is_tensor(array) else array)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"zarr_split: {name} sample has shape {np.asarray(array).shape}, expected one time and member")
    return arr.astype(np.float32, copy=False)


class ZarrSplitSource:
    """Rows of one datamodule split, addressed by (init date, lead)."""

    def __init__(self, datamodule, split: str = "test") -> None:
        if USING_LEGACY_DS:
            raise SystemExit("input_source zarr_split needs the unified multi-ds datamodule, not KEYSTONE_LEGACY_DS")
        if split not in _SPLIT_DATASETS:
            raise SystemExit(f"input_source zarr_split: unknown split {split!r}, expected one of {list(_SPLIT_DATASETS)}")
        self.split = split
        self.dataset = getattr(datamodule, _SPLIT_DATASETS[split])
        # One reader, the full grid of every dataset (what DDPGroupStrategy sets in training);
        # per_worker_init reseeds torch with the training base seed, so it runs here, once,
        # before any rule-31 draw seed.
        shard_sizes = {name: [int(reader.data.shape[-1])] for name, reader in self.dataset.datasets.items()}
        self.dataset.set_comm_group_info(0, 0, 0, 1, 0, 1, shard_sizes)
        self.dataset.per_worker_init(1, 0)
        self.stores = _store_paths(datamodule, split)
        self._check_name_to_index(datamodule)
        self._rows: dict[int, dict[str, tuple[int, str]]] = {}
        self._cached_key: tuple[str, int] | None = None
        self._cached_row: dict[str, np.ndarray] | None = None

    def _check_name_to_index(self, datamodule) -> None:
        # predict_step reads its inputs at the data indices of the datamodule's data_indices; the
        # split's readers must hold every such variable at the same index.
        split_n2i = self.dataset.name_to_index
        for name in ("in_lres", "in_hres"):
            expected = dict(datamodule.data_indices[name].data.input.name_to_index)
            if any(split_n2i[name].get(var) != idx for var, idx in expected.items()):
                raise SystemExit(
                    f"input_source zarr_split: the {self.split} split's {name} variables {dict(split_n2i[name])} "
                    f"differ from the model's data indices {expected}"
                )

    def rows_at_lead(self, lead: int) -> dict[str, tuple[int, str]]:
        """{init date YYYYMMDD: (row index, fake date)} of every row of the split at this lead."""
        lead = int(lead)
        if lead not in self._rows:
            reader = self.dataset.datasets[_LEAD_DATASET]
            owner = _fake_hindcast_owner(reader)
            rows: dict[str, tuple[int, str]] = {}
            for index, fake in enumerate(np.asarray(reader.dates)):
                _, init, step = owner.fake_date_to_date_step(fake)
                if int(step) != lead:
                    continue
                date = str(np.datetime64(init, "D")).replace("-", "")
                if date in rows:
                    raise SystemExit(f"input_source zarr_split: two {self.split} rows for {date} +{lead} h")
                rows[date] = (index, str(np.datetime64(fake, "s")))
            self._rows[lead] = rows
        return self._rows[lead]

    def find_row(self, date: str, lead: int) -> tuple[int, str]:
        rows = self.rows_at_lead(lead)
        if str(date) not in rows:
            raise SystemExit(
                f"input_source zarr_split: no {self.split} row for {date} +{int(lead)} h "
                f"({len(rows)} rows at this lead, e.g. {sorted(rows)[:3]})"
            )
        return rows[str(date)]

    def sample(self, date: str, lead: int) -> dict[str, np.ndarray]:
        """(grid, vars) arrays of in_lres, in_hres and out_hres of the row; read once per case."""
        key = (str(date), int(lead))
        if self._cached_key != key:
            index, _ = self.find_row(date, lead)
            raw = self.dataset.get_sample(index)
            if "lead_hours" in raw and float(raw["lead_hours"]) != float(lead):
                raise SystemExit(f"input_source zarr_split: row {index} has lead {float(raw['lead_hours'])}, not {lead}")
            self._cached_row = {name: _squeeze_to_grid_vars(raw[name], name) for name in ("in_lres", "in_hres", "out_hres")}
            self._cached_key = key
        return self._cached_row

    def coordinates(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        data = self.dataset.datasets[name].data
        return np.asarray(data.longitudes), np.asarray(data.latitudes)

    def provenance(self, date: str, lead: int) -> dict:
        """File attributes in place of the bundle paths."""
        index, fake = self.find_row(date, lead)
        return {
            "input_source": "zarr_split",
            "input_split": self.split,
            "input_row_index": int(index),
            "input_fake_date": fake,
            "input_stores": json.dumps(self.stores, sort_keys=True, default=str),
        }

    def label(self, date: str, lead: int) -> str:
        """One line for the manifest's bundle_path column."""
        index, fake = self.find_row(date, lead)
        return f"zarr_split:{self.split}:row={index}:fake_date={fake}"


def predict_single_zarr_row(
    inference_model,
    datamodule,
    device: str,
    source: ZarrSplitSource,
    date: str,
    step: int,
    *,
    extra_args: dict,
    precision: str,
    model_comm_group,
    output_weather_state_mode: str = "surface-plus-core-pl",
    output_weather_states: Sequence[str] | None = None,
) -> PredictionResult:
    """One draw from the split row of (date, step); the zarr_split twin of ``predict_from_bundle``."""

    name_to_idx_lres = datamodule.data_indices["in_lres"].data.input.name_to_index
    name_to_idx_out = datamodule.data_indices["out_hres"].model.output.name_to_index
    name_to_idx_truth = source.dataset.name_to_index["out_hres"]

    row = source.sample(date, step)
    _mi_predict.check_input_distribution(
        inference_model, row["in_lres"], name_to_idx_lres, label=source.label(date, step)
    )
    x_in = torch.from_numpy(row["in_lres"]).to(device)[None, None, None, ...]
    x_in_hres = torch.from_numpy(row["in_hres"]).to(device)[None, None, None, ...]
    if str(device).startswith("cuda") and precision == "bf16" and os.environ.get("MI_BF16_INPUTS", "0") == "1":
        x_in = x_in.bfloat16()
        x_in_hres = x_in_hres.bfloat16()

    amp_enabled = str(device).startswith("cuda") and precision in {"fp16", "bf16"}
    if os.environ.get("MI_BF16_NATIVE", "0") == "1" and precision == "bf16":
        amp_enabled = False
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    batch = {"in_lres": x_in, "in_hres": x_in_hres}
    if _model_takes_lead_hours(inference_model):
        extra_args = {**extra_args, "lead_hours": float(step)}
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            pred = _predict_with_compatible_kwargs(
                inference_model=inference_model,
                batch=batch,
                model_comm_group=model_comm_group,
                extra_args=extra_args,
            )
    pred_tensor = pred["out_hres"] if isinstance(pred, dict) else pred

    weather_states, selected_indices = resolve_output_weather_states(
        weather_states=list(name_to_idx_out.keys()),
        mode=output_weather_state_mode,
        explicit_weather_states=output_weather_states,
    )
    pred_np = pred_tensor[0, 0, 0][..., selected_indices].detach().cpu().numpy().astype(np.float32)
    del pred, pred_tensor
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    lon_lres, lat_lres = source.coordinates("in_lres")
    lon_hres, lat_hres = source.coordinates("out_hres")
    x_np = np.full((row["in_lres"].shape[0], len(weather_states)), np.nan, dtype=np.float32)
    y_np = np.full((row["out_hres"].shape[0], len(weather_states)), np.nan, dtype=np.float32)
    for j, name in enumerate(weather_states):
        if name in name_to_idx_lres:
            x_np[:, j] = row["in_lres"][:, name_to_idx_lres[name]]
        if name in name_to_idx_truth:
            y_np[:, j] = row["out_hres"][:, name_to_idx_truth[name]]

    local_graph_mask = getattr(inference_model, "_local_graph_cut_data_mask", None)
    if local_graph_mask is not None:
        local_graph_mask = local_graph_mask.detach().cpu().numpy().astype(bool)
        if pred_np.shape[-2] != int(local_graph_mask.sum()):
            raise ValueError(
                "Local cut-graph prediction size mismatch: "
                f"model returned {pred_np.shape[-2]} hres nodes but mask selects {int(local_graph_mask.sum())}."
            )
        lon_hres, lat_hres, y_np = lon_hres[local_graph_mask], lat_hres[local_graph_mask], y_np[local_graph_mask]

    return PredictionResult(
        x=x_np[None, ...],
        y=y_np[None, None, ...],
        y_pred=pred_np[None, None, ...],
        lon_lres=lon_lres,
        lat_lres=lat_lres,
        lon_hres=lon_hres,
        lat_hres=lat_hres,
        weather_states=[str(name) for name in weather_states],
        bundle_path=None,
        dates=None,
    )
