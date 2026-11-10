"""Inference-loop wrappers for ensemble prediction from bundles or from a zarr split."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ._mi import predict as _mi_predict
_compute_x_interp_for_export = _mi_predict._compute_x_interp_for_export
_predict_from_bundle = _mi_predict._predict_from_bundle

from .bundle_manager import date_str_to_datetime64
from .seeding import resolve_base_seed, seed_draw
from .types import BundleKey, EnsemblePrediction, PredictionResult
from .zarr_split import predict_single_zarr_row


def _squeeze_leading_singletons(array: np.ndarray, target_ndim: int) -> np.ndarray:
    squeezed = np.asarray(array)
    while squeezed.ndim > target_ndim and squeezed.shape[0] == 1:
        squeezed = squeezed[0]
    if squeezed.ndim != target_ndim:
        raise ValueError(
            f"Unsupported array shape {tuple(np.asarray(array).shape)}; expected {target_ndim} dimensions "
            f"after removing leading singleton axes."
        )
    return squeezed.astype(np.float32, copy=False)


def _cleanup_member_outputs(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_single_bundle(
    inference_model,
    datamodule,
    device: str,
    bundle_path: Path,
    *,
    extra_args: dict,
    precision: str,
    model_comm_group,
    output_weather_state_mode: str = "surface-plus-core-pl",
    output_weather_states: Sequence[str] | None = None,
) -> PredictionResult:
    """Run prediction for a single input bundle and return structured outputs."""

    x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, dates = _predict_from_bundle(
        inference_model=inference_model,
        datamodule=datamodule,
        device=device,
        bundle_nc=str(bundle_path),
        member_index=0,
        extra_args=extra_args,
        precision=precision,
        model_comm_group=model_comm_group,
        output_weather_state_mode=output_weather_state_mode,
        output_weather_states=output_weather_states,
    )
    return PredictionResult(
        x=np.asarray(x),
        y=None if y is None else np.asarray(y),
        y_pred=np.asarray(y_pred),
        lon_lres=np.asarray(lon_lres),
        lat_lres=np.asarray(lat_lres),
        lon_hres=np.asarray(lon_hres),
        lat_hres=np.asarray(lat_hres),
        weather_states=[str(name) for name in weather_states],
        bundle_path=bundle_path,
        dates=None if dates is None else np.asarray(dates),
    )


def predict_ensemble_members(
    inference_model,
    datamodule,
    device: str,
    bundle_map: dict[BundleKey, Path] | None,
    date: str,
    step: int,
    members: Sequence[int],
    *,
    extra_args: dict,
    precision: str,
    model_comm_group,
    allow_missing_target_unsafe: bool = False,
    output_weather_state_mode: str = "surface-plus-core-pl",
    output_weather_states: Sequence[str] | None = None,
    keep_outputs: bool = True,
    zarr_source=None,
) -> EnsemblePrediction:
    """Predict all requested members for one date/step pair.

    Inputs come from ``bundle_map`` (one bundle per member), or, when ``zarr_source`` (a
    ``zarr_split.ZarrSplitSource``) is given, from the split row of (date, step), the same
    row for every member.
    """

    x_members: list[np.ndarray] = []
    y_members: list[np.ndarray | None] = []
    y_pred_members: list[np.ndarray] = []
    source_bundle_paths: list[str] = []
    members_missing_target: list[int] = []
    draw_seeds: list[int] = []
    lon_lres = lat_lres = lon_hres = lat_hres = None
    weather_states: list[str] = []

    # Rule 31: every draw gets its own seed from (base, date, lead, member), the
    # same in every arm that shares the base, never a constant reused across
    # dates and leads and never "wherever the generator happens to be".
    seed_base = resolve_base_seed()

    for member in members:
        if zarr_source is None:
            bundle_path = bundle_map[BundleKey(date, step, int(member))]
        draw_seeds.append(seed_draw(seed_base, int(date), int(step), int(member)))
        if zarr_source is None:
            result = predict_single_bundle(
                inference_model,
                datamodule,
                device,
                bundle_path,
                extra_args=extra_args,
                precision=precision,
                model_comm_group=model_comm_group,
                output_weather_state_mode=output_weather_state_mode,
                output_weather_states=output_weather_states,
            )
            source_bundle_paths.append(str(bundle_path))
        else:
            result = predict_single_zarr_row(
                inference_model,
                datamodule,
                device,
                zarr_source,
                date,
                step,
                extra_args=extra_args,
                precision=precision,
                model_comm_group=model_comm_group,
                output_weather_state_mode=output_weather_state_mode,
                output_weather_states=output_weather_states,
            )
        lon_lres = result.lon_lres
        lat_lres = result.lat_lres
        lon_hres = result.lon_hres
        lat_hres = result.lat_hres
        weather_states = result.weather_states
        if result.y is None:
            members_missing_target.append(int(member))

        if keep_outputs:
            x_members.append(_squeeze_leading_singletons(result.x, 2))
            y_members.append(None if result.y is None else _squeeze_leading_singletons(result.y, 2))
            y_pred_members.append(_squeeze_leading_singletons(result.y_pred, 2))

        del result
        _cleanup_member_outputs(device)

    ensemble = EnsemblePrediction(
        init_date=date_str_to_datetime64(date),
        lead_step_hours=int(step),
        member_ids=[int(member) for member in members],
        source_bundle_paths=source_bundle_paths,
        members_missing_target=members_missing_target,
        weather_states=weather_states,
        draw_seeds=draw_seeds,
        seed_base=int(seed_base),
        lon_lres=lon_lres,
        lat_lres=lat_lres,
        lon_hres=lon_hres,
        lat_hres=lat_hres,
        input_provenance=None if zarr_source is None else zarr_source.provenance(date, step),
    )
    if not keep_outputs:
        return ensemble

    x_stack = np.stack(x_members, axis=0)[None, ...]
    # zarr_split route: a saved state absent from in_lres is an all-NaN column of x; interpolate
    # zeros there and give x_interp the same NaN column (bundle x has no such column).
    nan_columns = np.isnan(x_stack).all(axis=tuple(range(x_stack.ndim - 1)))
    try:
        x_interp_stack = _compute_x_interp_for_export(
            inference_model=inference_model,
            x=np.where(nan_columns, np.float32(0.0), x_stack) if nan_columns.any() else x_stack,
            device=device,
            model_comm_group=model_comm_group,
        )
        if x_interp_stack is not None and nan_columns.any():
            x_interp_stack[..., nan_columns] = np.nan
    except (RuntimeError, ValueError) as exc:
        msg = str(exc)
        recognized = (
            "cannot export x_interp" in msg
            or "expects one ensemble" in msg
        )
        if not recognized:
            raise
        x_interp_stack = None
        print(
            f"WARNING date={date} step={step}: skipping x_interp export because the "
            f"inference model rejected ensemble interpolation ({exc})."
        )

    y_pred_stack = np.stack(y_pred_members, axis=0)[None, ...]
    used_missing_target_unsafe = False
    if any(member_y is not None for member_y in y_members):
        template = next(member_y for member_y in y_members if member_y is not None)
        y_filled = [
            member_y if member_y is not None else np.full_like(template, np.nan, dtype=np.float32)
            for member_y in y_members
        ]
        y_stack = np.stack(y_filled, axis=0)[None, ...]
        if members_missing_target:
            print(
                f"WARNING date={date} step={step}: missing target y for members {members_missing_target}; "
                "filled with NaN to keep y present in predictions file."
            )
    else:
        if not allow_missing_target_unsafe:
            raise SystemExit(
                f"No target truth (y) extracted for date={date} step={step}. "
                "Rebuild bundles with target_hres_* fields."
            )
        y_stack = np.full_like(y_pred_stack, np.nan, dtype=np.float32)
        used_missing_target_unsafe = True
        print(
            f"WARNING date={date} step={step}: no target y for any selected member; "
            "filled all y with NaN because --allow-missing-target-unsafe was set. "
            "Treat this file as prediction-only and non-canonical for truth-aware evaluation."
        )

    ensemble.x_stack = x_stack
    ensemble.y_stack = y_stack
    ensemble.y_pred_stack = y_pred_stack
    ensemble.x_interp_stack = x_interp_stack
    ensemble.used_missing_target_unsafe = used_missing_target_unsafe
    return ensemble
