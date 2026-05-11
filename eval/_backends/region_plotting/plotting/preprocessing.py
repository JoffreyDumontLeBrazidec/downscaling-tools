"""Preprocessing helpers shared by region plotting scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from eval.checkpoint_interpolation import CheckpointResidualInterpolator, resolve_checkpoint_path


def ensure_member_zero_plot_variables(ds: xr.Dataset) -> xr.Dataset:
    """Add member-zero aliases used by legacy plotting code when they are missing."""
    alias_specs = (
        ("x", "x_0"),
        ("x_interp", "x_interp_0"),
        ("y", "y_0"),
        ("y_pred", "y_pred_0"),
    )
    updates: dict[str, xr.DataArray] = {}
    for base_name, alias_name in alias_specs:
        if alias_name in ds.variables or base_name not in ds.variables:
            continue
        da = ds[base_name]
        alias = da.isel(ensemble_member=0) if "ensemble_member" in da.dims else da
        alias = alias.rename(alias_name)
        alias.attrs = dict(da.attrs)
        updates[alias_name] = alias
    if updates:
        ds = ds.assign(**updates)
    return ds


def ensure_x_interp_for_plotting(
    ds: xr.Dataset,
    *,
    predictions_path: str | Path | None = None,
    checkpoint_path: str = "",
) -> xr.Dataset:
    """Ensure ``x_interp`` and member-zero aliases exist for plotting."""
    if "x_interp" not in ds.variables:
        if "x" not in ds.variables:
            return ensure_member_zero_plot_variables(ds)

        pred_dir = Path(predictions_path).expanduser().resolve().parent if predictions_path else Path.cwd()
        resolved_checkpoint = resolve_checkpoint_path(pred_dir=pred_dir, ds=ds, explicit_path=checkpoint_path)
        if resolved_checkpoint is None:
            return ensure_member_zero_plot_variables(ds)

        interpolator = CheckpointResidualInterpolator(resolved_checkpoint)
        x_da = ds["x"]
        x_np = np.asarray(x_da.values, dtype=np.float32)
        # Interpolator expects (grid_point_lres, features); transpose if needed.
        ws_first = "weather_state" in x_da.dims and x_da.dims.index("weather_state") == 0
        if ws_first:
            x_np = x_np.T  # (weather_state, grid_point_lres) → (grid_point_lres, weather_state)
        interpolated = interpolator.interpolate(x_np)  # → (grid_point_hres, weather_state)
        if ws_first:
            interpolated = interpolated.T  # → (weather_state, grid_point_hres)
        ws_coord = x_da.coords.get("weather_state", ds.coords.get("weather_state"))
        x_interp = xr.DataArray(
            interpolated.astype(np.float32),
            dims=("weather_state", "grid_point_hres"),
            coords={"weather_state": ws_coord} if ws_coord is not None else {},
            name="x_interp",
        )
        x_interp.attrs["lon"] = "lon_hres"
        x_interp.attrs["lat"] = "lat_hres"
        ds = ds.assign(x_interp=x_interp)
    return ensure_member_zero_plot_variables(ds)
