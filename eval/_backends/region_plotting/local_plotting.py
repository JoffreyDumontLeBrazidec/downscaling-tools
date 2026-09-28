from __future__ import annotations

import copy
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from eval.checkpoint_interpolation import CheckpointResidualInterpolator, resolve_checkpoint_path
from .plotting.coordinate_utils import (
    _coord_name_for_array as shared_coord_name_for_array,
    get_region_ds as shared_get_region_ds,
    legacy_plotter_regions_for_grid,
)
from .plotting.datetime_utils import extract_date_from_dataset
from .plotting.preprocessing import ensure_member_zero_plot_variables as shared_ensure_member_zero_plot_variables
from .plotting.variable_utils import (
    DERIVED_MODEL_VARIABLE_SPECS as SHARED_DERIVED_MODEL_VARIABLE_SPECS,
    get_plot_data_array as shared_get_plot_data_array,
    is_residual_plot_variable as shared_is_residual_plot_variable,
    supports_plot_variable as shared_supports_plot_variable,
)

LOG = logging.getLogger(__name__)

DERIVED_MODEL_VARIABLE_SPECS = SHARED_DERIVED_MODEL_VARIABLE_SPECS


def get_minmax_weather_states(
    ds: xr.Dataset, weather_states: list[str], list_model_variables: list[str]
) -> dict[str, list[float]]:
    minmax_weather_states: dict[str, list[float]] = {}
    for weather_state in weather_states:
        fields: list[np.ndarray] = []
        for model_var in list_model_variables:
            if not supports_plot_variable(ds, model_var):
                continue
            da = get_plot_data_array(ds, model_var)
            if "weather_state" in da.dims:
                da = da.sel(weather_state=weather_state)
            fields.append(np.asarray(da.values).reshape(-1))
        if not fields:
            continue
        fields_val = np.concatenate(fields)
        finite = fields_val[np.isfinite(fields_val)]
        if finite.size == 0:
            continue
        # Cap the shared colorbar at the 99.5th percentile rather than the
        # absolute max. Heavy-tailed fields (tp) otherwise scale the whole
        # colorbar to a single extreme pixel, rendering widespread moderate
        # precip invisible. For near-Gaussian fields p99.5 ~ max, so this is a
        # no-op there; for precip it makes the bulk field visible while the
        # rarest cells saturate at the top colour.
        vmax = float(np.nanpercentile(finite, 99.5))
        vmin = float(np.nanmin(finite))
        if vmax <= vmin:  # degenerate / near-constant field
            vmax = float(np.nanmax(finite)) or (vmin + 1.0)
        minmax_weather_states[weather_state] = [vmin, vmax]
    return minmax_weather_states


def supports_plot_variable(ds: xr.Dataset, model_var: str) -> bool:
    return shared_supports_plot_variable(ds, model_var)


def _coord_name_for_array(ds: xr.Dataset, da: xr.DataArray, axis: str) -> str:
    return shared_coord_name_for_array(ds, da, axis)


def get_plot_data_array(ds: xr.Dataset, model_var: str) -> xr.DataArray:
    return shared_get_plot_data_array(ds, model_var)


def ensure_x_interp_for_plotting(
    ds: xr.Dataset,
    *,
    predictions_path: str | Path | None = None,
    checkpoint_path: str = "",
) -> xr.Dataset:
    if "x_interp" not in ds.variables:
        if "x" not in ds.variables:
            return ensure_member_zero_plot_variables(ds)

        pred_dir = Path(predictions_path).expanduser().resolve().parent if predictions_path else Path.cwd()
        resolved_checkpoint = resolve_checkpoint_path(pred_dir=pred_dir, ds=ds, explicit_path=checkpoint_path)
        if resolved_checkpoint is None:
            return ensure_member_zero_plot_variables(ds)

        interpolator = CheckpointResidualInterpolator(resolved_checkpoint)
        interpolated = interpolator.interpolate(np.asarray(ds["x"].values))
        x_interp = xr.DataArray(
            interpolated.astype(np.float32),
            dims=ds["y_pred"].dims,
            coords={dim: ds.coords[dim] for dim in ds["y_pred"].dims if dim in ds.coords},
            attrs=dict(ds["y_pred"].attrs),
            name="x_interp",
        )
        if "lon" not in x_interp.attrs and "lon_hres" in ds.coords:
            x_interp.attrs["lon"] = "lon_hres"
        if "lat" not in x_interp.attrs and "lat_hres" in ds.coords:
            x_interp.attrs["lat"] = "lat_hres"
        ds = ds.assign(x_interp=x_interp)
    return ensure_member_zero_plot_variables(ds)


def ensure_member_zero_plot_variables(ds: xr.Dataset) -> xr.Dataset:
    return shared_ensure_member_zero_plot_variables(ds)


def plot_variable_title(model_var: str) -> str:
    return DERIVED_MODEL_VARIABLE_SPECS.get(model_var, {}).get("title", model_var)


def is_residual_plot_variable(model_var: str) -> bool:
    return shared_is_residual_plot_variable(model_var)


def _residual_vmax(da: xr.DataArray) -> float:
    values = np.asarray(da.values, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    vmax = float(np.max(finite))
    return vmax if vmax > 0 else 1.0


def _region_extent(ds_sample: xr.Dataset) -> tuple[float, float, float, float]:
    """``(west, east, south, north)`` of a region dataset: its ``region`` attribute, else the data."""
    region = ds_sample.attrs.get("region")
    if region is not None and len(region) == 4:
        lat_min, lat_max, lon_min, lon_max = (float(v) for v in region)
        return lon_min, lon_max, lat_min, lat_max
    lons, lats = [], []
    for lon_name, lat_name in (("lon_hres", "lat_hres"), ("lon_lres", "lat_lres")):
        if lon_name in ds_sample.variables and ds_sample[lon_name].size:
            lons.append(np.asarray(ds_sample[lon_name].values, dtype=float))
            lats.append(np.asarray(ds_sample[lat_name].values, dtype=float))
    lon = np.concatenate(lons)
    lat = np.concatenate(lats)
    return float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))


def _panel_group(model_var: str, consistent_cbar: list[str]) -> str:
    """Colour-scale group of a panel: ``field`` (shared per row), ``difference``, or its own."""
    if model_var in consistent_cbar:
        return "field"
    if is_residual_plot_variable(model_var):
        return "difference"
    return f"own:{model_var}"


def plot_x_y(
    ds_sample: xr.Dataset,
    list_model_variables: list[str],
    weather_states: list[str],
    consistent_cbar: list[str] = [
        "x_0",
        "x_interp_0",
        "y_0",
        "y_pred_0",
        "x",
        "x_interp",
        "y",
        "y_pred",
        "x_interp_0",
        "y_pred_0",
        "y_pred_1",
        "y_pred_2",
        "x_interp_1",
        "x_interp_2",
        "x_0",
        "x_1",
        "x_2",
        "y_0",
        "y_1",
        "y_2",
    ],
    title: str | None = None,
    *,
    input_grid: str | None = None,
    target_grid: str | None = None,
    truth_label: str | None = None,
    input_label: str | None = None,
):
    """Map grid of one region: one row per weather state, one column per panel key.

    Every panel is a Cartopy map (projection from ``select_projection``, coastlines,
    borders, labelled grid lines) in display units from ``eval.plotting.variables``.
    Field panels listed in ``consistent_cbar`` share one colour scale per row; the
    difference panels (interpolated input minus truth / minus model) share one
    zero-centred ``RdBu_r`` scale per row (``BrBG`` for precipitation); any other panel
    (for example a noisy intermediate diffusion state) keeps its own scale. Each group
    has one colour bar with the unit. ``input_grid`` / ``target_grid`` (for example
    "O320", "O1280") name the grids in the panel titles; ``truth_label`` /
    ``input_label`` replace them when the source is known ("ENFO O1280").
    """
    from eval.plotting import convert, convert_difference, eval_style, extend_for, select_projection
    from eval.plotting import add_geography, shared_norm, symmetric_norm, variable_spec
    from eval.plotting.maps_helpers import (
        colorbar_beside,
        draw_unstructured,
        region_panel_title,
        set_inner_extent,
    )

    list_model_variables = [v for v in list_model_variables if supports_plot_variable(ds_sample, v)]
    extent = _region_extent(ds_sample)
    target_grid = target_grid or (str(ds_sample.attrs.get("grid", "")).strip() or None)
    nrows, ncols = len(weather_states), len(list_model_variables)
    groups = [_panel_group(v, consistent_cbar) for v in list_model_variables]

    # Colour-bar slots: one after every run of consecutive columns that share a scale.
    run_ends = [j for j in range(ncols) if j == ncols - 1 or groups[j + 1] != groups[j]]

    # Data, in display units, per (row, column).
    fields: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, weather_state in enumerate(weather_states):
        for j, model_var in enumerate(list_model_variables):
            da = get_plot_data_array(ds_sample, model_var)
            lon_name = _coord_name_for_array(ds_sample, da, "lon")
            lat_name = _coord_name_for_array(ds_sample, da, "lat")
            if len(ds_sample[lon_name].values) == 0:
                continue
            if "weather_state" in da.dims:
                da = da.sel(weather_state=weather_state)
            values = np.asarray(da.values, dtype=float)
            if groups[j] == "difference":
                values = np.asarray(convert_difference(weather_state, values), dtype=float)
            else:
                values = np.asarray(convert(weather_state, values), dtype=float)
            fields[(i, j)] = (
                np.asarray(ds_sample[lon_name].values, dtype=float),
                np.asarray(ds_sample[lat_name].values, dtype=float),
                values,
            )

    proj = select_projection(*extent)
    panel_w = 3.1
    west, east, south, north = extent
    aspect = max(0.45, min(1.6, (north - south) / max((east - west) * np.cos(np.radians(0.5 * (south + north))), 1e-6)))
    panel_h = panel_w * aspect + 0.45
    fig_w = ncols * panel_w + 0.95 * len(run_ends) + 0.8
    fig_h = nrows * panel_h + 1.0

    with eval_style():
        fig = plt.figure(figsize=(fig_w, fig_h))
        width_ratios: list[float] = []
        col_slot: dict[int, int] = {}
        for j in range(ncols):
            col_slot[j] = len(width_ratios)
            width_ratios.append(1.0)
            if j in run_ends:
                width_ratios.append(0.30)
        gs = fig.add_gridspec(nrows, len(width_ratios), width_ratios=width_ratios,
                              left=0.6 / fig_w, right=1.0 - 0.1 / fig_w,
                              bottom=0.35 / fig_h, top=1.0 - 0.75 / fig_h,
                              wspace=0.06, hspace=0.28)
        axs = np.empty((nrows, ncols), dtype=object)
        for i, weather_state in enumerate(weather_states):
            spec = variable_spec(weather_state)
            for j, model_var in enumerate(list_model_variables):
                ax = fig.add_subplot(gs[i, col_slot[j]], projection=proj)
                axs[i, j] = ax
                set_inner_extent(ax, extent)
                gl = add_geography(ax, label_size=6.5)
                if gl is not None:
                    gl.left_labels = j == 0
                    gl.bottom_labels = i == nrows - 1
                ax.set_title(
                    region_panel_title(model_var, input_grid=input_grid, target_grid=target_grid,
                                       truth=truth_label, input_name=input_label),
                    fontsize=8.5, pad=3,
                )
                if (i, j) not in fields:
                    ax.text(0.5, 0.5, "no data in region", transform=ax.transAxes,
                            ha="center", va="center", fontsize=8, color="0.4")

            # One scale per run of columns; the difference group is centred on zero.
            start = 0
            for end in run_ends:
                cols = list(range(start, end + 1))
                start = end + 1
                arrays = [fields[(i, j)][2] for j in cols if (i, j) in fields]
                if not arrays:
                    continue
                group = groups[cols[0]]
                if group == "difference":
                    norm, _ = symmetric_norm(*arrays, q=99.5)
                    cmap = spec.error_cmap()
                    label = f"{spec.name} difference ({spec.unit})" if spec.unit else f"{spec.name} difference"
                elif spec.signed:
                    norm = shared_norm(*arrays, q=(0.5, 99.5), centered=True)
                    cmap = spec.field_cmap()
                    label = spec.label
                elif spec.accumulated:
                    norm = shared_norm(*arrays, q=(0.0, 99.5), vmin=0.0)
                    cmap = spec.field_cmap()
                    label = spec.label
                else:
                    norm = shared_norm(*arrays, q=(0.5, 99.5))
                    cmap = spec.field_cmap()
                    label = spec.label
                mesh = None
                for j in cols:
                    if (i, j) not in fields:
                        continue
                    lon, lat, values = fields[(i, j)]
                    mesh = draw_unstructured(axs[i, j], lon, lat, values, extent, cmap=cmap, norm=norm)
                if mesh is not None:
                    colorbar_beside(fig, [axs[i, j] for j in cols], mesh, label,
                                    extend=extend_for(norm, *arrays), width=0.16 / fig_w,
                                    pad=0.08 / fig_w)
            axs[i, 0].text(-0.30, 0.5, spec.name, transform=axs[i, 0].transAxes, rotation=90,
                           ha="center", va="center", fontsize=9, fontweight="bold")

        fig.suptitle(title or extract_date_from_dataset(ds_sample) or "Unknown date", y=1.0 - 0.2 / fig_h)
    return fig


def get_region_ds(ds: xr.Dataset, region_box: Union[str, list[int]] = "default") -> xr.Dataset:
    return shared_get_region_ds(ds, region_box)


@dataclass
class LocalInferencePlotter:
    """Deprecated: use ``render_region_suite_from_predictions_file()`` instead."""

    dir_exp: str
    name_exp: str
    name_predictions_file: str

    def __post_init__(self):
        warnings.warn(
            "LocalInferencePlotter is deprecated. Use render_region_suite_from_predictions_file() from plot_regions.py.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.ds = xr.open_dataset(os.path.join(self.dir_exp, self.name_exp, self.name_predictions_file))
        self.ds = ensure_x_interp_for_plotting(
            self.ds,
            predictions_path=Path(self.dir_exp) / self.name_exp / self.name_predictions_file,
        )
        self.regions = legacy_plotter_regions_for_grid(str(self.ds.attrs["grid"]))

    def save_plot(
        self,
        list_regions: list[str],
        list_model_variables: list[str] = ["x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0"],
        weather_states: list[str] = ["10u", "10v", "2t", "msl", "tp", "z_500", "u_850", "v_850", "t_850"],
        num_samples_to_plot: int = 2,
    ) -> None:
        selected_model_variables = [v for v in list_model_variables if supports_plot_variable(self.ds, v)]
        if not selected_model_variables:
            raise ValueError(
                f"None of the requested model variables are available in {self.name_predictions_file}. "
                f"Requested={list_model_variables}"
            )
        available_weather_states = [str(v) for v in self.ds["weather_state"].values.tolist()]
        selected_weather_states = [w for w in weather_states if w in available_weather_states]
        if not selected_weather_states:
            selected_weather_states = available_weather_states

        pdf_path = f"{self.dir_exp}/{self.name_exp}/all_regions_plots.pdf"
        if os.path.exists(pdf_path):
            LOG.info("Removing existing PDF at %s", pdf_path)
            os.remove(pdf_path)
        from eval.plotting import FigureBook

        with FigureBook(pdf_path, png=True) as pdf:
            for region in list_regions:
                LOG.info("Plotting region %s", region)
                region_ds = get_region_ds(self.ds, region)
                region_ds.attrs["region_name"] = region

                if "sample" in region_ds.dims:
                    n_available = int(region_ds.sizes.get("sample", 0))
                    n_to_plot = min(num_samples_to_plot, n_available)
                    for sample in range(n_to_plot):
                        fig = plot_x_y(
                            ds_sample=region_ds.sel(sample=sample),
                            list_model_variables=selected_model_variables,
                            weather_states=selected_weather_states,
                            title=f"{region} - sample {sample}",
                        )
                        pdf.add(fig, name=f"{region}_sample{sample}")
                else:
                    sample_count = 0
                    for step in region_ds.step.values:
                        for ft in np.atleast_1d(region_ds.forecast_reference_time.values):
                            if sample_count >= num_samples_to_plot:
                                break
                            fig = plot_x_y(
                                ds_sample=region_ds.sel(step=step, forecast_reference_time=ft),
                                list_model_variables=selected_model_variables,
                                weather_states=selected_weather_states,
                                title=f"{region} - step {step} - forecast {pd.to_datetime(ft).strftime('%Y-%m-%d')}",
                            )
                            pdf.add(fig, name=f"{region}_step{step}")
                            sample_count += 1
                        if sample_count >= num_samples_to_plot:
                            break

        LOG.info("Plot saved successfully at %s", pdf_path)
