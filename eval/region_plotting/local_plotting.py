from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import xarray as xr
from anemoi.training.diagnostics.maps import Coastlines
from matplotlib.backends.backend_pdf import PdfPages

from eval.checkpoint_interpolation import CheckpointResidualInterpolator, resolve_checkpoint_path

continents = Coastlines()
LOG = logging.getLogger(__name__)

DERIVED_MODEL_VARIABLE_SPECS: dict[str, dict[str, str]] = {
    "residuals_pred_0": {
        "left": "x_interp_0",
        "right": "y_pred_0",
        "title": "residuals_pred_0",
    },
    "residuals_pred": {
        "left": "x_interp",
        "right": "y_pred",
        "title": "residuals_pred",
    },
    "x_interp_minus_y_pred": {
        "left": "x_interp",
        "right": "y_pred",
        "title": "residuals_pred",
    },
    "residuals_0": {
        "left": "x_interp_0",
        "right": "y_0",
        "title": "residuals_0",
    },
    "residuals": {
        "left": "x_interp",
        "right": "y",
        "title": "residuals",
    },
    "x_interp_minus_y": {
        "left": "x_interp",
        "right": "y",
        "title": "residuals",
    },
}


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
        minmax_weather_states[weather_state] = [
            float(np.nanmin(fields_val)),
            float(np.nanmax(fields_val)),
        ]
    return minmax_weather_states


def supports_plot_variable(ds: xr.Dataset, model_var: str) -> bool:
    if model_var in ds.variables:
        return True
    spec = DERIVED_MODEL_VARIABLE_SPECS.get(model_var)
    if spec is None:
        return False
    return spec["left"] in ds.variables and spec["right"] in ds.variables


def _coord_name_for_array(ds: xr.Dataset, da: xr.DataArray, axis: str) -> str:
    attr_name = da.attrs.get(axis)
    if attr_name in ds.variables or attr_name in ds.coords:
        return attr_name

    dim_candidates = {
        "grid_point_hres": f"{axis}_hres",
        "grid_point_lres": f"{axis}_lres",
    }
    for dim_name, coord_name in dim_candidates.items():
        if dim_name in da.dims and (coord_name in ds.variables or coord_name in ds.coords):
            return coord_name

    raise KeyError(f"Could not infer {axis} coordinate for {da.name}")


def get_plot_data_array(ds: xr.Dataset, model_var: str) -> xr.DataArray:
    if model_var in ds.variables:
        return ds[model_var]

    spec = DERIVED_MODEL_VARIABLE_SPECS.get(model_var)
    if spec is None:
        raise KeyError(f"Unsupported model variable: {model_var}")

    derived = (ds[spec["left"]] - ds[spec["right"]]).rename(model_var)
    attrs = dict(ds[spec["left"]].attrs)
    if "lon" not in attrs and "lon" in ds[spec["right"]].attrs:
        attrs["lon"] = ds[spec["right"]].attrs["lon"]
    if "lat" not in attrs and "lat" in ds[spec["right"]].attrs:
        attrs["lat"] = ds[spec["right"]].attrs["lat"]
    derived.attrs = attrs
    return derived


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


def plot_variable_title(model_var: str) -> str:
    return DERIVED_MODEL_VARIABLE_SPECS.get(model_var, {}).get("title", model_var)


def is_residual_plot_variable(model_var: str) -> bool:
    return model_var == "y_diff" or model_var in DERIVED_MODEL_VARIABLE_SPECS


def _residual_vmax(da: xr.DataArray) -> float:
    values = np.asarray(da.values, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    vmax = float(np.max(finite))
    return vmax if vmax > 0 else 1.0


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
):
    list_model_variables = [v for v in list_model_variables if supports_plot_variable(ds_sample, v)]
    overlap = [
        model_var
        for model_var in list_model_variables
        if model_var in consistent_cbar and supports_plot_variable(ds_sample, model_var)
    ]
    minmax_weather_states = get_minmax_weather_states(ds_sample, weather_states, overlap)

    figsize = (len(list_model_variables) * 4, len(weather_states) * 3)
    fig, axs = plt.subplots(len(weather_states), len(list_model_variables), figsize=figsize)

    if len(list_model_variables) == 1:
        axs = np.array([axs]).transpose()
    if len(weather_states) == 1:
        axs = np.array([axs])

    ims = {}
    cbars = {}
    for i_ax0, weather_state in enumerate(weather_states):
        for i_ax1, model_var in enumerate(list_model_variables):
            da = get_plot_data_array(ds_sample, model_var)
            lon_name = _coord_name_for_array(ds_sample, da, "lon")
            lat_name = _coord_name_for_array(ds_sample, da, "lat")
            if "weather_state" in da.dims:
                da = da.sel(weather_state=weather_state)
            scatter_params = dict(
                x=ds_sample[lon_name].values,
                y=ds_sample[lat_name].values,
                c=da.values,
                s=75_000 / len(ds_sample[lon_name].values),
                alpha=1.0,
                rasterized=True,
            )

            if model_var in consistent_cbar and weather_state in minmax_weather_states:
                scatter_params.update(
                    vmin=minmax_weather_states[weather_state][0],
                    vmax=minmax_weather_states[weather_state][1],
                    cmap="viridis",
                )
            elif is_residual_plot_variable(model_var):
                vmax = _residual_vmax(da)
                scatter_params.update(vmin=-vmax, vmax=vmax, cmap="bwr")
            else:
                vmax = float(np.nanmax(da.values))
                vmin = float(np.nanmin(da.values))
                scatter_params.update(vmin=vmin, vmax=vmax, cmap="viridis")

            ims[(i_ax0, i_ax1)] = axs[i_ax0, i_ax1].scatter(**scatter_params)
            cbars[(i_ax0, i_ax1)] = plt.colorbar(
                ims[(i_ax0, i_ax1)],
                ax=axs[i_ax0, i_ax1],
                orientation="vertical",
                pad=0.05,
            )

    for i_ax0, _weather_state in enumerate(weather_states):
        axs[i_ax0, 0].set_ylabel("Latitude (°)", fontsize=12)
    for i_ax1, _model_var in enumerate(list_model_variables):
        axs[-1, i_ax1].set_xlabel("Longitude (°)", fontsize=12)

    for i_ax0, weather_state in enumerate(weather_states):
        for i_ax1, model_var in enumerate(list_model_variables):
            axs[i_ax0, i_ax1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            axs[i_ax0, i_ax1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d°"))
            axs[i_ax0, i_ax1].tick_params(axis="both", which="major", labelsize=10)
            axs[i_ax0, i_ax1].set_title(f"{plot_variable_title(model_var)} - {weather_state}")

            if "region" in ds_sample.attrs:
                axs[i_ax0, i_ax1].set_xlim(ds_sample.attrs["region"][2], ds_sample.attrs["region"][3])
                axs[i_ax0, i_ax1].set_ylim(ds_sample.attrs["region"][0], ds_sample.attrs["region"][1])
            continents.plot_continents(axs[i_ax0, i_ax1])
            axs[i_ax0, i_ax1].set_aspect("auto", adjustable=None)
            axs[i_ax0, i_ax1].grid(False)
            axs[i_ax0, i_ax1].patch.set_edgecolor("black")
            axs[i_ax0, i_ax1].patch.set_linewidth(2)
            cbars[(i_ax0, i_ax1)].outline.set_edgecolor("black")
            cbars[(i_ax0, i_ax1)].outline.set_linewidth(1.0)
            cbars[(i_ax0, i_ax1)].ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, fontsize=16, y=1.0)
    else:
        fig.suptitle(str(ds_sample.date.dt.strftime("%Y-%m-%d").values), fontsize=16, y=1.0)
    fig.tight_layout()
    return fig


def get_region_ds(ds: xr.Dataset, region_box: Union[str, list[int]] = "default") -> xr.Dataset:
    predefined_boxes = {
        "default": [40, 50, 0, 10],
        "pyrenees_alpes": [40, 50, 0, 10],
        "rocky_mountains": [35, 50, -120, -100],
        "amazon_forest": [-15, 5, -75, -45],
        "amazon_forest_core": [-8, 2, -72, -62],
        "southeast_asia": [-10, 20, 95, 150],
        "maritime_continent": [-2, 8, 108, 118],
        "west_sahara": [15, 30, -20, 0],
        "himalayas": [25, 40, 75, 100],
        "greatbarrier_reef": [-25, -10, 140, 155],
        "eastern_us": [25, 45, -90, -70],
        "eastern_us_coast": [30, 40, -82, -72],
        "idalia": [10, 40, -100, -70],
        "idalia_center": [18, 32, -92, -78],
        "central_africa": [-10, 10, 10, 30],
        "congo_basin": [-5, 5, 15, 25],
        "andes_central": [-34, -26, -74, -66],
        "european_arctic": [-25, 0, 75, 90],
        "rocky_mountains_central": [40, 45, -115, -105],
        "rocky_mountains_north": [45, 50, -115, -105],
        "rocky_mountains_south": [35, 40, -110, -100],
        "amazon_forest_central": [-5, 5, -75, -65],
        "amazon_forest_west": [-10, 0, -75, -65],
        "amazon_forest_east": [-10, 0, -55, -45],
        "southeast_asia_central": [0, 10, 100, 110],
        "southeast_asia_mainland": [10, 20, 100, 110],
        "southeast_asia_maritime": [-5, 5, 115, 125],
        "west_sahara_central": [20, 25, -15, -5],
        "west_sahara_coastal": [20, 25, -20, -10],
        "west_sahara_east": [20, 25, -10, 0],
        "himalayas_central": [30, 35, 80, 90],
        "himalayas_west": [30, 35, 75, 85],
        "himalayas_east": [25, 30, 90, 100],
        "greatbarrier_reef_central": [-20, -15, 145, 150],
        "greatbarrier_reef_north": [-15, -10, 145, 150],
        "greatbarrier_reef_south": [-25, -20, 150, 155],
        "eastern_us_central": [35, 40, -85, -75],
        "eastern_us_north": [40, 45, -80, -70],
        "eastern_us_south": [30, 35, -85, -75],
        "central_africa_congo": [-5, 5, 15, 25],
        "central_africa_north": [0, 10, 15, 25],
        "central_africa_south": [-10, 0, 20, 30],
    }

    if isinstance(region_box, str):
        region_box = predefined_boxes.get(region_box)
        if region_box is None:
            raise ValueError(f"Bounding box '{region_box}' is not predefined.")
    elif isinstance(region_box, list) and len(region_box) != 4:
        raise ValueError("Bounding box list must have exactly 4 elements.")

    lat_min, lat_max, lon_min, lon_max = region_box
    mask_hres = (
        (ds["lon_hres"] >= lon_min)
        & (ds["lon_hres"] <= lon_max)
        & (ds["lat_hres"] >= lat_min)
        & (ds["lat_hres"] <= lat_max)
    )
    region_hres = ds.isel(grid_point_hres=np.flatnonzero(np.asarray(mask_hres.values)))
    if "lon_lres" in ds.variables:
        mask_lres = (
            (ds["lon_lres"] >= lon_min)
            & (ds["lon_lres"] <= lon_max)
            & (ds["lat_lres"] >= lat_min)
            & (ds["lat_lres"] <= lat_max)
        )
        region_lres = region_hres.isel(
            grid_point_lres=np.flatnonzero(np.asarray(mask_lres.values))
        )
    else:
        region_lres = region_hres

    region_lres.attrs["region"] = region_box
    return region_lres


@dataclass
class LocalInferencePlotter:
    dir_exp: str
    name_exp: str
    name_predictions_file: str

    def __post_init__(self):
        self.ds = xr.open_dataset(os.path.join(self.dir_exp, self.name_exp, self.name_predictions_file))
        self.ds = ensure_x_interp_for_plotting(
            self.ds,
            predictions_path=Path(self.dir_exp) / self.name_exp / self.name_predictions_file,
        )
        if self.ds.attrs["grid"] == "O320":
            self.regions = [
                "amazon_forest",
                "european_arctic",
                "himalayas",
                "rocky_mountains",
                "west_sahara",
                "pyrenees_alpes",
                "eastern_us",
                "central_africa",
            ]
        elif self.ds.attrs["grid"] == "O1280":
            self.regions = [
                "rocky_mountains_central",
                "rocky_mountains_north",
                "rocky_mountains_south",
                "amazon_forest_central",
                "amazon_forest_west",
                "amazon_forest_east",
                "southeast_asia_central",
                "southeast_asia_mainland",
                "southeast_asia_maritime",
                "west_sahara_central",
                "west_sahara_coastal",
                "west_sahara_east",
                "himalayas_central",
                "himalayas_west",
                "himalayas_east",
                "greatbarrier_reef_central",
                "greatbarrier_reef_north",
                "greatbarrier_reef_south",
                "eastern_us_central",
                "eastern_us_north",
                "eastern_us_south",
                "central_africa_congo",
                "central_africa_north",
                "central_africa_south",
            ]
        else:
            raise ValueError(
                f"Unsupported grid type: {self.ds.attrs['grid']}. Please ensure grid is O320 or O1280."
            )

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
        with PdfPages(pdf_path) as pdf:
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
                        pdf.savefig(fig)
                        plt.close(fig)
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
                            pdf.savefig(fig)
                            plt.close(fig)
                            sample_count += 1
                        if sample_count >= num_samples_to_plot:
                            break

        LOG.info("Plot saved successfully at %s", pdf_path)
