from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import xarray as xr
from anemoi.training.diagnostics.maps import Coastlines
from matplotlib.backends.backend_pdf import PdfPages

continents = Coastlines()
LOG = logging.getLogger(__name__)


def get_minmax_weather_states(
    ds: xr.Dataset, weather_states: list[str], list_model_variables: list[str]
) -> dict[str, list[float]]:
    minmax_weather_states: dict[str, list[float]] = {}
    for weather_state in weather_states:
        fields_val = np.concatenate(
            [
                ds.sel(weather_state=weather_state)[model_var].values.flatten().ravel()
                for model_var in list_model_variables
                if model_var in ds.variables
            ]
        ).tolist()
        minmax_weather_states[weather_state] = [np.min(fields_val), np.max(fields_val)]
    return minmax_weather_states


def plot_x_y(
    ds_sample: xr.Dataset,
    list_model_variables: list[str],
    weather_states: list[str],
    consistent_cbar: list[str] = [
        "x",
        "y",
        "y_pred",
        "y_pred_0",
        "y_pred_1",
        "y_pred_2",
        "x_0",
        "x_1",
        "x_2",
        "y_0",
        "y_1",
        "y_2",
    ],
    title: str | None = None,
):
    overlap = list(set(consistent_cbar) & set(list_model_variables))
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
            scatter_params = dict(
                x=ds_sample[ds_sample[model_var].attrs.get("lon", None)].values,
                y=ds_sample[ds_sample[model_var].attrs.get("lat", None)].values,
                c=ds_sample[model_var].sel(weather_state=weather_state).values,
                s=75_000 / len(ds_sample[ds_sample[model_var].attrs.get("lon", None)].values),
                alpha=1.0,
                rasterized=True,
            )

            if model_var in consistent_cbar:
                scatter_params.update(
                    vmin=minmax_weather_states[weather_state][0],
                    vmax=minmax_weather_states[weather_state][1],
                    cmap="viridis",
                )
            elif model_var == "y_diff":
                vmax = np.max(np.abs(ds_sample[model_var].sel(weather_state=weather_state)))
                scatter_params.update(vmin=-vmax, vmax=vmax, cmap="bwr")
            else:
                vmax = np.max(ds_sample[model_var].sel(weather_state=weather_state))
                vmin = np.min(ds_sample[model_var].sel(weather_state=weather_state))
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
            axs[i_ax0, i_ax1].set_title(f"{model_var} - {weather_state}")

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
        "southeast_asia": [-10, 20, 95, 150],
        "west_sahara": [15, 30, -20, 0],
        "himalayas": [25, 40, 75, 100],
        "greatbarrier_reef": [-25, -10, 140, 155],
        "eastern_us": [25, 45, -90, -70],
        "idalia": [10, 40, -100, -70],
        "idalia_center": [18, 32, -92, -78],
        "central_africa": [-10, 10, 10, 30],
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
    region_hres = ds.sel(grid_point_hres=ds.grid_point_hres.where(mask_hres, drop=True))
    if "lon_lres" in ds.variables:
        mask_lres = (
            (ds["lon_lres"] >= lon_min)
            & (ds["lon_lres"] <= lon_max)
            & (ds["lat_lres"] >= lat_min)
            & (ds["lat_lres"] <= lat_max)
        )
        region_lres = region_hres.sel(
            grid_point_lres=ds.grid_point_lres.where(mask_lres, drop=True)
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
        list_model_variables: list[str] = ["x_0", "y_0", "y_1", "y_pred_0", "y_pred_1"],
        weather_states: list[str] = ["10u", "10v", "2t", "z_500", "u_850", "v_850", "t_850"],
        num_samples_to_plot: int = 2,
    ) -> None:
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
                            list_model_variables=list_model_variables,
                            weather_states=weather_states,
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
                                list_model_variables=list_model_variables,
                                weather_states=weather_states,
                                title=f"{region} - step {step} - forecast {pd.to_datetime(ft).strftime('%Y-%m-%d')}",
                            )
                            pdf.savefig(fig)
                            plt.close(fig)
                            sample_count += 1
                        if sample_count >= num_samples_to_plot:
                            break

        LOG.info("Plot saved successfully at %s", pdf_path)
