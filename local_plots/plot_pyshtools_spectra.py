from icecream import ic
import torch
import numpy as np
import sys

from hydra import compose, initialize
import matplotlib.pyplot as plt
from hydra.core.global_hydra import GlobalHydra
from anemoi.utils.config import DotDict
from omegaconf import OmegaConf

from anemoi.datasets import open_dataset
import xarray as xr

sys.path.append("/home/ecm5702/dev/notebooks")
import os
import seaborn as sns

import glob
import xarray as xr
import sys
import os
from icecream import ic
import torch
import numpy as np
import matplotlib.pyplot as plt
from anemoi.training.diagnostics.maps import Coastlines
import matplotlib.ticker as ticker
import seaborn as sns
import pyshtools
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from scipy.interpolate import griddata
import cmcrameri as cm


continents = Coastlines()
import itertools

sys.path.append("/home/ecm5702/dev/post_prepml/region_plotting")
sys.path.append("/home/ecm5702/dev/downscaling-tools/downscalingdata")
from downscalingdata import DownscalingDatasetProcessor
from power_spectrum import GridInfo, compute_spectra


# ---- Configuration (edit here in the notebook) ----
origin_data = "fdb"
if origin_data == "fdb":
    expver = "ituv"
    date = "20230801"
    number = "1/2/3"
    step = "48/120"
    sfc_param = "2t/10u/10v/sp"
    pl_param = "z/t/u/v"
    level = "500/850"

    low_res_reference_grib = "eefo_reference_o96-early-august.grib"
    high_res_reference_grib = "enfo_reference_o320-early-august.grib"

    # ---- Build dataset processor ----
    ds_pred_processor = DownscalingDatasetProcessor(
        expver=expver,
        date=date,
        number=number,
        step=step,
        sfc_param=sfc_param,
        pl_param=pl_param,
        level=level,
        low_res_reference_grib=low_res_reference_grib,
        high_res_reference_grib=high_res_reference_grib,
    )
    ds_pred_processor.request_predictions_dataset()
    ds_pred = ds_pred_processor.clean_predictions_dataset()
    ds_target = ds_pred_processor.build_target_dataset()
    ds_input = ds_pred_processor.build_input_dataset()
    print(ds_pred)
    print(ds_target)
    print(ds_input)
    ds = xr.merge([ds_pred, ds_target, ds_input])
elif origin_data == "local":
    dir_exp = "/home/ecm5702/scratch/aifs/checkpoint"
    name_exp = "3261d0a600604c6b85a5f9f5899aa39e"
    name_pred = "predictions.nc"
    ds = xr.open_dataset(os.path.join(dir_exp, name_exp, name_pred))
else:
    raise ValueError(f"Unknown origin_data: {origin_data}")

# normalisation
mean_weather_states = ds["x"].mean(dim=("grid_point_lres"))
std_weather_states = ds["x"].std(dim=("grid_point_lres"))

for var in ["x", "x_0", "y", "y_0", "y_pred", "y_pred_0", "y_pred_1"]:
    if var in ds:
        ds[var] = (ds[var] - mean_weather_states) / std_weather_states

grid_o320 = GridInfo("o320", np.column_stack((ds.lat_hres, ds.lon_hres)))
grid_o320.get_regular_grid_for_pyshtools(method="anemoi")

grid_o96 = GridInfo("o96", np.column_stack((ds.lat_lres, ds.lon_lres)))
grid_o96.get_regular_grid_for_pyshtools(method="anemoi")

weather_states = ["10u", "2t", "z_500", "u_850"]

colormap = cm.cm.roma

spectra_data = {}

for i, weather_state in enumerate(weather_states):
    ic(weather_state)
    spectra_data[weather_state] = {
        "spectra_truth": compute_spectra(
            ds.y_0.isel(forecast_reference_time=0, step=0).sel(
                weather_state=weather_state
            ),
            grid_o320,
            "anemoi",
        ),
        "spectra_pred0": compute_spectra(
            ds.y_pred_0.isel(forecast_reference_time=0, step=0).sel(
                weather_state=weather_state
            ),
            grid_o320,
            "anemoi",
        ),
        "spectra_input": compute_spectra(
            ds.x_0.isel(forecast_reference_time=0, step=0).sel(
                weather_state=weather_state
            ),
            grid_o96,
            "anemoi",
        ),
    }

fig, axs = plt.subplots(
    int(len(weather_states) / 2),
    int(len(weather_states) / 2),
    figsize=(15, 10),
    constrained_layout=True,
)
axs = axs.flatten()

colors = colormap(np.linspace(0, 1, len(spectra_data[weather_states[0]])))
for i, weather_state in enumerate(weather_states):
    for (label, data), color in zip(spectra_data[weather_state].items(), colors):
        axs[i].loglog(
            np.arange(1, data.shape[0]),
            data[1 : data.shape[0]],
            label=label,
            color=color,
        )

    axs[i].set_title(f"Weather State: {weather_state}", fontsize=12)
    axs[i].set_xlabel("Wavenumber", fontsize=12)
    axs[i].set_ylabel("Spectral Density", fontsize=12)
    axs[i].grid(True)

axs[-1].legend(fontsize=14)

fig.suptitle("Spectra Comparison Across Weather States", fontsize=14)
plt.savefig(
    f"/home/ecm5702/dev/outputs/pyshtools_spectra/spectra_comparison_{expver}.png",
    dpi=300,
)
