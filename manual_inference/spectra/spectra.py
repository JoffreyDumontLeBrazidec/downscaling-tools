import os
from pathlib import Path

import numpy as np
import xarray as xr
from icecream import ic
import sys

sys.path.append("/home/ecm5702/dev/downscaling-tools/local_plots")
from power_spectrum import GridInfo, compute_spectra

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DIR_EXP = "/home/ecm5702/scratch/aifs/checkpoint"
NAME_EXP = "b4187af3064c426e805e83747b9f879d"
PREDICTION_PATTERN = "predictions*.nc"  # e.g. predictions.nc, predictions_epoch_*.nc

WEATHER_STATES = ["10u", "2t", "10v", "u_850"]


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Normalise x, y, y_pred* and residuals in-place (returns ds for chaining)."""

    # Normalisation on low-res x/y/y_pred*
    mean_weather_states = ds["x"].mean(dim=("sample", "grid_point_lres"))
    std_weather_states = ds["x"].std(dim=("sample", "grid_point_lres"))

    ds["x"] = (ds["x"] - mean_weather_states) / std_weather_states
    ds["y"] = (ds["y"] - mean_weather_states) / std_weather_states

    # y_pred may or may not exist depending on version; guard it
    if "y_pred" in ds:
        ds["y_pred"] = (ds["y_pred"] - mean_weather_states) / std_weather_states

    if "y_pred_0" in ds:
        ds["y_pred_0"] = (ds["y_pred_0"] - mean_weather_states) / std_weather_states
    if "y_pred_1" in ds:
        ds["y_pred_1"] = (ds["y_pred_1"] - mean_weather_states) / std_weather_states

    # Normalisation on residuals
    if "y_residuals" in ds:
        ds["y_residuals"] = (
            ds["y_residuals"]
            - ds["y_residuals"].mean(dim=("sample", "grid_point_hres"))
        ) / ds["y_residuals"].std(dim=("sample", "grid_point_hres"))

    if "y_pred_residuals" in ds:
        ds["y_pred_residuals"] = (
            ds["y_pred_residuals"]
            - ds["y_pred_residuals"].mean(
                dim=("sample", "ensemble_member", "grid_point_hres")
            )
        ) / ds["y_pred_residuals"].std(
            dim=("sample", "ensemble_member", "grid_point_hres")
        )

        # convenience: first ensemble member
        ds["y_pred_residuals_0"] = ds["y_pred_residuals"].sel(ensemble_member=0)

    return ds


def build_grids(ds: xr.Dataset) -> tuple[GridInfo, GridInfo]:
    """Build GridInfo objects for high-res (o320) and low-res (o96) from ds coords."""

    grid_o320 = GridInfo("o320", np.column_stack((ds.lat_hres, ds.lon_hres)))
    grid_o320.get_regular_grid_for_pyshtools(method="anemoi")

    grid_o96 = GridInfo("o96", np.column_stack((ds.lat_lres, ds.lon_lres)))
    grid_o96.get_regular_grid_for_pyshtools(method="anemoi")

    return grid_o320, grid_o96


def add_spectra_to_dataset(
    ds: xr.Dataset,
    weather_states: list[str],
) -> xr.Dataset:
    """
    Compute spectra for ALL samples and selected weather states and add them
    into the dataset.

    Resulting variables shape:
      spectra_truth   : (weather_state, sample, degree)
      spectra_pred0   : (weather_state, sample, degree)
      spectra_pred1   : (weather_state, sample, degree)  [if present]
      spectra_input   : (weather_state, sample, degree)
    """

    grid_o320, grid_o96 = build_grids(ds)

    spectra_truth_list = []
    spectra_pred0_list = []
    spectra_pred1_list = []
    spectra_input_list = []

    sample_vals = ds["sample"].values  # assumes a 'sample' coord exists

    for ws in weather_states:
        ic(f"Computing spectra for {ws}")

        # --- truth / y on high-res grid ---
        truth_specs = []
        for s in sample_vals:
            field = ds.y.sel(sample=s, weather_state=ws)
            arr = compute_spectra(field, grid_o320, "anemoi")
            truth_specs.append(arr)
        truth_specs = np.stack(truth_specs, axis=0)  # (sample, degree)

        da_truth = xr.DataArray(
            truth_specs,
            dims=("sample", "degree"),
            coords={
                "sample": sample_vals,
                "degree": np.arange(truth_specs.shape[1]),
            },
        ).expand_dims({"weather_state": [ws]})

        # --- pred0 on high-res grid ---
        pred0_specs = []
        for s in sample_vals:
            field = ds.y_pred_0.sel(sample=s, weather_state=ws)
            arr = compute_spectra(field, grid_o320, "anemoi")
            pred0_specs.append(arr)
        pred0_specs = np.stack(pred0_specs, axis=0)

        da_pred0 = xr.DataArray(
            pred0_specs,
            dims=("sample", "degree"),
            coords={
                "sample": sample_vals,
                "degree": np.arange(pred0_specs.shape[1]),
            },
        ).expand_dims({"weather_state": [ws]})

        # --- pred1 on high-res grid (optional) ---
        if "y_pred_1" in ds:
            pred1_specs = []
            for s in sample_vals:
                field = ds.y_pred_1.sel(sample=s, weather_state=ws)
                arr = compute_spectra(field, grid_o320, "anemoi")
                pred1_specs.append(arr)
            pred1_specs = np.stack(pred1_specs, axis=0)

            da_pred1 = xr.DataArray(
                pred1_specs,
                dims=("sample", "degree"),
                coords={
                    "sample": sample_vals,
                    "degree": np.arange(pred1_specs.shape[1]),
                },
            ).expand_dims({"weather_state": [ws]})

        # --- input x on low-res grid ---
        input_specs = []
        for s in sample_vals:
            field = ds.x.sel(sample=s, weather_state=ws)
            arr = compute_spectra(field, grid_o96, "anemoi")
            input_specs.append(arr)
        input_specs = np.stack(input_specs, axis=0)

        da_input = xr.DataArray(
            input_specs,
            dims=("sample", "degree"),
            coords={
                "sample": sample_vals,
                "degree": np.arange(input_specs.shape[1]),
            },
        ).expand_dims({"weather_state": [ws]})

        # collect
        spectra_truth_list.append(da_truth)
        spectra_pred0_list.append(da_pred0)
        if "y_pred_1" in ds:
            spectra_pred1_list.append(da_pred1)
        spectra_input_list.append(da_input)

    # Concatenate along weather_state
    ds["spectra_truth"] = xr.concat(spectra_truth_list, dim="weather_state")
    ds["spectra_pred0"] = xr.concat(spectra_pred0_list, dim="weather_state")
    if "y_pred_1" in ds:
        ds["spectra_pred1"] = xr.concat(spectra_pred1_list, dim="weather_state")
    ds["spectra_input"] = xr.concat(spectra_input_list, dim="weather_state")

    return ds


def process_prediction_files(
    dir_exp: str,
    name_exp: str,
    pattern: str = "predictions*.nc",
    weather_states: list[str] | None = None,
) -> None:
    """
    Loop over all prediction files matching pattern, normalise, compute spectra,
    and overwrite each file with spectra added.
    """
    if weather_states is None:
        weather_states = WEATHER_STATES

    exp_dir = Path(dir_exp) / name_exp
    files = sorted(exp_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {exp_dir}")

    ic(f"Found {len(files)} prediction files in {exp_dir}")

    for fpath in files:
        ic(f"Processing {fpath}")
        ds = xr.open_dataset(fpath)

        ds = normalize_dataset(ds)
        ds = add_spectra_to_dataset(ds, weather_states=weather_states)

        # Overwrite file with new variables included
        if fpath.exists():
            fpath.unlink()
        ds.to_netcdf(fpath, mode="w")
        ic(f"Saved spectra back to {fpath}")


if __name__ == "__main__":
    process_prediction_files(DIR_EXP, NAME_EXP, pattern=PREDICTION_PATTERN)
