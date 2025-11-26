#!/usr/bin/env python3

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pyshtools
from anemoi.training.diagnostics.maps import Coastlines, EquirectangularProjection
from icecream import ic
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from pyshtools.expand import SHGLQ, SHExpandGLQ
from scipy.interpolate import griddata

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

continents = Coastlines()


@dataclass
class GridInfo:
    """Contains grid informations, projection."""

    name: str
    latlons: list

    def __post_init__(self):

        self.get_equirectangular_projection()
        # self.get_regular_grid_for_pyshtools()
        # self.get_regular_grid_for_anemoi

    def get_equirectangular_projection(self):
        """Get equirectangular projection of the latlons."""
        pc = EquirectangularProjection()
        lat, lon = self.latlons[:, 0], self.latlons[:, 1]
        self.equirec_pc_lon, self.equirec_pc_lat = pc(lon, lat)
        self.equirec_pc_lon = np.array(self.equirec_pc_lon)
        self.equirec_pc_lat = np.array(self.equirec_pc_lat)

    def get_regular_grid_for_pyshtools(self, method="anemoi"):
        """
        Get a regular grid for use with pyshtools.

        This function calculates a regular grid based on the equirectangular
        projection coordinates (longitude and latitude) stored in the instance.
        """
        # Calculate delta_lon and delta_lat on the projected grid
        delta_lon = abs(np.diff(self.equirec_pc_lon))
        non_zero_delta_lon = delta_lon[delta_lon != 0]
        delta_lat = abs(np.diff(self.equirec_pc_lat))
        non_zero_delta_lat = delta_lat[delta_lat != 0]

        n_pix_lat = int(
            np.floor(
                abs(self.equirec_pc_lat.max() - self.equirec_pc_lat.min())
                / abs(np.min(non_zero_delta_lat))
            ),
        )  # around 192 for O96. dimension lmax - 1
        if method == "anemoi":
            n_pix_lon = (n_pix_lat - 1) * 2 + 1  # 2*lmax + 1
        elif method == "shtools":
            n_pix_lon = (n_pix_lat) * 2  # + 1  # 2*lmax + 1
        elif method == "tobias":
            n_pix_lon = n_pix_lat
        else:
            raise ValueError("method must be 'anemoi' or 'shtools'")

        regular_pc_lon = np.linspace(
            self.equirec_pc_lon.min(), self.equirec_pc_lon.max(), n_pix_lon
        )
        regular_pc_lat = np.linspace(
            self.equirec_pc_lat.min(), self.equirec_pc_lat.max(), n_pix_lat
        )
        self.regular_grid_equirec_pc_lon, self.regular_grid_equirec_pc_lat = (
            np.meshgrid(regular_pc_lon, regular_pc_lat)
        )


def compute_spectra(
    data: np.ndarray, gridinfo: GridInfo, method: str = "anemoi"
) -> np.ndarray:
    """Compute spectral variability of a field by wavenumber."""

    interpolated_data = griddata(
        points=(gridinfo.equirec_pc_lon, gridinfo.equirec_pc_lat),
        values=data.squeeze(),
        xi=(
            gridinfo.regular_grid_equirec_pc_lon,
            gridinfo.regular_grid_equirec_pc_lat,
        ),
        method="cubic",
        fill_value=0.0,
    )

    if method == "anemoi":
        field = np.array(interpolated_data)

        lmax = field.shape[0] - 1
        zero_w = SHGLQ(lmax)
        coeffs_field = SHExpandGLQ(field, w=zero_w[1], zero=zero_w[0])

        coeff_amp = coeffs_field[0, :, :] ** 2 + coeffs_field[1, :, :] ** 2

        return np.sum(coeff_amp, axis=0)

    elif method == "shtools" or "pyshtools":
        grid = pyshtools.SHGrid.from_array(interpolated_data)
        coeffs = grid.expand()
        power_spectrum = coeffs.spectrum()
        return power_spectrum

    else:
        raise ValueError("method must be 'anemoi' or 'shtools'")


@dataclass
class ModelVariableAmplitude:
    """Used to compute amplitude related to a model variable (for example od n320)"""

    name: str
    gridinfo: GridInfo
    data: np.ndarray
    dict_weather_states: list[str]

    def __post_init__(self):

        # TODO: check that data corresponds to grid by checking number of points
        self.compute_amplitude(self.data, self.dict_weather_states)

    def compute_amplitude(self, data: np.ndarray, weather_states: list[str]):
        """Compute amplitude of data channels in list of weather states."""
        self.spectra_weather_states = {}

        for idx, (weather_state_idx, weather_state_name) in enumerate(
            weather_states.items()
        ):

            """
            interpolated_data = griddata(
                points=(self.gridinfo.equirec_pc_lon, self.gridinfo.equirec_pc_lat),
                values=data[..., weather_state_idx].squeeze(),
                xi=(
                    self.gridinfo.grid_equirec_pc_lon,
                    self.gridinfo.grid_equirec_pc_lat,
                ),
                method="cubic",
                fill_value=0.0,
            )

            spectrum_weather_state = np.array(compute_spectra(interpolated_data))
            """
            self.spectra_weather_states[weather_state_name] = compute_spectra(
                data[..., weather_state_idx].squeeze(), self.gridinfo, method="anemoi"
            )


@dataclass
class SpectraPlotter:
    """Used to plot only and deal with fig, axs, matplotlib parameters."""

    list_model_variable_amplitudes: list[ModelVariableAmplitude]

    def plot(self, list_weather_states: list[str], ncols=3):
        n_plots = len(list_weather_states)
        nrows = (
            n_plots + ncols - 1
        ) // ncols  # Calculate rows needed for the given number of columns

        figsize = (ncols * 5, nrows * 4)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        ax = np.atleast_2d(ax)  # Ensure ax is always a 2D array, even with a single row

        for plot_idx, weather_state_name in enumerate(list_weather_states):
            row, col = divmod(plot_idx, ncols)

            for model_variable in self.list_model_variable_amplitudes:
                amplitude = model_variable.spectra_weather_states[weather_state_name]
                ax[row, col].loglog(
                    np.arange(1, amplitude.shape[0]),
                    amplitude[1 : amplitude.shape[0]],
                    label=f"{model_variable.name}",
                )

            ax[row, col].legend()
            ax[row, col].set_title(weather_state_name, fontsize=12)
            ax[row, col].set_xlabel("$k$", fontsize=10)
            ax[row, col].set_ylabel("$P(k)$", fontsize=10)
            ax[row, col].grid(True, which="both", linestyle="--", linewidth=0.7)
            ax[row, col].set_aspect("auto")

        # Hide any unused subplots
        for i in range(plot_idx + 1, nrows * ncols):
            fig.delaxes(ax.flat[i])

        fig.tight_layout(pad=3)
        fig.suptitle("Weather State Spectra", fontsize=16)
        return fig


from anemoi.datasets import open_dataset

if __name__ == "__main__":
    ds_era5_o96 = open_dataset("aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6")
    ds_era5_n320 = open_dataset("aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6")
    ds_od_o96 = open_dataset("aifs-od-an-oper-0001-mars-o96-2016-2023-6h-v6")
    ds_od_n320 = open_dataset("aifs-od-an-oper-0001-mars-n320-2016-2023-6h-v6")

    grid_o96 = GridInfo(
        "o96", np.column_stack((ds_era5_o96.latitudes, ds_era5_o96.longitudes))
    )
    grid_n320 = GridInfo(
        "n320", np.column_stack((ds_era5_n320.latitudes, ds_era5_n320.longitudes))
    )

    dict_weather_states = {
        72: "z_500",
        23: "t_850",
        36: "u_850",
        49: "v_850",
        81: "2t",
        78: "10u",
        79: "10v",
        84: "sp",
        87: "tp",
        86: "cp",
    }

    ic("Compute amplitude era5 o96")
    amplitude_era5_o96 = ModelVariableAmplitude(
        "era5_o96", grid_o96, ds_era5_o96[0].T, dict_weather_states
    )
    amplitude_od_o96 = ModelVariableAmplitude(
        "od_o96", grid_o96, ds_od_o96[0].T, dict_weather_states
    )

    ic("Compute amplitude era5 n320")
    amplitude_era5_n320 = ModelVariableAmplitude(
        "era5_n320", grid_n320, ds_era5_n320[0].T, dict_weather_states
    )

    ic("Compute amplitude od n320")
    amplitude_od_n320 = ModelVariableAmplitude(
        "od_n320", grid_n320, ds_od_n320[0].T, dict_weather_states
    )

    spectra_plotter = SpectraPlotter(
        [amplitude_era5_o96, amplitude_od_o96, amplitude_era5_n320, amplitude_od_n320]
    )
    spectra_plotter.plot(["t_850", "u_850"])
    plt.savefig("spectra.png")
