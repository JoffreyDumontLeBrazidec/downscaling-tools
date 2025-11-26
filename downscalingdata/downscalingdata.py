import os
import sys
import numpy as np
import xarray as xr
import earthkit as ek
import earthkit.data as ekd
from dataclasses import dataclass
from typing import List, Tuple
from icecream import ic
import time


import pandas as pd

import logging

# from post_prepml.tc.save_fields_idalia import to_xarray

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class DownscalingDatasetProcessor:

    expver: str = "ik60"
    date: str = "20230801/20230810"
    number: str = "1/2"
    step: str = "48/120"
    sfc_param: str = "2t/10u/10v/sp"
    pl_param: str = "z/t/u/v"
    level: str = "500/850"
    low_res_reference_grib: str = "eefo_reference_o96-early-august.grib"
    high_res_reference_grib: str = "enfo_reference_o320-early-august.grib"

    def request_predictions_dataset(self):

        sfc_request = {
            "class": "rd",
            "stream": "enfo",
            "time": "0",
            "domain": "g",
            "type": "pf",
            "levtype": "sfc",
            "expver": self.expver,
            "date": f"{self.date}",
            "number": self.number,
            "step": self.step,
            "param": self.sfc_param,
        }

        pl_request = {
            "class": "rd",
            "stream": "enfo",
            "time": "0",
            "domain": "g",
            "type": "pf",
            "levtype": "pl",
            "expver": self.expver,
            "date": f"{self.date}",
            "level": self.level,
            "number": self.number,
            "step": self.step,
            "param": self.pl_param,
        }

        ds_sfc = ekd.from_source("mars", sfc_request).to_xarray(squeeze=False)
        ds_sfc = ds_sfc.squeeze(dim="levtype", drop=True)
        ds_pl = ekd.from_source("mars", pl_request).to_xarray(squeeze=False)
        ds_pl = ds_pl.squeeze(dim="levtype", drop=True)

        self.ds = xr.merge([ds_sfc, ds_pl])

    def clean_predictions_dataset(self):
        ds_pred = self.ds.copy()
        ds_pred = ds_pred.rename({"values": "grid_point_hres"})
        grid_points = range(ds_pred.sizes["grid_point_hres"])
        ds_pred = ds_pred.assign_coords(grid_point_hres=grid_points)
        ds_pred = ds_pred.rename({"number": "ensemble_member"})
        ds_pred = ds_pred.rename({"latitude": "lat_hres", "longitude": "lon_hres"})
        if len(ds_pred.grid_point_hres) == 421120:
            ds_pred.attrs["grid"] = "O320"
        elif len(ds_pred.grid_point_hres) == 6599680:
            ds_pred.attrs["grid"] = "O1280"
        else:
            raise ValueError(
                f"Unsupported grid size: {len(ds_pred.grid_point_hres)}. "
                "Please ensure the grid size is either 421120 or 6599680."
            )

        ds_pred["z_500"] = ds_pred.z.sel(levelist=500).drop_vars("levelist")
        ds_pred["u_850"] = ds_pred.u.sel(levelist=850).drop_vars("levelist")
        ds_pred["v_850"] = ds_pred.v.sel(levelist=850).drop_vars("levelist")
        ds_pred["t_850"] = ds_pred.t.sel(levelist=850).drop_vars("levelist")
        ds_pred["lon_hres"] = ((ds_pred.lon_hres + 180) % 360) - 180
        if "lon_lres" in ds_pred:
            ds_pred["lon_lres"] = ((ds_pred.lon_lres + 180) % 360) - 180
        selected_weather_vars = [
            "z_500",
            "u_850",
            "v_850",
            "t_850",
            "2t",
            "10u",
            "10v",
            "sp",
        ]
        ds_pred["y_pred"] = xr.concat(
            [ds_pred[var] for var in selected_weather_vars],
            dim=pd.Index(selected_weather_vars, name="weather_state"),
        )
        ds_pred = ds_pred.drop_vars(
            [
                "z",
                "t",
                "u",
                "v",
                "z_500",
                "u_850",
                "v_850",
                "t_850",
                "2t",
                "sp",
                "10u",
                "10v",
                "levelist",
            ]
        )
        for var in ds_pred.variables.values():
            if "_earthkit" in var.attrs:
                del var.attrs["_earthkit"]

        ds_pred.y_pred.attrs["lon"] = "lon_hres"
        ds_pred.y_pred.attrs["lat"] = "lat_hres"
        for i, member in enumerate(ds_pred.ensemble_member.values):
            member_data = ds_pred.y_pred.sel(ensemble_member=member)
            ds_pred[f"y_pred_{i}"] = member_data

        ds_pred = ds_pred.assign_coords(
            step=ds_pred.step.values.astype("timedelta64[h]").astype(int)
        )
        ds_pred = ds_pred.sel(
            forecast_reference_time=self.date.split("/"),
            step=[int(s) for s in self.step.split("/")],
            ensemble_member=[int(n) for n in self.number.split("/")],
        )
        return ds_pred

    def build_target_dataset(self):
        target_ds_file = (
            os.path.join(
                "/home/ecm5702/hpcperm/reference_grib",
                self.high_res_reference_grib,
            ),
        )
        ic(target_ds_file)
        ds_enfo = ekd.from_source(
            "file",
            target_ds_file,
        ).to_xarray()

        ds_enfo = ds_enfo.copy()
        step_hours = ds_enfo.step.values.astype("timedelta64[h]").astype(int)
        ds_enfo = ds_enfo.assign_coords(step=step_hours)
        ds_enfo = ds_enfo.rename(
            {
                "number": "ensemble_member",
                "latitude": "lat_hres",
                "longitude": "lon_hres",
                "z": "z_500",
                "u": "u_850",
                "v": "v_850",
                "t": "t_850",
                "values": "grid_point_hres",
            }
        )
        grid_points = range(ds_enfo.sizes["grid_point_hres"])
        ds_enfo = ds_enfo.assign_coords(grid_point_hres=grid_points)

        selected_weather_vars = [
            "z_500",
            "u_850",
            "v_850",
            "t_850",
            "2t",
            "10u",
            "10v",
            "sp",
        ]
        ds_enfo["y"] = xr.concat(
            [ds_enfo[var] for var in selected_weather_vars],
            dim=pd.Index(selected_weather_vars, name="weather_state"),
        )
        ds_enfo = ds_enfo.drop_vars(
            ["z_500", "u_850", "v_850", "t_850", "2t", "10u", "10v", "sp"],
        )

        ds_enfo.y.attrs["lon"] = "lon_hres"
        ds_enfo.y.attrs["lat"] = "lat_hres"
        ds_enfo["lon_hres"] = ((ds_enfo.lon_hres + 180) % 360) - 180

        for i, member in enumerate(ds_enfo.ensemble_member.values):
            member_data = ds_enfo.y.sel(ensemble_member=member)
            ds_enfo[f"y_{i}"] = member_data

        ds_enfo = ds_enfo.assign_coords(
            step=ds_enfo.step.values.astype("timedelta64[h]").astype(int)
        )
        for var in ds_enfo.variables.values():
            if "_earthkit" in var.attrs:
                del var.attrs["_earthkit"]

        ic(
            self.date.split("/"),
            [int(s) for s in self.step.split("/")],
            [int(n) for n in self.number.split("/")],
        )
        ds_enfo = ds_enfo.sel(
            forecast_reference_time=self.date.split("/"),
            step=[int(s) for s in self.step.split("/")],
            ensemble_member=[int(n) for n in self.number.split("/")],
        )
        return ds_enfo

    def build_input_dataset(self):
        input_ds_file = os.path.join(
            "/home/ecm5702/hpcperm/reference_grib",
            self.low_res_reference_grib,
        )

        ic(input_ds_file)
        ds_eefo = ekd.from_source(
            "file",
            input_ds_file,
        ).to_xarray()

        ds_eefo = ds_eefo.copy()
        step_hours = ds_eefo.step.values.astype("timedelta64[h]").astype(int)
        ds_eefo = ds_eefo.assign_coords(step=step_hours)

        ds_eefo = ds_eefo.rename({"values": "grid_point_lres"})
        grid_points = range(ds_eefo.sizes["grid_point_lres"])
        ds_eefo = ds_eefo.assign_coords(grid_point_lres=grid_points)
        ds_eefo = ds_eefo.rename({"number": "ensemble_member"})
        ds_eefo = ds_eefo.rename(
            {
                "latitude": "lat_lres",
                "longitude": "lon_lres",
                "z": "z_500",
                "u": "u_850",
                "v": "v_850",
                "t": "t_850",
            }
        )
        ds_eefo["lon_lres"] = ((ds_eefo.lon_lres + 180) % 360) - 180

        selected_weather_vars = [
            "z_500",
            "u_850",
            "v_850",
            "t_850",
            "2t",
            "10u",
            "10v",
            "sp",
        ]
        ds_eefo["x"] = xr.concat(
            [ds_eefo[var] for var in selected_weather_vars],
            dim=pd.Index(selected_weather_vars, name="weather_state"),
        )
        ds_eefo = ds_eefo.drop_vars(
            ["z_500", "u_850", "v_850", "t_850", "2t", "10u", "10v", "sp"]
        )

        ds_eefo.x.attrs["lon"] = "lon_lres"
        ds_eefo.x.attrs["lat"] = "lat_lres"
        for i, member in enumerate(ds_eefo.ensemble_member.values):
            member_data = ds_eefo.x.sel(ensemble_member=member)
            ds_eefo[f"x_{i}"] = member_data

        ds_eefo = ds_eefo.assign_coords(
            step=ds_eefo.step.values.astype("timedelta64[h]").astype(int)
        )
        for var in ds_eefo.variables.values():
            if "_earthkit" in var.attrs:
                del var.attrs["_earthkit"]
        ds_eefo = ds_eefo.sel(
            forecast_reference_time=self.date.split("/"),
            step=[int(s) for s in self.step.split("/")],
            ensemble_member=[int(n) for n in self.number.split("/")],
        )
        return ds_eefo
