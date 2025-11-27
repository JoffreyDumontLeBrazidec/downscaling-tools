import os
import sys
import torch
import numpy as np
import xarray as xr
import earthkit as ek
import earthkit.data as ekd
from einops import rearrange
from icecream import ic
from anemoi.training.train.tasks.downscaler import match_tensor_channels
from dataclasses import dataclass

sys.path.append("/home/ecm5702/dev/downscaling-tools")
from utils import tensors_to_numpy
from anemoi.training.data.datamodule import DownscalingAnemoiDatasetsDataModule

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
sys.path.append("/home/ecm5702/dev/inference")


class ZarrDataBatch:
    def __init__(self, config_checkpoint, frequency):
        self.config_checkpoint = config_checkpoint
        self.config_checkpoint.dataloader.validation.frequency = frequency
        graph_data = torch.load(
            os.path.join(
                config_checkpoint.hardware.paths.output,
                "graphs",
                config_checkpoint.hardware.files.graph,
            ),
            weights_only=False,
        )
        self.datamodule = DownscalingAnemoiDatasetsDataModule(
            self.config_checkpoint, graph_data
        )
        self.dataset = self.datamodule.ds_valid

    def prepare_tensors(self, idx: int, N_samples: int):
        x_in = self.dataset.data[idx : idx + N_samples][0]
        x_in_hres = self.dataset.data[idx : idx + N_samples][1]
        y = self.dataset.data[idx : idx + N_samples][2]

        self.x_in = self._process_tensor(x_in)[:, None, ...]
        self.x_in_hres = self._process_tensor(x_in_hres)[:, None, ...]
        self.y = self._process_tensor(y)[:, None, ...]

        self.date = self.dataset.data.dates[idx : idx + N_samples]

        self.lon_lres = self.dataset.data.longitudes[0]
        self.lat_lres = self.dataset.data.latitudes[0]
        self.lon_hres = self.dataset.data.longitudes[2]
        self.lat_hres = self.dataset.data.latitudes[2]


    def _process_tensor(self, tensor):
        tensor = rearrange(
            tensor,
            "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
        )
        return torch.from_numpy(tensor)  # .to(self.device)


@dataclass
class LocalGribDataBatch:
    date = "20230802"
    number = "1/2/3"
    step = "96/240"
    sfc_param = "2t/10u/10v/sp"
    pl_param = "z/t/u/v"
    level = "500/850"
    low_res_reference_grib = "eefo_reference_o96-early-august.grib"
    high_res_reference_grib = "enfo_reference_o320-early-august.grib"

    def prepare_tensors(self):

        self.prepare_input_tensor()
        self.prepare_target_tensor()

    def prepare_input_tensor(self):
        input_ds_file = os.path.join(
            "/home/ecm5702/hpcperm/reference_grib",
            self.low_res_reference_grib,
        )

        input_dataset = ekd.from_source(
            "file",
            input_ds_file,
        ).to_xarray()
        step_hours = input_dataset.step.values.astype("timedelta64[h]").astype(int)
        input_dataset = input_dataset.assign_coords(step=step_hours)        
        input_dataset = input_dataset.rename({"values": "grid_point_lres"})
        grid_points = range(input_dataset.sizes["grid_point_lres"])
        input_dataset = input_dataset.assign_coords(grid_point_lres=grid_points)
        input_dataset = input_dataset.rename({"number": "ensemble_member"})
        input_dataset["lon_lres"] = ((input_dataset.lon_lres + 180) % 360) - 180
        input_dataset = input_dataset.rename(
            {
                "latitude": "lat_lres",
                "longitude": "lon_lres",
            }
        )

        ds_target = ds_pred_processor.build_target_dataset()
        ds_input = ds_pred_processor.build_input_dataset()        

@dataclass
class WeatherDataBatch:
    device: str = "cuda"

    def prepare_tensors(self, origin, **kwargs):
        if origin == "zarr":
            zarr_data_batch = ZarrDataBatch(
                kwargs["config_checkpoint"], kwargs["frequency"]
            )
            zarr_data_batch.prepare_tensors(kwargs["idx"], kwargs["N_samples"])
            self.x_in = zarr_data_batch.x_in
            self.x_in_hres = zarr_data_batch.x_in_hres
            self.y = zarr_data_batch.y
            self.date = zarr_data_batch.date
            self.lon_lres = zarr_data_batch.lon_lres
            self.lat_lres = zarr_data_batch.lat_lres
            self.lon_hres = zarr_data_batch.lon_hres
            self.lat_hres = zarr_data_batch.lat_hres
        elif origin == "anemoi" or origin == "fdb":


    def prepare_tensors_for_model(self, idx: int, N_samples: int):
        self.prepare(idx, N_samples)
        self.x_in = self.x_in.to(self.device)
        self.x_in_hres = self.x_in_hres.to(self.device)
        self.y = self.y.to(self.device)

    def send_tensors_to_numpy(self):
        tensor_attributes = [
            "x_in",
            "x_in_hres",
            "y",
            "x_in_interp_to_hres",
            "y_residuals",
            "y_pred",
            "y_pred_residuals",
        ]
        for attr in tensor_attributes:
            if hasattr(self, attr):
                setattr(self, attr, tensors_to_numpy(getattr(self, attr)))