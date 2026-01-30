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
from anemoi.training.diagnostics.local_inference.plot_predictions import (
    LocalInferencePlotter,
)
import argparse
import pandas as pd

import logging

# from post_prepml.tc.save_fields_idalia import to_xarray

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


sys.path.append("/home/ecm5702/dev/downscaling-tools/downscalingdata")
from downscalingdata import DownscalingDatasetProcessor


def save_local_plots(
    expver,
    date,
    number,
    step,
    sfc_param,
    pl_param,
    level,
    low_res_reference_grib,
    high_res_reference_grib,
):
    dir_exp = f"/home/ecm5702/prepml"
    name_predictions_file = "predictions.nc"
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
    logging.info("Requesting dataset")
    ds_pred_processor.request_predictions_dataset()
    logging.info("Cleaning dataset")
    ds_pred = ds_pred_processor.clean_predictions_dataset()
    logging.info("Building target dataset")
    ds_target = ds_pred_processor.build_target_dataset()
    logging.info("Building input dataset")
    ds_input = ds_pred_processor.build_input_dataset()
    logging.info("Merging datasets")
    print(ds_pred)
    print(ds_target)
    print(ds_input)
    ds = xr.merge([ds_pred, ds_target, ds_input])

    predictions_path = os.path.join(dir_exp, expver, name_predictions_file)
    logging.info(f"Saving predictions to {predictions_path}")
    if os.path.exists(predictions_path):
        os.remove(predictions_path)
    ds.to_netcdf(predictions_path, mode="w")
    time.sleep(
        30
    )  # to make sure all processes are ready and predictions.nc is well saved

    logging.info("Initializing LocalInferencePlotter")
    lip = LocalInferencePlotter(dir_exp, expver, name_predictions_file)
    logging.info("Saving plots")
    lip.save_plot(
        lip.regions,
        list_model_variables=["x_0", "y_0", "y_1", "y_pred_0", "y_pred_1"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process forecast verification parameters"
    )

    parser.add_argument("--expver", type=str, default="ik60", help="Experiment version")
    parser.add_argument("--date", type=str, default="20230801/20230810", help="Dates")
    parser.add_argument("--number", type=str, default="1/2/3", help="Numbers")
    parser.add_argument("--step", type=str, default="48/120", help="Steps")
    parser.add_argument(
        "--sfc_param", type=str, default="2t/10u/10v/sp", help="Surface parameters"
    )
    parser.add_argument(
        "--pl_param", type=str, default="z/t/u/v", help="Pressure level parameters"
    )
    parser.add_argument(
        "--low_res_reference_grib",
        type=str,
        default="eefo_reference_o96-early-august.grib",
        help="Low resolution reference grib file",
    )
    parser.add_argument(
        "--high_res_reference_grib",
        type=str,
        default="enfo_reference_o320-early-august.grib",
        help="High resolution reference grib file",
    )
    parser.add_argument("--level", type=str, default="500/850", help="Levels")

    args = parser.parse_args()
    save_local_plots(
        args.expver,
        args.date,
        args.number,
        args.step,
        args.sfc_param,
        args.pl_param,
        args.level,
        low_res_reference_grib=args.low_res_reference_grib,
        high_res_reference_grib=args.high_res_reference_grib,
    )


if __name__ == "__main__":
    main()
