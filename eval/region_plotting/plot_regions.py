from __future__ import annotations

import argparse
import logging
from pathlib import Path

import xarray as xr
from anemoi.training.diagnostics.local_inference.plot_predictions import (
    LocalInferencePlotter,
)

from manual_inference.data import DownscalingDatasetProcessor

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_VARIABLES = ["x_0", "y_0", "y_1", "y_pred_0", "y_pred_1"]


def build_predictions_dataset_from_expver(
    *,
    expver: str,
    date: str,
    number: str,
    step: str,
    sfc_param: str,
    pl_param: str,
    level: str,
    low_res_reference_grib: str,
    high_res_reference_grib: str,
) -> xr.Dataset:
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
    LOG.info("Requesting MARS prediction dataset for expver=%s", expver)
    ds_pred_processor.request_predictions_dataset()
    LOG.info("Cleaning prediction dataset")
    ds_pred = ds_pred_processor.clean_predictions_dataset()
    LOG.info("Building target dataset")
    ds_target = ds_pred_processor.build_target_dataset()
    LOG.info("Building input dataset")
    ds_input = ds_pred_processor.build_input_dataset()
    LOG.info("Merging prediction/target/input datasets")
    return xr.merge([ds_pred, ds_target, ds_input])


def save_predictions_dataset(ds: xr.Dataset, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    ds.to_netcdf(out_path, mode="w")
    LOG.info("Saved predictions to %s", out_path)
    return out_path


def run_region_plots_from_predictions(
    *,
    run_parent_dir: str | Path,
    run_name: str,
    predictions_filename: str = "predictions.nc",
    model_variables: list[str] | None = None,
) -> None:
    run_parent_dir = Path(run_parent_dir)
    run_parent_dir.mkdir(parents=True, exist_ok=True)
    model_variables = model_variables or DEFAULT_MODEL_VARIABLES
    predictions_path = run_parent_dir / run_name / predictions_filename
    with xr.open_dataset(predictions_path) as ds_pred:
        available = set(ds_pred.data_vars)
    selected_variables = [v for v in model_variables if v in available]
    if not selected_variables:
        raise ValueError(
            f"None of the requested model variables are available in {predictions_path}. "
            f"Requested={model_variables}"
        )
    lip = LocalInferencePlotter(str(run_parent_dir), run_name, predictions_filename)
    LOG.info("Saving region plots for %s", run_name)
    lip.save_plot(lip.regions, list_model_variables=selected_variables)


def save_local_plots(
    expver: str,
    date: str,
    number: str,
    step: str,
    sfc_param: str,
    pl_param: str,
    level: str,
    low_res_reference_grib: str,
    high_res_reference_grib: str,
    output_root: str = "/home/ecm5702/prepml",
) -> Path:
    run_dir = Path(output_root) / expver
    predictions_filename = "predictions.nc"
    ds = build_predictions_dataset_from_expver(
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
    predictions_path = save_predictions_dataset(ds, run_dir / predictions_filename)
    run_region_plots_from_predictions(
        run_parent_dir=Path(output_root),
        run_name=expver,
        predictions_filename=predictions_filename,
    )
    return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build predictions from expver and save local region plots.")
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
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/ecm5702/prepml",
        help="Parent output directory where <expver>/predictions.nc and plots are stored.",
    )

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
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
