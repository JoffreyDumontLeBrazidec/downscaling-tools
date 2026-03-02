from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from eval.region_plotting.local_plotting import LocalInferencePlotter, get_region_ds, plot_x_y
from manual_inference.data import DownscalingDatasetProcessor

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_VARIABLES = ["x_0", "y_0", "y_1", "y_pred_0", "y_pred_1"]
DEFAULT_WEATHER_STATES = ["10u", "10v", "2t", "msl", "z_500", "u_850", "v_850", "t_850"]

# Curated "interesting" O1280 regions discovered from the 2026-02-27 analysis workflow.
O1280_INTERESTING_REGIONS: dict[str, list[float]] = {
    "tibet_karakoram": [33.5, 41.5, 79.5, 91.5],
    "andes_central": [-37.5, -29.5, -75.5, -63.5],
    "greenland_south_tip": [58.5, 66.5, -54.5, -42.5],
    "horn_of_africa": [5.5, 13.5, 35.5, 47.5],
    "iran_zagros": [31.5, 39.5, 46.5, 58.5],
    "rockies_wyoming": [40.5, 48.5, -113.5, -101.5],
    "himalayas_west": [31.5, 39.5, 64.5, 76.5],
    "new_zealand_north": [-38.5, -30.5, 168.5, 180.5],
    "hawaii_big_island": [15.5, 23.5, -161.5, -149.5],
    "japan_hokkaido": [37.5, 45.5, 134.5, 146.5],
}


def _safe_datetime_str(value) -> str:
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _sample_meta_title(ds_region: xr.Dataset, region_name: str, sample_idx: int) -> str:
    parts = [f"{region_name}", f"sample_pos={sample_idx}"]
    if "sample" in ds_region.coords:
        try:
            sample_id = int(np.asarray(ds_region["sample"].isel(sample=sample_idx).values).item())
            parts.append(f"sample_id={sample_id}")
        except Exception:
            pass
    if "date" in ds_region and "sample" in ds_region["date"].dims:
        date_val = ds_region["date"].isel(sample=sample_idx).values
        date_str = _safe_datetime_str(date_val)
        if date_str:
            parts.append(f"date={date_str}")
    if "init_date" in ds_region and "sample" in ds_region["init_date"].dims:
        init_val = ds_region["init_date"].isel(sample=sample_idx).values
        init_str = _safe_datetime_str(init_val)
        if init_str:
            parts.append(f"init={init_str}")
    if "lead_step_hours" in ds_region and "sample" in ds_region["lead_step_hours"].dims:
        lead_val = ds_region["lead_step_hours"].isel(sample=sample_idx).values
        try:
            lead_int = int(np.asarray(lead_val).item())
            parts.append(f"lead_h={lead_int}")
        except Exception:
            pass
    return " | ".join(parts)


def _step_meta_title(region_name: str, step, forecast_ref_time) -> str:
    parts = [f"{region_name}", f"step={step}"]
    ft_str = _safe_datetime_str(forecast_ref_time)
    if ft_str:
        parts.append(f"forecast_ref={ft_str}")
    return " | ".join(parts)


def _save_custom_o1280_plots(
    *,
    ds_pred: xr.Dataset,
    out_pdf_path: Path,
    selected_variables: list[str],
    regions: dict[str, list[float]] = O1280_INTERESTING_REGIONS,
) -> None:
    weather_states = [w for w in DEFAULT_WEATHER_STATES if w in ds_pred["weather_state"].values]
    if out_pdf_path.exists():
        out_pdf_path.unlink()
    with PdfPages(out_pdf_path) as pdf:
        for region_name, region_box in regions.items():
            LOG.info("Plotting custom O1280 region %s %s", region_name, region_box)
            region_ds = get_region_ds(ds_pred, region_box)
            region_ds.attrs["region_name"] = region_name
            if "sample" in region_ds.dims:
                n_available = int(region_ds.sizes.get("sample", 0))
                n_to_plot = min(2, n_available)
                for sample_idx in range(n_to_plot):
                    title = _sample_meta_title(region_ds, region_name, sample_idx)
                    fig = plot_x_y(
                        ds_sample=region_ds.sel(sample=sample_idx),
                        list_model_variables=selected_variables,
                        weather_states=weather_states,
                        title=title,
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
            else:
                sample_count = 0
                for step in region_ds.step.values:
                    for ft in np.atleast_1d(region_ds.forecast_reference_time.values):
                        if sample_count >= 2:
                            break
                        title = _step_meta_title(region_name, step, ft)
                        fig = plot_x_y(
                            ds_sample=region_ds.sel(step=step, forecast_reference_time=ft),
                            list_model_variables=selected_variables,
                            weather_states=weather_states,
                            title=title,
                        )
                        pdf.savefig(fig)
                        plt.close(fig)
                        sample_count += 1
                    if sample_count >= 2:
                        break


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
        if ds_pred.attrs.get("grid") == "O1280":
            out_pdf = run_parent_dir / run_name / "all_regions_plots.pdf"
            LOG.info("Saving custom O1280 region plots for %s", run_name)
            _save_custom_o1280_plots(
                ds_pred=ds_pred,
                out_pdf_path=out_pdf,
                selected_variables=selected_variables,
            )
            return
    lip = LocalInferencePlotter(str(run_parent_dir), run_name, predictions_filename)
    LOG.info("Saving default region plots for %s", run_name)
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
