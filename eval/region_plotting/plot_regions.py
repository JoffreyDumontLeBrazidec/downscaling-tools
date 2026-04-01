from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from eval.region_plotting.local_plotting import (
    LocalInferencePlotter,
    ensure_x_interp_for_plotting,
    get_region_ds,
    plot_x_y,
    supports_plot_variable,
)
from manual_inference.data import DownscalingDatasetProcessor

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_VARIABLES = ["x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0"]
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

PREDICTION_REGION_BOXES: dict[str, list[float]] = {
    **O1280_INTERESTING_REGIONS,
    "amazon_forest": [-15.0, 5.0, -75.0, -45.0],
    "eastern_us": [25.0, 45.0, -90.0, -70.0],
    "himalayas": [25.0, 40.0, 75.0, 100.0],
    "southeast_asia": [-10.0, 20.0, 95.0, 150.0],
    "central_africa": [-10.0, 10.0, 10.0, 30.0],
    "idalia": [10.0, 40.0, -100.0, -70.0],
    "idalia_center": [18.0, 32.0, -92.0, -78.0],
    "franklin": [10.0, 40.0, -80.0, -50.0],
    "franklin_center": [18.0, 32.0, -73.0, -59.0],
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
            sample_coord = ds_region["sample"]
            if "sample" in sample_coord.dims:
                sample_value = sample_coord.isel(sample=sample_idx).values
            else:
                sample_value = sample_coord.values
            sample_id = int(np.asarray(sample_value).item())
            parts.append(f"sample_id={sample_id}")
        except Exception:
            pass
    if "date" in ds_region:
        date_da = ds_region["date"]
        if "sample" in date_da.dims:
            date_val = date_da.isel(sample=sample_idx).values
        else:
            date_val = date_da.values
        date_str = _safe_datetime_str(date_val)
        if date_str:
            parts.append(f"date={date_str}")
    if "init_date" in ds_region:
        init_da = ds_region["init_date"]
        if "sample" in init_da.dims:
            init_val = init_da.isel(sample=sample_idx).values
        else:
            init_val = init_da.values
        init_str = _safe_datetime_str(init_val)
        if init_str:
            parts.append(f"init={init_str}")
    if "lead_step_hours" in ds_region:
        lead_da = ds_region["lead_step_hours"]
        if "sample" in lead_da.dims:
            lead_val = lead_da.isel(sample=sample_idx).values
        else:
            lead_val = lead_da.values
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


def _select_prediction_variables(
    ds_pred: xr.Dataset,
    model_variables: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    requested_model_variables = model_variables or DEFAULT_MODEL_VARIABLES
    selected_variables: list[str] = []
    for model_var in requested_model_variables:
        if not supports_plot_variable(ds_pred, model_var):
            continue
        selected_variables.append(model_var)
    if not selected_variables:
        raise ValueError(
            f"None of the requested model variables are available. Requested={requested_model_variables}"
        )

    available_weather_states = [str(v) for v in ds_pred["weather_state"].values.tolist()]
    selected_weather_states = [w for w in DEFAULT_WEATHER_STATES if w in available_weather_states]
    if not selected_weather_states:
        selected_weather_states = available_weather_states
    return selected_variables, selected_weather_states


def _write_suite_manifest(
    *,
    out_root: Path,
    suite_kind: str,
    predictions_path: Path,
    region_boxes: dict[str, list[float]],
    selected_variables: list[str],
    selected_weather_states: list[str],
    sample_index: int,
    ensemble_member_index: int,
    generated: list[str],
) -> Path:
    manifest_path = out_root / "manifest.json"
    payload = {
        "suite_kind": suite_kind,
        "plot_style": "region_six_panel",
        "predictions_file": str(predictions_path),
        "out_dir": str(out_root),
        "regions": [
            {"name": name, "box": [float(v) for v in box]}
            for name, box in region_boxes.items()
        ],
        "model_variables": list(selected_variables),
        "weather_states": list(selected_weather_states),
        "sample_index": int(sample_index),
        "ensemble_member_index": int(ensemble_member_index),
        "generated_files": list(generated),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _region_boxes_for_names(region_names: list[str] | None, *, grid: str) -> dict[str, list[float]]:
    if region_names:
        unknown = [name for name in region_names if name not in PREDICTION_REGION_BOXES]
        if unknown:
            raise ValueError(f"Unknown region names: {unknown}. Known regions: {sorted(PREDICTION_REGION_BOXES)}")
        return {name: PREDICTION_REGION_BOXES[name] for name in region_names}

    if grid == "O1280":
        return O1280_INTERESTING_REGIONS

    return {"amazon_forest": [-15.0, 5.0, -75.0, -45.0]}


def render_region_suite_from_predictions_file(
    *,
    predictions_nc: str | Path,
    out_dir: str | Path,
    region_names: list[str] | None = None,
    model_variables: list[str] | None = None,
    weather_states: list[str] | None = None,
    sample_index: int = 0,
    ensemble_member_index: int = 0,
    also_png: bool = True,
    suite_kind: str = "regions",
) -> list[str]:
    predictions_path = Path(predictions_nc).expanduser().resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    combined_pdf = out_root / "all_regions_plots.pdf"
    generated: list[str] = [str(combined_pdf)]

    suite_kind = str(suite_kind).strip().lower() or "regions"
    if suite_kind not in {"regions", "storm"}:
        raise ValueError(f"Unsupported suite_kind={suite_kind!r}; expected 'regions' or 'storm'.")

    requested_weather_states = weather_states or DEFAULT_WEATHER_STATES
    with xr.open_dataset(predictions_path) as ds_pred:
        region_boxes = _region_boxes_for_names(
            region_names,
            grid=str(ds_pred.attrs.get("grid", "")).strip(),
        )

        if "sample" in ds_pred.dims:
            if not 0 <= sample_index < int(ds_pred.sizes["sample"]):
                raise IndexError(f"sample_index={sample_index} outside 0..{int(ds_pred.sizes['sample']) - 1}")
            ds_pred = ds_pred.isel(sample=sample_index)
        if "ensemble_member" in ds_pred.dims:
            if not 0 <= ensemble_member_index < int(ds_pred.sizes["ensemble_member"]):
                raise IndexError(
                    "ensemble_member_index="
                    f"{ensemble_member_index} outside 0..{int(ds_pred.sizes['ensemble_member']) - 1}"
                )
            ds_pred = ds_pred.isel(ensemble_member=ensemble_member_index)
        ds_pred = ensure_x_interp_for_plotting(ds_pred, predictions_path=predictions_path)

        selected_variables, selected_weather_states = _select_prediction_variables(
            ds_pred,
            model_variables=model_variables,
        )
        if weather_states:
            selected_weather_states = [w for w in requested_weather_states if w in selected_weather_states]
            if not selected_weather_states:
                selected_weather_states = requested_weather_states

        with PdfPages(combined_pdf) as pdf:
            for region_name, region_box in region_boxes.items():
                LOG.info("Plotting prediction region %s %s", region_name, region_box)
                region_ds = get_region_ds(ds_pred, region_box)
                title = _sample_meta_title(region_ds, region_name, sample_index)
                fig = plot_x_y(
                    ds_sample=region_ds,
                    list_model_variables=selected_variables,
                    weather_states=selected_weather_states,
                    title=title,
                )
                pdf.savefig(fig)
                region_pdf = out_root / f"{region_name}.pdf"
                fig.savefig(region_pdf, dpi=220)
                generated.append(str(region_pdf))
                if also_png:
                    region_png = out_root / f"{region_name}.png"
                    fig.savefig(region_png, dpi=220)
                    generated.append(str(region_png))
                plt.close(fig)

        manifest_path = _write_suite_manifest(
            out_root=out_root,
            suite_kind=suite_kind,
            predictions_path=predictions_path,
            region_boxes=region_boxes,
            selected_variables=selected_variables,
            selected_weather_states=selected_weather_states,
            sample_index=sample_index,
            ensemble_member_index=ensemble_member_index,
            generated=generated,
        )
        generated.append(str(manifest_path))

    return generated


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
        ds_pred = ensure_x_interp_for_plotting(ds_pred, predictions_path=predictions_path)
        selected_variables, selected_weather_states = _select_prediction_variables(
            ds_pred,
            model_variables=model_variables,
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
    lip.save_plot(
        lip.regions,
        list_model_variables=selected_variables,
        weather_states=selected_weather_states,
    )


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
    parser = argparse.ArgumentParser(description="Build predictions from expver or render local region plots from predictions.")
    parser.add_argument("--expver", type=str, default="", help="Experiment version")
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
    parser.add_argument(
        "--predictions-nc",
        type=str,
        default="",
        help="Existing predictions_*.nc file to render directly.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for direct predictions rendering mode.",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="",
        help="Comma-separated region names for direct predictions rendering mode. Blank => default suite.",
    )
    parser.add_argument(
        "--model-variables",
        type=str,
        default=",".join(DEFAULT_MODEL_VARIABLES),
        help="Comma-separated model variables for direct predictions rendering mode.",
    )
    parser.add_argument(
        "--weather-states",
        type=str,
        default=",".join(DEFAULT_WEATHER_STATES),
        help="Comma-separated weather states for direct predictions rendering mode.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ensemble-member-index", type=int, default=0)
    parser.add_argument("--also-png", action="store_true")
    parser.add_argument("--suite-kind", default="regions", choices=["regions", "storm"])

    args = parser.parse_args()
    if args.predictions_nc:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --predictions-nc")
        generated = render_region_suite_from_predictions_file(
            predictions_nc=args.predictions_nc,
            out_dir=args.out_dir,
            region_names=[v.strip() for v in args.regions.split(",") if v.strip()] or None,
            model_variables=[v.strip() for v in args.model_variables.split(",") if v.strip()],
            weather_states=[v.strip() for v in args.weather_states.split(",") if v.strip()],
            sample_index=args.sample_index,
            ensemble_member_index=args.ensemble_member_index,
            also_png=args.also_png,
            suite_kind=args.suite_kind,
        )
        for path in generated:
            print(path)
        return

    if not args.expver:
        raise SystemExit("Either --predictions-nc or --expver is required.")
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
