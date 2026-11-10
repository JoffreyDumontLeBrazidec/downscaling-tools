from __future__ import annotations

import pytest

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eval.evaluators.tc.core import events as events_mod
from eval.evaluators.tc.core import workflows as workflows_mod
from eval.evaluators.tc.core import loading_grib as loading_grib_mod
from eval.evaluators.tc.core import loading_predictions as loading_pred_mod
from eval.evaluators.tc.core import data_types as data_types_mod
from eval.evaluators.tc.core import plot_config as plot_config_mod
from eval.evaluators.tc.core import experiment_config as exp_config_mod
from eval.evaluators.tc import runner as tc_runner_mod


@pytest.fixture(autouse=True)
def _no_box_edge_strip(monkeypatch):
    """Toy grids of a few points have no interior: the box-edge strip (eval.shared.edge_strip) is off here."""
    monkeypatch.setenv("EVAL_BOX_EDGE_DEG", "0")



def test_workflows_pdf_main_defaults_to_regridded(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_tc_pdf(**kwargs):
        captured.update(kwargs)
        return "unused"

    monkeypatch.setattr(workflows_mod, "run_tc_pdf", _fake_run_tc_pdf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflows.py",
            "pdf",
            "--predictions-dir",
            "/tmp/preds",
            "--outdir",
            "/tmp/out",
            "--run-label",
            "demo",
        ],
    )

    workflows_mod.main()

    assert captured["support_mode"] == "regridded"


def test_bundle_curves_share_regridded_analysis_support(monkeypatch):
    event = events_mod.EVENTS["franklin"]
    captured: dict[str, object] = {}

    def _fake_target_points(bbox, resolution, sample_path):
        captured.update({"bbox": bbox, "resolution": resolution, "sample_path": sample_path})
        return np.array([1.0]), np.array([2.0])

    monkeypatch.setattr(tc_runner_mod, "regridded_target_points", _fake_target_points)
    kwargs = tc_runner_mod._bundle_curve_kwargs(
        mode="regridded",
        event=event,
        grib_dir="/reference/tc",
        analysis_expid="OPER_O1280_0001",
        regrid_resolution=0.25,
    )

    assert kwargs["support_mode"] == "regridded"
    np.testing.assert_array_equal(kwargs["target_lon"], np.array([1.0]))
    np.testing.assert_array_equal(kwargs["target_lat"], np.array([2.0]))
    assert captured["sample_path"].endswith(
        f"franklin/surface_an_OPER_O1280_0001_{event.analysis_dates[0]}.grib"
    )

    native = tc_runner_mod._bundle_curve_kwargs(
        mode="native",
        event=event,
        grib_dir=None,
        analysis_expid=None,
        regrid_resolution=0.25,
    )
    assert native == {"support_mode": "native", "target_lon": None, "target_lat": None}


def test_workflows_pdf_main_passes_display_label(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_tc_pdf(**kwargs):
        captured.update(kwargs)
        return "unused"

    monkeypatch.setattr(workflows_mod, "run_tc_pdf", _fake_run_tc_pdf)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflows.py",
            "pdf",
            "--predictions-dir",
            "/tmp/preds",
            "--outdir",
            "/tmp/out",
            "--run-label",
            "manual_long_id",
            "--display-label",
            "short label",
        ],
    )

    workflows_mod.main()

    assert captured["display_label"] == "short label"


def test_regridded_prediction_curve_interpolates_gridded_source(tmp_path: Path):
    pred_path = tmp_path / "predictions_20230827_step024.nc"
    ds = xr.Dataset(
        data_vars={
            "y_pred": (
                ("sample", "ensemble_member", "lat_hres", "lon_hres", "weather_state"),
                np.array(
                    [
                        [
                            [
                                [[100000.0, 0.0, 0.0], [101000.0, 10.0, 0.0]],
                                [[102000.0, 20.0, 0.0], [103000.0, 30.0, 0.0]],
                            ]
                        ]
                    ],
                    dtype=np.float32,
                ),
            )
        },
        coords={
            "sample": [0],
            "ensemble_member": [0],
            "lat_hres": np.array([0.0, 1.0], dtype=np.float32),
            "lon_hres": np.array([0.0, 1.0], dtype=np.float32),
            "weather_state": np.array(["msl", "10u", "10v"], dtype=object),
        },
    )
    ds["y_pred"].attrs["lat"] = "lat_hres"
    ds["y_pred"].attrs["lon"] = "lon_hres"
    ds.to_netcdf(pred_path)

    curve = loading_pred_mod._load_prediction_curve_regridded(
        [(pred_path, 20230827, 24)],
        target_lon=np.array([0.5], dtype=np.float64),
        target_lat=np.array([0.5], dtype=np.float64),
    )

    assert curve.msl.shape == (1,)
    assert curve.wind.shape == (1,)
    assert np.isclose(curve.msl[0], 1015.0)
    assert np.isclose(curve.wind[0], 15.0)


def test_analysis_row_indices_supports_with_and_without_leading_analysis_frame():
    assert loading_grib_mod._analysis_row_indices(6, None) == slice(1, None)
    assert loading_grib_mod._analysis_row_indices(6, [0, 2]) == [1, 3]
    assert loading_grib_mod._analysis_row_indices(5, None) == slice(0, None)
    assert loading_grib_mod._analysis_row_indices(5, [0, 2]) == [0, 2]


def test_humberto_compute_event_stats_uses_runtime_oper_and_references():
    event = events_mod.EVENTS["humberto"]
    exp_cfg = exp_config_mod.TCExperimentConfig(
        analysis_expid="OPER_O96_0001",
        reference_expids=("ENFO_O48_0001",),
    )
    plot_cfg = plot_config_mod.PLOT_CONFIGS["humberto"]
    curves = {
        exp_cfg.analysis_expid: data_types_mod.CurveVectors(
            msl=np.array([999.0, 1001.0, 1003.0, 1005.0]),
            wind=np.array([12.0, 14.0, 16.0, 18.0]),
        ),
        exp_cfg.reference_expids[0]: data_types_mod.CurveVectors(
            msl=np.array([998.0, 1000.0, 1002.0, 1004.0]),
            wind=np.array([13.0, 15.0, 17.0, 19.0]),
        ),
        "demo-run": data_types_mod.CurveVectors(
            msl=np.array([996.0, 998.0, 1000.0, 1002.0]),
            wind=np.array([10.0, 12.0, 14.0, 16.0]),
        ),
    }

    stats = workflows_mod.compute_event_stats(
        curves,
        analysis_key=exp_cfg.analysis_expid,
        plot_config=plot_cfg,
        curve_order=["demo-run", *exp_cfg.reference_expids],
    )

    assert stats["curve_order"] == ["demo-run", *exp_cfg.reference_expids]
    assert exp_cfg.reference_expids[0] in stats["variables"]["mslp_hpa"]["curves"]


def test_crop_native_dataset_applies_the_event_bbox_to_grib_values():
    ds = xr.Dataset(
        {"msl": ("values", np.array([100000.0, 101000.0, 102000.0]))},
        coords={
            "longitude": ("values", np.array([-95.0, -75.0, -95.0])),
            "latitude": ("values", np.array([20.0, 20.0, 45.0])),
        },
    )
    bbox = events_mod.EVENTS["idalia"].bbox  # single source of truth: events YAML

    cropped = loading_grib_mod._crop_native_dataset(ds, bbox)

    assert cropped.sizes["values"] == 1
    assert cropped["msl"].values.tolist() == [100000.0]


def test_native_per_date_analysis_keeps_each_verification_date(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_load_native_curve(files, **kwargs):
        calls.append(tuple(files))
        value = float(len(calls))
        return data_types_mod.CurveVectors(msl=np.array([value]), wind=np.array([value]))

    monkeypatch.setattr(loading_grib_mod, "_is_per_date_an", lambda path: True)
    monkeypatch.setattr(
        loading_grib_mod, "_expand_analysis_files",
        lambda *args, **kwargs: ["analysis-27.grib", "analysis-28.grib"],
    )
    monkeypatch.setattr(loading_grib_mod, "_load_native_curve", fake_load_native_curve)

    curves = loading_grib_mod.load_grib_curves(
        dir_data_base="/tmp/tc",
        event_name="idalia",
        analysis_expid="OPER_O320_0001",
        analysis_dates=["20230826"],
        forecast_dates=[],
        support_mode="native",
        bbox=events_mod.EVENTS["idalia"].bbox,
    )

    assert calls == [("analysis-27.grib",), ("analysis-28.grib",)]
    assert curves["OPER_O320_0001"].msl.tolist() == [1.0, 2.0]


def _native_input_file(path: Path, *, with_x: bool = True) -> None:
    """Output points every 0.5 deg in 10-20 N, 60-70 W; native input every 2 deg; msl 1000 hPa + native index."""
    lat_h, lon_h = np.meshgrid(np.arange(10.0, 20.01, 0.5), np.arange(-70.0, -59.99, 0.5), indexing="ij")
    lat_h, lon_h = lat_h.ravel(), lon_h.ravel()
    lat_l, lon_l = np.meshgrid(np.arange(9.0, 21.01, 2.0), np.arange(-71.0, -58.99, 2.0), indexing="ij")
    lat_l, lon_l = lat_l.ravel(), lon_l.ravel()
    n, nl = lat_h.size, lat_l.size
    hres = np.zeros((1, 2, n, 3), np.float32)
    hres[..., 0] = 101000.0
    data = {
        "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), hres),
        "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), hres),
        "x_interp": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), hres + 50.0),
        "lat_hres": (("grid_point_hres",), lat_h), "lon_hres": (("grid_point_hres",), lon_h),
    }
    if with_x:
        x = np.zeros((1, 2, nl, 3), np.float32)
        x[..., 0] = 100000.0 + 100.0 * np.arange(nl)
        x[..., 1] = 3.0
        x[..., 2] = 4.0
        data.update({"x": (("sample", "ensemble_member", "grid_point_lres", "weather_state"), x),
                     "lat_lres": (("grid_point_lres",), lat_l), "lon_lres": (("grid_point_lres",), lon_l)})
    xr.Dataset(data, coords={"weather_state": np.array(["msl", "10u", "10v"], dtype=object)}).to_netcdf(path)


def test_input_curve_from_the_native_grid(tmp_path: Path, monkeypatch):
    """Rule 41: the TC input curve is the native x at the nearest native point of each curve point, on the same
    support as the model curve (native and regridded); without x it is x_interp and the source says so."""
    from eval.shared.native_input import nearest_native_index

    monkeypatch.setenv("EVAL_NATIVE_INDEX_CACHE", str(tmp_path / "idx"))
    path = tmp_path / "predictions_20230827_step024.nc"
    _native_input_file(path)
    files = [(path, 20230827, 24)]
    bbox = data_types_mod.BoundingBox(north=18.0, south=12.0, west=-68.0, east=-62.0)
    model = loading_pred_mod.load_prediction_curves(files, bbox=bbox, support_mode="native")
    curve, src = loading_pred_mod.load_input_curves(files, bbox=bbox, support_mode="native")
    assert src == "x (native input grid)" and curve.support_signature == model.support_signature
    assert curve.msl.shape == model.msl.shape
    assert np.isclose(np.mean(curve.wind), 5.0) and set(np.unique(curve.msl)) <= {1000.0 + k for k in range(49)}
    tlon, tlat = np.meshgrid(np.arange(-68.0, -61.9, 0.25), np.arange(12.0, 18.01, 0.25))
    rg, src_rg = loading_pred_mod.load_input_curves(files, bbox=bbox, support_mode="regridded",
                                                    target_lon=tlon.ravel(), target_lat=tlat.ravel())
    idx = nearest_native_index(xr.open_dataset(path)["lat_lres"].values, xr.open_dataset(path)["lon_lres"].values,
                               tlat.ravel(), tlon.ravel())
    np.testing.assert_allclose(rg.msl[: tlat.size], 1000.0 + idx)          # member 0: native values, no blending
    assert rg.support_mode == "regridded" and src_rg == src
    old, src_old = loading_pred_mod.load_input_curves(files, bbox=bbox, support_mode="native", native_input=False)
    assert "interpolated" in src_old and np.allclose(old.msl, 1010.5)
    # (the member maps that also read x were retired on main, 30 Sep: load_prediction_member_fields is gone)
    path2 = tmp_path / "predictions_20230828_step024.nc"
    _native_input_file(path2, with_x=False)
    _, src2 = loading_pred_mod.load_input_curves([(path2, 20230828, 24)], bbox=bbox, support_mode="native")
    assert src2 == "x_interp (input interpolated to the output grid; no native x in the file)"
