from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from eval.tc import plot_pdf_tc as tc_plot_mod
from eval.tc import plot_pdf_tc_from_predictions as pred_plot_mod
from eval.tc import tc_vector_loading as loading_mod


def test_plot_pdf_tc_main_defaults_to_regridded(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_tc_pdf(**kwargs):
        captured.update(kwargs)
        return "unused"

    monkeypatch.setattr(tc_plot_mod, "run_tc_pdf", _fake_run_tc_pdf)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_pdf_tc.py", "--expver", "j2hh", "--outdir", "/tmp/tc-default"],
    )

    tc_plot_mod.main()

    assert captured["support_mode"] == "regridded"


def test_plot_pdf_tc_from_predictions_main_defaults_to_regridded(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_tc_pdf_from_predictions(**kwargs):
        captured.update(kwargs)
        return "unused"

    monkeypatch.setattr(pred_plot_mod, "run_tc_pdf_from_predictions", _fake_run_tc_pdf_from_predictions)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_pdf_tc_from_predictions.py",
            "--predictions-dir",
            "/tmp/preds",
            "--outdir",
            "/tmp/out",
            "--run-label",
            "demo",
        ],
    )

    pred_plot_mod.main()

    assert captured["support_mode"] == "regridded"


def test_plot_pdf_tc_from_predictions_main_passes_display_label(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_tc_pdf_from_predictions(**kwargs):
        captured.update(kwargs)
        return "unused"

    monkeypatch.setattr(pred_plot_mod, "run_tc_pdf_from_predictions", _fake_run_tc_pdf_from_predictions)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_pdf_tc_from_predictions.py",
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

    pred_plot_mod.main()

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

    curve = loading_mod._load_prediction_curve_regridded(
        [(pred_path, 20230827, 24)],
        target_lon=np.array([0.5], dtype=np.float64),
        target_lat=np.array([0.5], dtype=np.float64),
    )

    assert curve.msl.shape == (1,)
    assert curve.wind.shape == (1,)
    assert np.isclose(curve.msl[0], 1015.0)
    assert np.isclose(curve.wind[0], 15.0)
