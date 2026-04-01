from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eval.tc import plot_pdf_tc as tc_plot_mod
from eval.tc import plot_pdf_tc_from_predictions as pred_plot_mod
from eval.tc import tc_events as events_mod
from eval.tc import tc_pdf_plot as tc_pdf_plot_mod
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


def test_analysis_row_indices_supports_with_and_without_leading_analysis_frame():
    assert loading_mod._analysis_row_indices(6, None) == slice(1, None)
    assert loading_mod._analysis_row_indices(6, [0, 2]) == [1, 3]
    assert loading_mod._analysis_row_indices(5, None) == slice(0, None)
    assert loading_mod._analysis_row_indices(5, [0, 2]) == [0, 2]


def test_humberto_plot_event_curves_uses_event_specific_oper_and_references():
    cfg = events_mod.EVENTS["humberto"]
    curves = {
        cfg.analysis: loading_mod.CurveVectors(
            msl=np.array([999.0, 1001.0, 1003.0, 1005.0]),
            wind=np.array([12.0, 14.0, 16.0, 18.0]),
        ),
        cfg.expid_enfo_o320: loading_mod.CurveVectors(
            msl=np.array([998.0, 1000.0, 1002.0, 1004.0]),
            wind=np.array([13.0, 15.0, 17.0, 19.0]),
        ),
        "demo-run": loading_mod.CurveVectors(
            msl=np.array([996.0, 998.0, 1000.0, 1002.0]),
            wind=np.array([10.0, 12.0, 14.0, 16.0]),
        ),
    }

    fig, stats = tc_pdf_plot_mod.plot_event_curves(
        cfg,
        curves=curves,
        curve_order=["demo-run", *cfg.reference_expids],
        exp_labels={"demo-run": "demo"},
        return_stats=True,
    )

    assert stats["curve_order"] == ["demo-run", *cfg.reference_expids]
    assert cfg.expid_enfo_o320 in stats["variables"]["mslp_hpa"]["curves"]
    plt.close(fig)
