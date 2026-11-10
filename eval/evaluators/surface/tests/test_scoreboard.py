"""Tests for surface scoring module."""

from __future__ import annotations

import json

import pytest

from eval.evaluators.surface.core.scoreboard import (
    format_surface_loss_for_scoreboard,
    load_surface_loss_metrics,
    surface_weighted_nmse,
)


def test_surface_weighted_nmse_with_truth_std():
    variables = {
        "msl": {"mean_mse": 10.0, "normalized_weight": 0.6},
        "sp": {"mean_mse": 5.0, "normalized_weight": 0.3},
        "2t": {"mean_mse": 2.0, "normalized_weight": 0.1},
    }
    truth_std = {"msl": 10.0, "sp": 5.0, "2t": 2.0}

    result, per_var = surface_weighted_nmse(variables, truth_std_by_variable=truth_std)

    assert result is not None
    assert result == pytest.approx(0.6 * 0.1 + 0.3 * 0.2 + 0.1 * 0.5)
    assert per_var["msl"] == pytest.approx(0.1)
    assert per_var["sp"] == pytest.approx(0.2)
    assert per_var["2t"] == pytest.approx(0.5)


def test_surface_weighted_nmse_with_existing_nmse():
    variables = {
        "msl": {"mean_nmse": 0.1, "normalized_weight": 0.6},
        "sp": {"mean_nmse": 0.2, "normalized_weight": 0.4},
    }
    result, per_var = surface_weighted_nmse(variables)
    assert result == pytest.approx(0.6 * 0.1 + 0.4 * 0.2)


def test_surface_weighted_nmse_empty():
    result, per_var = surface_weighted_nmse({})
    assert result is None
    assert per_var == {}


def test_load_surface_loss_metrics(tmp_path):
    summary = tmp_path / "surface_loss_summary.json"
    summary.write_text(json.dumps({
        "weighted_surface_mse": 10.0,
        "variables": {
            "msl": {"mean_mse": 10.0, "normalized_weight": 0.6},
            "sp": {"mean_mse": 5.0, "normalized_weight": 0.3},
            "2t": {"mean_mse": 2.0, "normalized_weight": 0.1},
        },
    }))

    result = load_surface_loss_metrics(
        summary,
        truth_std_by_variable={"msl": 10.0, "sp": 5.0, "2t": 2.0},
    )

    assert result["weighted_mse"] == pytest.approx(10.0)
    assert result["weighted_nmse"] == pytest.approx(0.6 * 0.1 + 0.3 * 0.2 + 0.1 * 0.5)
    assert [e["variable"] for e in result["top_contributors"]] == ["msl", "sp"]


def test_format_surface_loss_for_scoreboard():
    assert format_surface_loss_for_scoreboard({"weighted_nmse": 0.17}) == "0.1700"
    assert format_surface_loss_for_scoreboard({"weighted_mse": 1.23e-5}) == "1.230000e-05"
    assert format_surface_loss_for_scoreboard({}) == "na"


def test_input_surface_row_from_the_native_grid(tmp_path, monkeypatch):
    """Rule 41: the input surface MSE row scores the native x at the nearest native point, not x_interp."""
    import numpy as np
    import xarray as xr

    from eval.evaluators.surface.core.scoreboard import load_x_interp_surface_metrics

    monkeypatch.setenv("EVAL_NATIVE_INDEX_CACHE", str(tmp_path / "idx"))
    n, nl = 40, 8
    lat = np.linspace(-30, 30, n)
    states = np.array(["2t", "msl", "10u", "10v"], dtype=object)
    y = np.zeros((1, 1, n, 4), np.float32)
    y[..., 0] = 280.0
    y[..., 1] = 101000.0
    x = y[:, :, :nl].copy()
    x[..., 0] += 2.0                       # native input: 2 K too warm
    ds = xr.Dataset({
        "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
        "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
        "x_interp": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y + 1.0),
        "x": (("sample", "ensemble_member", "grid_point_lres", "weather_state"), x),
        "lat_hres": (("grid_point_hres",), lat), "lon_hres": (("grid_point_hres",), np.linspace(0, 40, n)),
        "lat_lres": (("grid_point_lres",), np.linspace(-30, 30, nl)),
        "lon_lres": (("grid_point_lres",), np.linspace(0, 40, nl)),
    }, coords={"weather_state": states})
    pred = tmp_path / "pred"
    pred.mkdir()
    ds.to_netcdf(pred / "predictions_20230101_step024.nc")
    m = load_x_interp_surface_metrics(pred)
    assert m["input_source"] == "x (native input grid)"
    assert m["variables"]["2t"]["mean_mse"] == pytest.approx(4.0, rel=1e-5)
    old = load_x_interp_surface_metrics(pred, native_input=False)
    assert "interpolated" in old["input_source"] and old["variables"]["2t"]["mean_mse"] == pytest.approx(1.0)
