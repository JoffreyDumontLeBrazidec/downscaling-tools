"""spread_proxy reads the input spread from the native input grid (rule 41)."""
from __future__ import annotations

import csv
import json

import numpy as np
import xarray as xr

from eval.evaluators.spread_proxy.core import compute_spread_proxy


def _write(path):
    n, nl, m = 60, 12, 3
    rng = np.random.default_rng(0)
    states = np.array(["2t", "msl"], dtype=object)
    base = np.zeros((1, m, n, 2), np.float32)
    base[..., 0] = 280.0
    base[..., 1] = 101000.0
    y = base + rng.standard_normal(base.shape).astype(np.float32)
    x = np.zeros((1, m, nl, 2), np.float32)
    x[..., 0] = 280.0 + 3.0 * rng.standard_normal((1, m, nl))      # native spread SD 3
    x[..., 1] = 101000.0
    xr.Dataset({
        "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
        "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y[:, ::-1].copy()),
        "x_interp": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), base),  # no spread
        "x": (("sample", "ensemble_member", "grid_point_lres", "weather_state"), x),
        "lat_hres": (("grid_point_hres",), np.linspace(30, 60, n)),
        "lon_hres": (("grid_point_hres",), np.linspace(-10, 20, n)),
        "lat_lres": (("grid_point_lres",), np.linspace(30, 60, nl)),
        "lon_lres": (("grid_point_lres",), np.linspace(-10, 20, nl)),
    }, coords={"weather_state": states}).to_netcdf(path)


def _spread_input(out):
    with open(out / "spread_by_lead.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("metric") == "spread_input"]
    return rows


def test_input_spread_from_the_native_grid(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_NATIVE_INDEX_CACHE", str(tmp_path / "idx"))
    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / "predictions_20230101_step024.nc")
    kw = dict(weather_states=["2t"], domains=["global"], spectra=False)
    out = tmp_path / "nat"
    payload = compute_spread_proxy(pred, out, **kw)
    assert payload["input_source"] == ["x (native input grid)"]
    old = compute_spread_proxy(pred, tmp_path / "old", native_input=False, **kw)
    assert "interpolated" in old["input_source"][0]
    nat_rows, old_rows = _spread_input(out), _spread_input(tmp_path / "old")
    assert nat_rows and old_rows
    assert float(nat_rows[0]["value"]) > 1.0 and float(old_rows[0]["value"]) < 1e-6
