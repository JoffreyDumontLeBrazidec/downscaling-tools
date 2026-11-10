"""box_spectra on synthetic files: the ratios are 1 where the prediction is the truth, the
residual ratio is 0 where the prediction is the input, rows exist pooled and per date."""
from __future__ import annotations

import json

import netCDF4 as nc
import numpy as np
import pytest

from eval.evaluators import box_spectra
from eval.evaluators.box_spectra import runner


def _write(path, *, mode: str, members=2, seed=0):
    rng = np.random.default_rng(seed)
    lat1 = np.arange(35.0, 45.0 + 1e-9, 0.08)
    lon1 = np.arange(-30.0, -15.0 + 1e-9, 0.08)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    n = LA.size
    states = ["10u", "10v", "2t"]
    with nc.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", members)
        d.createDimension("grid_point_hres", n)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = LA.ravel()
        d.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = np.mod(LO.ravel(), 360.0)
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            ws[i] = s
        d.createVariable("ensemble_member", "i4", ("ensemble_member",))[:] = np.arange(1, members + 1)
        dims = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        for name in ("y_pred", "y", "x_interp"):
            d.createVariable(name, "f4", dims)
        for k in range(members):
            xi = rng.standard_normal((n, len(states))) * 0.2 + np.cos(LA.ravel() / 3)[:, None]
            y = xi + rng.standard_normal((n, len(states)))
            yp = {"truth": y, "input": xi, "noise": xi + rng.standard_normal((n, len(states)))}[mode]
            d["x_interp"][0, k, :, :] = xi
            d["y"][0, k, :, :] = y
            d["y_pred"][0, k, :, :] = yp


@pytest.mark.parametrize("mode", ["truth", "input"])
def test_ratios_on_known_cases(tmp_path, mode):
    pred = tmp_path / "pred"
    pred.mkdir()
    for i, date in enumerate(("20230226", "20230822")):
        _write(pred / f"predictions_{date}_step024.nc", mode=mode, seed=i)
    out = box_spectra.run(pred, {}, {"variables": ["10u", "ws", "2t", "msl"]}, output_dir=tmp_path / "bs")
    rows = runner.read_bins(out / "bins.csv")
    assert {r["scope"] for r in rows} == {"all", "20230226", "20230822"}
    assert {r["variable"] for r in rows} == {"10u", "ws", "2t"}   # msl absent: skipped
    pooled = [r for r in rows if r["scope"] == "all" and r["variable"] == "10u"]
    assert pooled[0]["n_draws"] == "4"
    assert float(pooled[0]["bin_centre_km"]) > float(pooled[-1]["bin_centre_km"])   # large scales first
    full = np.array([float(r["ratio_full"]) for r in pooled])
    resid = np.array([float(r["ratio_resid"]) for r in pooled])
    if mode == "truth":
        assert np.allclose(full, 1.0, atol=1e-5) and np.allclose(resid, 1.0, atol=1e-5)
        assert np.allclose([float(r["coherence"]) for r in pooled], 1.0, atol=1e-5)
    else:
        assert np.allclose(resid, 0.0, atol=1e-6)
        assert np.allclose(full, [float(r["ratio_interp_to_truth"]) for r in pooled], atol=1e-5)
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["box_chosen_by"] == "file bounds" and meta["method"] == "linear"
    assert (out / "bins.md").read_text().count("| km |") == 3
    written = box_spectra.plot(out, {}, {})
    assert written and all(p.exists() for p in written)


def test_named_box_and_global_default():
    lat = np.array([-80.0, 80.0])
    lon = np.array([0.0, 359.0])
    assert runner.resolve_box(None, lat, lon) == ([10.0, 70.0, -80.0, 40.0], "e2 (global file)")
    assert runner.resolve_box("north_atlantic", lat, lon)[0] == [35.0, 55.0, -45.0, -20.0]
    assert runner.resolve_box("36,44,-28,-17", lat, lon) == ([36.0, 44.0, -28.0, -17.0], "given")


def test_input_spectrum_from_native_grid(tmp_path):
    """Rule 41: a file with a native x (lat_lres/lon_lres) gives the input spectrum from x, not x_interp.
    Here x is the truth on a 0.16-degree grid and x_interp is pure noise, so input/truth tracks 1 at large
    scales only with the native field."""
    pred = tmp_path / "pred"
    pred.mkdir()
    path = pred / "predictions_20230226_step024.nc"
    _write(path, mode="truth", seed=1)
    lat1 = np.arange(35.0, 45.0 + 1e-9, 0.16)
    lon1 = np.arange(-30.0, -15.0 + 1e-9, 0.16)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    with nc.Dataset(str(path), "a") as d:
        d.createDimension("grid_point_lres", LA.size)
        d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = LA.ravel()
        d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = np.mod(LO.ravel(), 360.0)
        x = d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))
        sig = np.cos(LA.ravel() / 3)[:, None] * 50.0
        x[0] = np.stack([np.repeat(sig, 3, 1)] * 2)
        hres_la = np.asarray(d["lat_hres"][:])
        sig_h = np.cos(hres_la / 3)[:, None] * 50.0
        d["y"][0] = np.stack([np.repeat(sig_h, 3, 1)] * 2)
        d["y_pred"][0] = d["y"][0]
        d["x_interp"][0] = np.random.default_rng(0).standard_normal(d["x_interp"][0].shape)
    out = box_spectra.run(pred, {}, {"variables": ["2t"]}, output_dir=tmp_path / "bs")
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["input_source"] == ["x (native input grid)"]
    rows = [r for r in runner.read_bins(out / "bins.csv") if r["scope"] == "all"]
    assert abs(float(rows[0]["ratio_interp_to_truth"]) - 1.0) < 0.05   # largest scale: the native input is the truth
