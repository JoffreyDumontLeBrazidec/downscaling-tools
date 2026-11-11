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


def _octahedral_rows(n_lat: int, lat_lo: float, lat_hi: float):
    """Northern-hemisphere rows of the O<n_lat> octahedral grid (Gaussian latitudes, 4i + 20 points
    on row i from the pole) between lat_lo and lat_hi; lon in [0, 360)."""
    x, _ = np.polynomial.legendre.leggauss(2 * n_lat)
    glat = np.sort(np.degrees(np.arcsin(x)))[::-1][:n_lat]
    lats, lons = [], []
    for i, lat in enumerate(glat):
        if lat_lo <= lat <= lat_hi:
            n = 4 * i + 20
            lon = np.arange(n) * 360.0 / n
            lats.append(np.full(n, lat))
            lons.append(lon)
    return np.concatenate(lats), np.concatenate(lons)


def test_coarse_octahedral_grid_widens_the_pad():
    """On O320 the rows near 70 N are ~1.2 deg apart in longitude: a 0.5 deg pad leaves corner points
    of the E2 grid outside the hull (36 at 0.07 deg on the real o96 -> o320 files, 6 at 0.25 deg);
    the sampler widens the pad and samples a linear field exactly."""
    from scipy.spatial import Delaunay

    from eval.evaluators.box_spectra.core.spectrum import Sampler

    lat, lon = _octahedral_rows(320, 5.0, 75.0)
    lon_c = runner._contiguous(lon, -80.0)
    extent = (11.0, 69.0, -79.0, 39.0)
    res = 0.25
    gx, gy = np.arange(extent[2], extent[3] + 1e-9, res), np.arange(extent[0], extent[1] + 1e-9, res)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    m = (lat >= extent[0] - 0.5) & (lat <= extent[1] + 0.5) & (lon_c >= extent[2] - 0.5) & (lon_c <= extent[3] + 0.5)
    assert np.any(Delaunay(np.column_stack([lon_c[m], lat[m]])).find_simplex(pts) < 0)   # the old failure

    smp = Sampler(lat, lon_c, extent, res)
    assert smp.pad_deg > 0.5
    field = 0.01 * lon_c + 0.02 * lat            # linear: barycentric sampling is exact
    assert np.allclose(smp(field), 0.01 * GX + 0.02 * GY, atol=1e-9)


def test_dense_grid_keeps_the_half_degree_pad():
    from eval.evaluators.box_spectra.core.spectrum import Sampler

    lat1 = np.arange(35.0, 45.0 + 1e-9, 0.08)
    lon1 = np.arange(-30.0, -15.0 + 1e-9, 0.08)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    smp = Sampler(LA.ravel(), LO.ravel(), (36.0, 44.0, -29.0, -16.0), 0.07)
    assert smp.pad_deg == 0.5


def test_target_note_from_lane_block(tmp_path):
    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / "predictions_20230226_step024.nc", mode="truth")
    out = box_spectra.run(pred, {}, {"variables": ["2t"], "target_note": "target 1 from 400 km to 125 km"},
                          output_dir=tmp_path / "bs")
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["target_note"] == "target 1 from 400 km to 125 km" and meta["source_pad_deg"] == 0.5
    assert "target 1 from 400 km to 125 km." in (out / "bins.md").read_text()
    default = box_spectra.run(pred, {}, {"variables": ["2t"]}, output_dir=tmp_path / "bs0")
    assert runner.TARGET_NOTE in (default / "bins.md").read_text()


def _plane_waves(lat, lon):
    """Three east-west waves of 600, 250 and 120 km (amplitudes 3, 2, 1) in x = lon * 111.195 km * cos 40."""
    x = np.asarray(lon) * 111.195 * np.cos(np.radians(40.0))
    return (3.0 * np.cos(2 * np.pi * x / 600.0) + 2.0 * np.cos(2 * np.pi * x / 250.0 + 1.0)
            + 1.0 * np.cos(2 * np.pi * x / 120.0 + 2.0) + 0.0 * np.asarray(lat))


def test_native_input_spectrum_of_a_known_field(tmp_path):
    """The input spectrum on a grid at the input's own spacing keeps the amplitude of a field the native grid
    resolves (input/truth ~1 at 600, 250 and 120 km), where linear sampling onto the fine grid damps the 120 km
    wave; bins below 3 native grid lengths are NaN. The residual takes member k's own x_interp_members[k]."""
    from eval.evaluators.box_spectra.core.spectrum import Sampler, Spectrum

    pred = tmp_path / "pred"
    pred.mkdir()
    path = pred / "predictions_20230226_step024.nc"
    _write(path, mode="truth", seed=3)
    # native grid: 0.25 degree rows, every other row shifted by half a step (an unstructured point set)
    rows = []
    for i, la in enumerate(np.arange(34.0, 46.0 + 1e-9, 0.25)):
        lo = np.arange(-31.0, -14.0 + 1e-9, 0.25) + (0.125 if i % 2 else 0.0)
        rows.append(np.column_stack([np.full(lo.size, la), lo]))
    nat = np.concatenate(rows)
    with nc.Dataset(str(path), "a") as d:
        hla, hlo = np.asarray(d["lat_hres"][:]), (np.asarray(d["lon_hres"][:]) + 180.0) % 360.0 - 180.0
        truth = _plane_waves(hla, hlo)
        d["y"][0] = np.repeat(truth[None, :, None], 2, 0).repeat(3, 2)
        d["y_pred"][0] = d["y"][0]
        d.createDimension("grid_point_lres", nat.shape[0])
        d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = nat[:, 0]
        d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = np.mod(nat[:, 1], 360.0)
        x = d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))
        x[0] = np.repeat(_plane_waves(nat[:, 0], nat[:, 1])[None, :, None], 2, 0).repeat(3, 2)
        xm = d.createVariable("x_interp_members", "f4", ("sample", "ensemble_member", "grid_point_hres", "weather_state"))
        xm[0] = d["y"][0]   # member k's input = the truth: residual true = 0 everywhere
    out = box_spectra.run(pred, {}, {"variables": ["2t"]}, output_dir=tmp_path / "bs")
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["residual_reference"] == ["x_interp_members"]
    rows = [r for r in runner.read_bins(out / "bins.csv") if r["scope"] == "all"]
    wl = np.array([float(r["bin_centre_km"]) for r in rows])
    ratio = np.array([float(r["ratio_interp_to_truth"]) for r in rows])
    amp_t = np.array([float(r["amp_truth_full"]) for r in rows])
    assert np.allclose([float(r["amp_resid_true"]) for r in rows], 0.0, atol=1e-6)  # against x_interp_members
    lim = meta["input_spectrum"]["min_wavelength_km"]
    assert 50.0 < lim < 90.0 and np.isnan(ratio[wl < lim]).all() and np.isfinite(ratio[wl >= lim]).all()
    for lam in (600.0, 250.0, 120.0):  # the bin holding the wave: the truth's peak near its wavelength
        near = np.flatnonzero(np.abs(np.log(wl / lam)) < np.log(1.35))
        k = near[np.argmax(amp_t[near])]
        assert abs(ratio[k] - 1.0) < 0.06, (lam, wl[k], ratio[k])
    # the old operator: linear sampling of the native points onto the fine grid damps the 120 km wave
    ext = tuple(meta["extent_sampled"])
    fine = Sampler(hla, hlo, ext, 0.07)
    old = Sampler(nat[:, 0], nat[:, 1], ext, 0.07)
    spec = Spectrum(len(fine.gy), len(fine.gx), *fine.dx_dy_km())
    p_t = spec.power(spec.fft(fine(truth)))
    p_o = spec.power(spec.fft(old(_plane_waves(nat[:, 0], nat[:, 1]))))
    near = np.flatnonzero(np.abs(np.log(spec.wavelength_km / 120.0)) < np.log(1.35))
    k = near[np.argmax(p_t[near])]
    assert np.sqrt(p_o[k] / p_t[k]) < 0.95   # 0.93 here, against ~1 on the native-spacing grid
