"""Rule 41 (owner, 9 Nov 2026): the input curves of spectra_coherence come from the native x on its own grid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("healpy")

from eval.evaluators.spectra_coherence import runner  # noqa: E402
from eval.evaluators.spectra_coherence._input_sampler import SphereLinearSampler  # noqa: E402


def _fibonacci(n: int) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * i / n))
    lon = np.mod(np.degrees(np.pi * (1.0 + 5.0**0.5) * i), 360.0)
    return lat, lon


def _smooth(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.cos(la) * np.cos(lo) + 0.5 * np.sin(la) + 0.3 * np.cos(la) ** 2 * np.cos(2 * lo)


def test_sphere_sampler_is_linear_and_global() -> None:
    slat, slon = _fibonacci(4000)
    tlat, tlon = _fibonacci(997)
    tlat, tlon = tlat * 0.999, np.mod(tlon + 1.234, 360.0)
    s = SphereLinearSampler(slat, slon, tlat, tlon)
    assert s.n_nearest == 0
    np.testing.assert_allclose(s.w.sum(axis=1), 1.0)
    assert (s.w >= -1e-9).all()
    err = np.abs(s(_smooth(slat, slon)) - _smooth(tlat, tlon))
    assert err.max() < 0.02            # linear interpolation of a smooth field on a ~3.6 degree mesh


def test_sphere_sampler_on_a_regional_input() -> None:
    """A box of native points: targets inside are interpolated, targets outside take the nearest point."""
    la, lo = np.meshgrid(np.arange(30.0, 60.01, 1.0), np.arange(-20.0, 20.01, 1.0), indexing="ij")
    tin_la, tin_lo = np.array([40.3, 51.7]), np.array([-3.3, 12.2])
    s = SphereLinearSampler(la.ravel(), lo.ravel(), np.r_[tin_la, -40.0], np.r_[tin_lo, 100.0])
    assert s.n_nearest == 1
    got = s(_smooth(la.ravel(), lo.ravel()))
    np.testing.assert_allclose(got[:2], _smooth(tin_la, tin_lo), atol=5e-3)


def _write(path: Path, *, native: bool) -> None:
    """Truth on a 2-degree grid; x_interp is noise; the native x is the truth's field on a 6-degree grid."""
    import netCDF4

    rng = np.random.default_rng(0)
    hla, hlo = np.meshgrid(np.arange(-89.0, 89.01, 2.0), np.arange(0.0, 359.9, 2.0), indexing="ij")
    hla, hlo = hla.ravel(), hlo.ravel()
    lla, llo = np.meshgrid(np.arange(-87.0, 87.01, 6.0), np.arange(0.0, 359.9, 6.0), indexing="ij")
    lla, llo = lla.ravel(), llo.ravel()
    n_m = 2
    with netCDF4.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", n_m)
        d.createDimension("grid_point", hla.size)
        d.createDimension("weather_state", 1)
        d.createVariable("lat_hres", "f8", ("grid_point",))[:] = hla
        d.createVariable("lon_hres", "f8", ("grid_point",))[:] = hlo
        v = d.createVariable("weather_state", str, ("weather_state",))
        v[0] = "2t"
        truth = _smooth(hla, hlo)
        d.createVariable("y", "f4", ("sample", "grid_point", "weather_state"))[:] = truth[None, :, None]
        d.createVariable("y_pred", "f4", ("sample", "ensemble_member", "grid_point", "weather_state"))[:] = \
            np.stack([truth] * n_m)[None, :, :, None]
        d.createVariable("x_interp", "f4", ("sample", "grid_point", "weather_state"))[:] = \
            rng.standard_normal((1, hla.size, 1))
        if native:
            d.createDimension("grid_point_lres", lla.size)
            d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = lla
            d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = llo
            d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))[:] = \
                np.stack([_smooth(lla, llo)] * n_m)[None, :, :, None]


@pytest.mark.parametrize("native", [True, False])
def test_input_curves_come_from_the_native_x(tmp_path: Path, native: bool) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / "predictions_20230826_step024.nc", native=native)
    out = runner.run(pred, {}, {"weather_states": ["2t"], "nside": 8, "lmax": 16,
                                "bands": [{"name": "large", "lo": 1, "hi": 4}]}, output_dir=tmp_path / "out")
    payload = json.loads((out / "coherence.json").read_text())
    row = payload["band_summary"][0]
    if native:
        assert payload["input_source"] == [runner.INPUT_NATIVE]
        assert row["interp_coherence"] > 0.99          # the native input carries the truth's large scales
    else:
        assert payload["input_source"] == [runner.INPUT_INTERP]
        assert abs(row["interp_coherence"]) < 0.5      # the x_interp here is noise


def test_native_input_false_keeps_x_interp(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / "predictions_20230826_step024.nc", native=True)
    out = runner.run(pred, {}, {"weather_states": ["2t"], "nside": 8, "lmax": 16, "native_input": False},
                     output_dir=tmp_path / "out")
    assert json.loads((out / "coherence.json").read_text())["input_source"] == [runner.INPUT_INTERP_OFF]


def test_stratified_input_columns_come_from_the_native_x(tmp_path: Path, monkeypatch) -> None:
    from eval.evaluators.spectra_coherence import stratified

    def one_class(lat, lon, nside):
        npix = 12 * nside * nside
        return {"all": np.ones(npix, dtype=bool)}, {"all": {"pixel_fraction": 1.0}}

    monkeypatch.setattr(stratified, "build_surface_classes", one_class)
    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / "predictions_20230826_step024.nc", native=True)
    out = stratified.run_stratified(pred, output_dir=tmp_path / "out", states=["2t"], nside=8, lmax=16,
                                    steps=None, bands=[("large", 1, 4)])
    d = json.loads((out / "coherence_by_surface.json").read_text())
    assert d["input_source"] == [runner.INPUT_NATIVE]
    assert d["rows"][0]["interp_correlation"] > 0.99


def _harmonic_field(lat, lon, degrees=(20, 60), seed=0) -> np.ndarray:
    """A field with power at the given spherical-harmonic degrees only, evaluated at the points."""
    import healpy as hp

    lmax = max(degrees)
    rng = np.random.default_rng(seed)
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    for ell in degrees:
        for m in range(ell + 1):
            alm[hp.Alm.getidx(lmax, ell, m)] = rng.standard_normal() + (1j * rng.standard_normal() if m else 0.0)
    fine = hp.alm2map(alm, 512, lmax=lmax)
    return hp.get_interp_val(fine, np.deg2rad(90.0 - lat), np.deg2rad(np.mod(lon, 360.0)))


def test_native_input_curves_keep_a_known_spectrum(tmp_path: Path) -> None:
    """The native input on its own HEALPix map (one pixel per native point, nearest point) keeps the amplitude
    of the degrees it resolves (input/truth ~1 at l = 20 and 60), NaN beyond 3 native grid lengths; the
    residual of member k is taken against x_interp_members[k]."""
    import netCDF4

    hla, hlo = _fibonacci(200_000)
    lla, llo = _fibonacci(20_000)
    truth, native = _harmonic_field(hla, hlo), _harmonic_field(lla, llo)
    pred = tmp_path / "pred"
    pred.mkdir()
    n_m = 2
    with netCDF4.Dataset(str(pred / "predictions_20230826_step024.nc"), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", n_m)
        d.createDimension("grid_point", hla.size)
        d.createDimension("grid_point_lres", lla.size)
        d.createDimension("weather_state", 1)
        d.createVariable("lat_hres", "f8", ("grid_point",))[:] = hla
        d.createVariable("lon_hres", "f8", ("grid_point",))[:] = hlo
        d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = lla
        d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = llo
        v = d.createVariable("weather_state", str, ("weather_state",))
        v[0] = "2t"
        full = ("sample", "ensemble_member", "grid_point", "weather_state")
        d.createVariable("y", "f4", ("sample", "grid_point", "weather_state"))[:] = truth[None, :, None]
        d.createVariable("y_pred", "f4", full)[:] = np.stack([truth] * n_m)[None, :, :, None]
        d.createVariable("x_interp", "f4", ("sample", "grid_point", "weather_state"))[:] = 0.0
        d.createVariable("x_interp_members", "f4", full)[:] = np.stack([truth] * n_m)[None, :, :, None]
        d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))[:] = \
            np.stack([native] * n_m)[None, :, :, None]
    out = runner.run(pred, {}, {"weather_states": ["2t"], "nside": 64, "lmax": 128,
                                "bands": [{"name": "low", "lo": 15, "hi": 25}, {"name": "all", "lo": 1, "hi": 128}]},
                     output_dir=tmp_path / "out")
    payload = json.loads((out / "coherence.json").read_text())
    im = payload["input_map"]
    assert payload["residual_reference"] == ["x_interp_members"]
    assert 35 <= im["nside_in"] <= 45 and im["lmax_in"] == min(2 * im["nside_in"], 128)
    c = payload["curves"]["2t"]
    ri = np.array(c["interp_amplitude_ratio"], dtype=float)
    ci = np.array(c["interp_coherence"], dtype=float)
    for ell in (20, 60):
        assert abs(ri[ell] - 1.0) < 0.08 and ci[ell] > 0.95, (ell, ri[ell], ci[ell])
    ell = np.arange(ri.size)
    wl = 2 * np.pi * runner.EARTH_RADIUS_KM / np.sqrt(np.maximum(ell * (ell + 1.0), 1e-9))
    assert np.isnan(ri[(wl < im["min_wavelength_km"]) & (ell > 0)]).all() and np.isnan(ri[im["lmax_in"] + 1:]).all()
    bands = {b["band"]: b for b in payload["band_summary"]}
    assert "interp_coherence" in bands["low"] and "interp_coherence" not in bands["all"]   # 'all' reaches past
    import csv
    with open(out / "curves.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["scope"] == "all"]
    p_rt = np.array([float(r["P_resid_true"]) for r in rows])
    p_t = np.array([float(r["P_true"]) for r in rows])
    assert p_rt.max() < 1e-8 * p_t.max()   # y - x_interp_members[k] = 0 (against x_interp, all zero, it is y)
