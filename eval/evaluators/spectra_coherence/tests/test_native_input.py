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
