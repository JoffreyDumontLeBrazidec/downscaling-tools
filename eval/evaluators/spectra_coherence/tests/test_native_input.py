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



def _octahedral(N: int) -> tuple[np.ndarray, np.ndarray]:
    """The global octahedral O<N> grid: 4i + 16 points on row i from each pole."""
    lats, lons = [], []
    for i in range(1, N + 1):
        n, la = 4 * i + 16, 90.0 - (i - 0.5) * 90.0 / N
        lo = np.arange(n) * 360.0 / n
        lats += [np.full(n, la), np.full(n, -la)]
        lons += [lo, lo]
    return np.concatenate(lats), np.concatenate(lons)


def test_native_input_curves_of_a_known_filter(tmp_path: Path) -> None:
    """Global truth on O320 points; input = the same field times a known filter H(l), truncated at l = 79, on O80
    points. The input curves (own map, truth at the displaced positions) give amplitude ratio H(l) and coherence
    ~1 at the degrees kept (6 native grid lengths, the input's truncation), NaN beyond; the model residual of
    member k is taken against x_interp_members[k], the truth residual against x_interp."""
    import csv

    import healpy as hp
    import netCDF4

    np.random.seed(7)
    lt = 300
    ell = np.arange(lt + 1.0)
    cl = (ell + 10.0) ** -3.0
    cl[0] = 0.0
    alm_t = hp.synalm(cl, lmax=lt)
    h = np.exp(-(ell / 80.0) ** 4)
    h[80:] = 0.0
    mt, mi = hp.alm2map(alm_t, 512, lmax=lt), hp.alm2map(hp.almxfl(alm_t, h), 512, lmax=lt)
    hla, hlo = _octahedral(320)
    lla, llo = _octahedral(80)
    truth = hp.get_interp_val(mt, np.deg2rad(90.0 - hla), np.deg2rad(hlo))
    native = hp.get_interp_val(mi, np.deg2rad(90.0 - lla), np.deg2rad(llo))
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
    edges = [(5, 15), (15, 25), (25, 35), (35, 45)]
    bands = [{"name": f"b{lo}", "lo": lo, "hi": hi} for lo, hi in edges] + [{"name": "all", "lo": 1, "hi": 200}]
    out = runner.run(pred, {}, {"weather_states": ["2t"], "nside": 128, "lmax": 256, "bands": bands},
                     output_dir=tmp_path / "out")
    payload = json.loads((out / "coherence.json").read_text())
    im = payload["input_map"]
    assert payload["residual_reference"] == ["model: x_interp_members; truth: x_interp"]
    assert im["truncation"] == 79 and im["lmax_in"] == 79 and 40 <= im["nside_in"] <= 55
    c = payload["curves"]["2t"]
    ri = np.array(c["interp_amplitude_ratio"], dtype=float)
    ci = np.array(c["interp_coherence"], dtype=float)
    lk = int(im["l_keep_max"])
    assert 45 <= lk <= 79 and np.isnan(ri[lk + 1:]).all() and np.isfinite(ri[1:lk + 1]).all()
    assert np.isfinite(ci[1:lk + 1]).all()
    got = {b["band"]: b for b in payload["band_summary"]}
    for lo, hi in edges:   # band sums of power, as the band rows form them
        s_ = slice(lo, hi)
        expected = np.sqrt(np.sum(h[s_] ** 2 * cl[s_] * (2 * ell[s_] + 1)) / np.sum(cl[s_] * (2 * ell[s_] + 1)))
        b = got[f"b{lo}"]
        assert abs(b["interp_amplitude_ratio"] / expected - 1.0) < 0.03 and b["interp_coherence"] > 0.98, (lo, b)
    assert "interp_coherence" not in got["all"]   # 'all' reaches past the kept degrees
    with open(out / "curves.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["scope"] == "all"]
    p_rp = np.array([float(r["P_resid_pred"]) for r in rows])
    p_rt = np.array([float(r["P_resid_true"]) for r in rows])
    p_t = np.array([float(r["P_true"]) for r in rows])
    assert p_rp.max() < 1e-8 * p_t.max()          # y_pred - x_interp_members[k] = 0
    np.testing.assert_allclose(p_rt, p_t, rtol=1e-6)   # y - x_interp, with x_interp = 0 here
