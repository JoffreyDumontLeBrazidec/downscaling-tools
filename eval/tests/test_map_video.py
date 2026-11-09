"""eval.tools.map_video: Gaussian latitudes, ring-wise bilinear weights, crops, palettes, a tiny video."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from eval.tools.map_video.grid import (OctahedralGrid, RingBilinear, gaussian_latitudes,
                                       octahedral_n_from_size, regular_lat_lon)
from eval.tools.map_video.palette import TP_LEVELS_MM, tp_palette


@pytest.mark.parametrize("n", [4, 32, 160])
def test_gaussian_latitudes_match_leggauss(n):
    x, _ = np.polynomial.legendre.leggauss(2 * n)
    ref = np.degrees(np.arcsin(x))[::-1]            # north to south
    assert np.max(np.abs(gaussian_latitudes(n) - ref)) < 1e-10


def test_octahedral_sizes():
    assert octahedral_n_from_size(421_120) == 320
    assert octahedral_n_from_size(6_599_680) == 1280
    assert octahedral_n_from_size(26_306_560) == 2560
    assert octahedral_n_from_size(1000) is None
    assert OctahedralGrid.build(32).size == 4 * 32 * 41


def test_index_of_roundtrip_and_rejects_wrong_grid():
    g = OctahedralGrid.build(32)
    lat, lon = g.point_lat_lon()
    perm = np.random.default_rng(0).permutation(g.size)
    assert np.array_equal(g.index_of(lat[perm], lon[perm]), perm)
    assert np.array_equal(g.index_of(lat, lon - 360.0), np.arange(g.size))   # -180..180 longitudes
    lat48, lon48 = OctahedralGrid.build(48).point_lat_lon()
    with pytest.raises(ValueError):
        g.index_of(lat48, lon48)


def test_bilinear_exact_on_rings_and_linear_between():
    g = OctahedralGrid.build(32)
    lat, lon = g.point_lat_lon()
    # constant field: exact everywhere, including beyond the polar rings
    lat_t, lon_t = regular_lat_lon(2.0)
    interp = RingBilinear(g, lat_t, lon_t)
    np.testing.assert_allclose(interp(np.full(g.size, 3.5)), 3.5, rtol=0, atol=1e-6)
    # field linear in latitude: exact between the first and last ring
    out = interp(lat.astype(np.float32))
    inside = (lat_t <= g.lats[0]) & (lat_t >= g.lats[-1])
    np.testing.assert_allclose(out[inside], np.repeat(lat_t[inside, None], lon_t.size, 1), atol=1e-4)
    # target points that are grid points reproduce the grid value
    ring = 5
    j = np.arange(g.nlon[ring])
    lon_r = 360.0 * j / g.nlon[ring]
    vals = np.random.default_rng(1).normal(size=g.size).astype(np.float32)
    on_ring = RingBilinear(g, np.array([g.lats[ring]]), lon_r)(vals)[0]
    np.testing.assert_allclose(on_ring, vals[g.offsets[ring] + j], atol=1e-6)


def test_crop_matches_whole_grid_inside_and_is_nan_outside():
    g = OctahedralGrid.build(48)
    lat, lon = g.point_lat_lon()
    vals = (np.cos(np.radians(lat)) * np.sin(np.radians(lon))).astype(np.float32)
    keep = (lat > 20) & (lat < 60) & ((lon > 340) | (lon < 30))
    lat_t, lon_t = regular_lat_lon(1.0, extent=(-30, 40, 0, 80))
    whole = RingBilinear(g, lat_t, lon_t)(vals)
    crop = RingBilinear(g, lat_t, lon_t, source_index=g.index_of(lat[keep], lon[keep]))(vals[keep])
    ok = np.isfinite(crop)
    assert ok.any() and (~ok).any()
    np.testing.assert_allclose(crop[ok], whole[ok], atol=1e-6)
    lat2 = np.repeat(lat_t[:, None], lon_t.size, 1)
    assert not ok[(lat2 > 62) | (lat2 < 18)].any()


def test_regular_grid_shapes():
    lat, lon = regular_lat_lon(0.1)
    assert (lat.size, lon.size) == (1801, 3600) and lat[0] == 90 and lon[0] == -180


@pytest.mark.parametrize("style", ["rain", "reference"])
def test_tp_palette_classes(style):
    pal = tp_palette(style)
    levels = np.asarray(TP_LEVELS_MM)
    v = np.array([0.0, 0.099, 0.1, 0.3, 49.0, 50.0, 400.0, np.nan])
    rgb = pal.rgb(v)
    assert tuple(rgb[0]) == (255, 255, 255) and tuple(rgb[1]) == (255, 255, 255)
    assert tuple(rgb[2]) != (255, 255, 255)
    assert tuple(rgb[5]) == tuple(rgb[6]) == tuple(pal.lut[-1])     # 50 mm and above: over colour
    assert pal.lut.shape == (levels.size + 1, 3)
    cmap, norm, bounds = pal.colorbar_cmap_norm()
    assert cmap.N == levels.size - 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_cli_writes_video_from_prediction_files(tmp_path):
    xr = pytest.importorskip("xarray")
    from eval.tools.map_video.__main__ import main

    g = OctahedralGrid.build(32)
    lat, lon = g.point_lat_lon()
    for step in (6, 12):
        tp = np.maximum(np.sin(np.radians(lon) * 3 + step) * np.cos(np.radians(lat) * 2), 0) * 0.01
        arr = np.stack([tp, tp], -1)[None, None].astype(np.float32)
        ds = xr.Dataset(
            {"y_pred": (["sample", "ensemble_member", "grid_point_hres", "weather_state"], arr),
             "date": (["sample"], np.array(["2023-08-29"])),
             "lon_hres": (["grid_point_hres"], lon), "lat_hres": (["grid_point_hres"], lat)},
            coords={"weather_state": ["tp", "cp"], "ensemble_member": [1]})
        ds.to_netcdf(tmp_path / f"predictions_20230829_step{step:03d}.nc")
    out = tmp_path / "v.mp4"
    poster = tmp_path / "p.png"
    rc = main([str(p) for p in sorted(tmp_path.glob("*.nc"))]
              + ["--out", str(out), "--poster", str(poster), "--res", "1", "--no-geography", "--tween", "1"])
    assert rc == 0 and out.stat().st_size > 0 and poster.stat().st_size > 0
