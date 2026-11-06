"""Box-edge strip and the slot-head edge-artefact scan (docs ablation note s21)."""
from __future__ import annotations

import numpy as np
import pytest

from eval.shared.edge_strip import box_edge_mask, edge_distance_deg, is_global, scan_edge_artefact


def _grid(lat0=10.0, lat1=70.0, lon0=-80.0, lon1=40.0, step=0.5):
    la, lo = np.meshgrid(np.arange(lat0, lat1 + 1e-9, step), np.arange(lon0, lon1 + 1e-9, step), indexing="ij")
    return la.ravel(), lo.ravel()


def _write(path, lat, lon, tp_mm):
    netCDF4 = pytest.importorskip("netCDF4")
    m, n = tp_mm.shape
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("sample", 1)
        ds.createDimension("ensemble_member", m)
        ds.createDimension("grid_point_hres", n)
        ds.createDimension("weather_state", 2)
        ds.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = lat
        ds.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = lon
        ws = ds.createVariable("weather_state", str, ("weather_state",))
        ws[0], ws[1] = "2t", "tp"
        y = np.zeros((1, m, n, 2))
        y[0, :, :, 1] = tp_mm / 1000.0
        ds.createVariable("y_pred", "f8", ("sample", "ensemble_member", "grid_point_hres", "weather_state"))[:] = y


def test_mask_drops_the_strip_and_keeps_global():
    lat, lon = _grid()
    keep = box_edge_mask(lat, lon, 1.0)
    de = edge_distance_deg(lat, lon)
    assert not keep[de < 1.0].any() and keep[de >= 1.0].all()
    glat, glon = _grid(-90, 90, -180, 179.5, 2.0)
    assert is_global(glat, glon) and box_edge_mask(glat, glon, 1.0).all()


def test_scan_flags_persistent_edge_rain(tmp_path):
    lat, lon = _grid()
    de = edge_distance_deg(lat, lon)
    rng = np.random.default_rng(0)
    for d in range(4):
        tp = rng.uniform(0, 50, (3, lat.size))
        tp[:, np.flatnonzero(de < 0.1)[:5]] = 830.0
        _write(tmp_path / f"predictions_2023010{d + 1}_step024.nc", lat, lon, tp)
    res = scan_edge_artefact(tmp_path)
    assert res["checked"] == 4 and res["artefact"] and res["hot_share"] == 1.0


def test_scan_ignores_inner_storm_and_clean_rows(tmp_path):
    lat, lon = _grid()
    de = edge_distance_deg(lat, lon)
    for d in range(4):
        tp = np.full((3, lat.size), 5.0)
        tp[:, np.flatnonzero(de > 5)[100]] = 600.0  # a storm far from the edge
        _write(tmp_path / f"predictions_2023010{d + 1}_step024.nc", lat, lon, tp)
    res = scan_edge_artefact(tmp_path)
    assert res["checked"] == 4 and not res["artefact"]


def test_station_matcher_drops_strip_points():
    pytest.importorskip("scipy")
    from eval.evaluators.obs_points.core.stations import StationMatcher

    lat, lon = _grid()
    m = StationMatcher(lat, lon)
    idx, keep = m.match(np.array([69.9, 40.0]), np.array([0.0, 0.0]), 100.0)
    assert keep.tolist() == [False, True]
    idx, keep = StationMatcher(lat, lon, 0.0).match(np.array([69.9]), np.array([0.0]), 100.0)
    assert keep.tolist() == [True]
