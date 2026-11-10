"""The box-edge strip applied by every evaluator (eval.shared.edge_strip; docs ablation note s21).

Synthetic small grids: a regional 1-degree box whose 1-degree edge strip carries a huge artefact (as the
slot-head models write there), and a global grid that must be left untouched.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from eval.shared import edge_strip as E

HOT = 900.0   # the artefact value written in the strip (mm of tp, m/s of wind, ...)


def _box(lat0=10.0, lat1=20.0, lon0=0.0, lon1=12.0, step=0.5):
    la, lo = np.meshgrid(np.arange(lat0, lat1 + 1e-9, step), np.arange(lon0, lon1 + 1e-9, step), indexing="ij")
    return la.ravel(), lo.ravel()


def _globe(step=5.0):
    la, lo = np.meshgrid(np.arange(-87.5, 90, step), np.arange(0, 360, step), indexing="ij")
    return la.ravel(), lo.ravel()


# --------------------------------------------------------------------------- the helpers


def test_resolve_order_config_then_active_then_env_then_default(monkeypatch):
    monkeypatch.delenv(E.EDGE_DEG_ENV, raising=False)
    assert E.resolve_box_edge_deg() == E.DEFAULT_BOX_EDGE_DEG == 1.0
    assert E.resolve_box_edge_deg({"box_edge_deg": None}) == 1.0      # null means the default
    assert E.resolve_box_edge_deg({"box_edge_deg": 0}) == 0.0         # the per-evaluator opt-out
    monkeypatch.setenv(E.EDGE_DEG_ENV, "2.5")                           # what a subprocess inherits
    assert E.resolve_box_edge_deg() == 2.5
    monkeypatch.delenv(E.EDGE_DEG_ENV)
    with E.configured({"box_edge_deg": 0.5}):
        assert E.resolve_box_edge_deg() == 0.5
        assert os.environ[E.EDGE_DEG_ENV] == "0.5"
        assert E.resolve_box_edge_deg({"box_edge_deg": 3}) == 3.0
        assert E.subprocess_env({"box_edge_deg": 0})[E.EDGE_DEG_ENV] == "0.0"
    assert E.EDGE_DEG_ENV not in os.environ and E.resolve_box_edge_deg() == 1.0


def test_index_is_a_free_slice_on_global_grids_and_without_a_strip():
    glat, glon = _globe()
    assert E.box_edge_index(glat, glon) == slice(None)
    lat, lon = _box()
    assert E.box_edge_index(lat, lon, 0.0) == slice(None)
    idx = E.box_edge_index(lat, lon, 1.0)
    assert np.array_equal(idx, np.flatnonzero(E.box_edge_mask(lat, lon, 1.0)))
    assert lat[idx].min() == 11.0 and lat[idx].max() == 19.0 and lon[idx].min() == 1.0 and lon[idx].max() == 11.0
    with pytest.raises(ValueError, match="covers every one"):
        E.box_edge_index(lat, lon, 6.0)


def test_inner_bounds_agree_with_the_mask_also_across_the_dateline():
    lat, lon = _box(lon0=170.0, lon1=190.0)
    lon = np.where(lon >= 180.0, lon - 360.0, lon)       # stored -180..180: the box crosses the dateline
    b = E.inner_bounds(lat, lon, 1.0)
    assert b is not None and b[4] == 360
    assert np.array_equal(E.inside_bounds(b, lat, lon), E.box_edge_mask(lat, lon, 1.0))
    # a regular map grid over the same box: the strip cells are outside, the inner ones inside
    assert E.inside_bounds(b, np.array([15.0, 10.2]), np.array([-178.0, -178.0])).tolist() == [True, False]
    glat, glon = _globe()
    assert E.inner_bounds(glat, glon) is None and E.inside_bounds(None, glat, glon).all()


def test_strip_box_edge_subsets_the_output_grid_only():
    xr = pytest.importorskip("xarray")
    lat, lon = _box()
    n = lat.size
    ds = xr.Dataset({
        "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), np.zeros((1, 2, n, 1))),
        "x": (("sample", "ensemble_member", "grid_point_lres", "weather_state"), np.zeros((1, 2, 7, 1))),
        "lat_hres": (("grid_point_hres",), lat), "lon_hres": (("grid_point_hres",), lon)})
    out = E.strip_box_edge(ds)
    keep = E.box_edge_mask(lat, lon)
    assert out.sizes["grid_point_hres"] == int(keep.sum()) and out.sizes["grid_point_lres"] == 7
    assert np.array_equal(out["lat_hres"].values, lat[keep]) and out.attrs["box_edge_strip_removed_deg"] == 1.0
    assert E.strip_box_edge(ds, 0.0) is ds
    glat, glon = _globe()
    g = xr.Dataset({"lat_hres": (("grid_point_hres",), glat), "lon_hres": (("grid_point_hres",), glon)})
    assert E.strip_box_edge(g) is g


# --------------------------------------------------------------------------- evaluators


def _write_pred(path, lat, lon, *, states=("tp", "10u", "10v", "msl", "2t"), members=2, hot=True, seed=0):
    """A prediction file whose strip holds HOT in every field of y_pred (truth and input are calm)."""
    netCDF4 = pytest.importorskip("netCDF4")
    rng = np.random.default_rng(seed)
    n, s = lat.size, len(states)
    strip = ~E.box_edge_mask(lat, lon) if hot else np.zeros(n, bool)
    base = {"tp": 0.002, "10u": 5.0, "10v": 2.0, "msl": 101000.0, "2t": 280.0}
    y = np.stack([np.full(n, base[v]) + 0.1 * rng.standard_normal(n) * (base[v] > 1) for v in states], -1)
    y = np.broadcast_to(y, (1, members, n, s)).copy()
    scale = np.array([1e-5 if v == "tp" else 0.01 for v in states])
    yp = y + scale * rng.standard_normal(y.shape)
    for j, v in enumerate(states):
        yp[0, :, strip, j] = HOT / 1000.0 if v == "tp" else HOT
    with netCDF4.Dataset(path, "w") as ds:
        for name, size in (("sample", 1), ("ensemble_member", members), ("grid_point_hres", n),
                           ("weather_state", s)):
            ds.createDimension(name, size)
        ds.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = lat
        ds.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = lon
        ws = ds.createVariable("weather_state", str, ("weather_state",))
        for j, v in enumerate(states):
            ws[j] = v
        ds.createVariable("ensemble_member", "i4", ("ensemble_member",))[:] = np.arange(1, members + 1)
        dims = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        ds.createVariable("y", "f8", dims)[:] = y
        ds.createVariable("y_pred", "f8", dims)[:] = yp
        ds.createVariable("x_interp", "f8", dims)[:] = y
    return strip


def test_surface_mse_ignores_the_strip(tmp_path):
    pytest.importorskip("xarray")
    from eval.evaluators.surface.core.compute import process_predictions_dir

    lat, lon = _box()
    d = tmp_path / "pred"
    d.mkdir()
    _write_pred(d / "predictions_20230101_step024.nc", lat, lon)
    out = process_predictions_dir(d)
    assert out["weighted_surface_mse"] < 1.0                 # the HOT strip would give ~1e5
    with E.configured({"box_edge_deg": 0}):
        assert process_predictions_dir(d)["weighted_surface_mse"] > 1e3


def test_lane_diagnostics_peaks_and_rmse_leave_the_strip_out(tmp_path):
    from eval.evaluators.lane_diagnostics import compute as C

    lat, lon = _box()
    d = tmp_path / "arm" / "predictions"
    d.mkdir(parents=True)
    _write_pred(d / "predictions_20230101_step024.nc", lat, lon)
    peaks = C.sampler_peaks({"arm": tmp_path / "arm"})["arm"]
    assert max(peaks["peaks_mm"]) < 10.0
    rows = C.pair_coherence(d, channel="2t")
    assert rows[0]["rmse_model"] < 1.0 and rows[0]["n"] == int(E.box_edge_mask(lat, lon).sum())
    wind = C.box_wind(d, {"south": 0, "north": 30, "west": -5, "east": 20})
    assert max(r["model"] for r in wind) < 20.0


def test_precip_read_and_precip_events_never_see_the_strip(tmp_path):
    from eval.evaluators.precip_events.core.precip_events import find_precip_events
    from eval.evaluators.precip_read.runner import read_tp

    lat, lon = _box()
    d = tmp_path / "pred"
    d.mkdir()
    _write_pred(d / "predictions_20230101_step024.nc", lat, lon)
    la, lo, y, xi, yp = read_tp(d / "predictions_20230101_step024.nc")
    assert la.size == int(E.box_edge_mask(lat, lon).sum()) and yp.max() < 10.0
    ev = find_precip_events(d, n_events=1, rank_by="pred", dlat=2, dlon=2)[0]
    assert ev.peak_value < 0.01 and 11.0 <= ev.lat <= 19.0


def test_precip_dist_tail_stats_without_the_strip(tmp_path):
    import csv

    from eval.evaluators.precip_dist.core.tail_stats import compute_tail_stats

    lat, lon = _box()
    d = tmp_path / "pred"
    d.mkdir()
    _write_pred(d / "predictions_20230101_step024.nc", lat, lon)
    rows = list(csv.DictReader(open(compute_tail_stats(d, tmp_path, per_date=False))))
    pred = next(r for r in rows if r["series"] == "pred")
    assert int(pred["n"]) == 2 * int(E.box_edge_mask(lat, lon).sum())
    raw = list(csv.DictReader(open(compute_tail_stats(d, tmp_path, per_date=False, box_edge_deg=0.0))))
    assert int(next(r for r in raw if r["series"] == "pred")["n"]) == 2 * lat.size


def test_tc_structure_rejects_a_centre_in_the_strip():
    from eval.evaluators.tc_structure.runner import _centre_outside_strip

    inner = np.array([True, True, False])
    assert _centre_outside_strip({"found": True, "imin": 1, "reason": ""}, inner)["found"]
    c = _centre_outside_strip({"found": True, "imin": 2, "reason": ""}, inner)
    assert not c["found"] and c["reason"] == "minimum in the box-edge strip"


def test_tc_native_curve_points_leave_the_strip_out(tmp_path):
    from eval.evaluators.tc.core.data_types import BoundingBox
    from eval.evaluators.tc.core.loading_predictions import load_prediction_curves

    lat, lon = _box()
    f = tmp_path / "predictions_20230101_step024.nc"
    _write_pred(f, lat, lon)
    bbox = BoundingBox(north=30.0, south=0.0, east=20.0, west=-5.0)
    cur = load_prediction_curves([(f, 20230101, 24)], bbox=bbox, support_mode="native")
    assert cur.wind.max() < 50.0 and cur.wind.size == 2 * int(E.box_edge_mask(lat, lon).sum())


def test_global_files_are_untouched(tmp_path):
    from eval.evaluators.precip_read.runner import read_tp

    glat, glon = _globe()
    f = tmp_path / "predictions_20230101_step024.nc"
    _write_pred(f, glat, glon, hot=False)
    assert read_tp(f)[0].size == glat.size


def test_cli_gate_safe_set_and_the_per_evaluator_opt_out(tmp_path, monkeypatch):
    from eval.cli import evaluate as ev

    assert "sigma_loss" not in ev.EDGE_STRIP_SAFE and "quaver" not in ev.EDGE_STRIP_SAFE
    assert {"surface", "texture", "wind_extremes", "tc_structure", "zoom_maps"} <= ev.EDGE_STRIP_SAFE
    monkeypatch.setattr(ev, "_edge_artefact_gate",
                        lambda p, o: {"artefact": True, "allowed_by_env": False, "detail": "test"})
    status = []
    monkeypatch.setattr(ev, "_write_evaluator_status", lambda o, n, s, detail="": status.append((n, s)))
    with pytest.raises(RuntimeError, match="surface"):   # gated: the run reports it as not produced
        ev._run_evaluators(tmp_path, {"surface": {"box_edge_deg": 0}}, ["surface"], tmp_path)
    assert status[0] == ("surface", "skipped")
