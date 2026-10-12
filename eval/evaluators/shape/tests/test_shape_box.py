"""The shape evaluator on a box other than the epic's (gap G1 of the run-judging protocol).

A small synthetic file stands in for an O1280 prediction: a regular 0.1-degree grid stored in
the 0..360 longitude convention, three members. The open-ocean masks are written to the
per-box cache beforehand (building them needs the global forcings of Atos).
"""
from __future__ import annotations

import json

import netCDF4 as nc
import numpy as np
import pytest

from eval.evaluators.shape import boxes as B
from eval.evaluators.shape import runner

BOX = [40.0, 46.0, -40.0, -30.0]
MASK_KEYS = ("interior", "all_interior", "ocean_interior", "open_ocean_interior",
             "coastal_interior", "flat_land_interior", "mountain_interior")


def _smooth_noise(rng, shape, n=3):
    f = rng.standard_normal(shape)
    for _ in range(n):
        f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5
    return f


def _write_file(path, *, shared_input: bool, members=3, seed=0):
    rng = np.random.default_rng(seed)
    lat1 = np.arange(35.0, 50.0 + 1e-9, 0.1)
    lon1 = np.arange(-50.0, -20.0 + 1e-9, 0.1)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    ny, nx = LA.shape
    lat = LA.ravel()
    lon = np.mod(LO.ravel(), 360.0)          # 0..360 convention, as in the O1280 files
    n = lat.size
    states = ["10u", "10v"]
    with nc.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", members)
        d.createDimension("grid_point_hres", n)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = lat
        d.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = lon
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            ws[i] = s
        dims = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        for name in ("y_pred", "y", "x_interp"):
            d.createVariable(name, "f4", dims)
        base = np.stack([5 + _smooth_noise(rng, (ny, nx), 8).ravel() * 3 for _ in states], -1)
        for k in range(members):
            xi = base if shared_input else base + 0.5 * k
            noise = lambda: np.stack([_smooth_noise(rng, (ny, nx)).ravel() for _ in states], -1)  # noqa: E731
            d["x_interp"][0, k, :, :] = xi
            d["y"][0, k, :, :] = xi + noise()
            d["y_pred"][0, k, :, :] = xi + noise()
    return lat, lon


@pytest.fixture()
def setup(tmp_path):
    pred = tmp_path / "pred"
    pred.mkdir()
    stats = tmp_path / "stats.npy"
    np.save(stats, {"stdev": {"10u": 2.0, "10v": 2.0}}, allow_pickle=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    return pred, stats, cache


def _premask(path, cache):
    box = B.resolve_box(BOX)
    sel, lat, lon, _ = B.select_box(path, box)
    cp = B.cache_paths(box, cache, n_points=sel.size)
    lat_lo, lat_hi = lat.min() + 0.7, lat.max() - 0.7
    lon_lo, lon_hi = lon.min() + 0.7, lon.max() - 0.7
    interior = (lat >= lat_lo) & (lat <= lat_hi) & (lon >= lon_lo) & (lon <= lon_hi)
    np.savez(cp["mask_cache"], **{k: (interior if "interior" in k and "land" not in k
                                      and "mountain" not in k and "coastal" not in k
                                      else np.zeros_like(interior)) for k in MASK_KEYS})
    return box, sel, cp


def test_resolve_box_names_lists_and_strings():
    assert B.resolve_box(None).tag == "tc_atlantic"
    assert B.resolve_box("north_atlantic").as_list() == [35.0, 55.0, -45.0, -20.0]
    assert B.resolve_box([35, 55, 315, 340]).tag == "north_atlantic"   # 0..360 input
    custom = B.resolve_box("40,46,-40,-30")
    assert custom.as_list() == BOX and custom.tag not in B.BOXES
    with pytest.raises(ValueError):
        B.resolve_box("atlantis")


def test_select_box_handles_0_360_files_and_refuses_partial_boxes(tmp_path):
    f = tmp_path / "predictions_20230226_step024.nc"
    _write_file(f, shared_input=False)
    sel, lat, lon, note = B.select_box(f, B.resolve_box(BOX))
    assert 60 * 100 <= sel.size <= 61 * 101          # edges up to float rounding
    assert -40.0 <= lon.min() < -39.85 and -30.15 < lon.max() <= -30.0
    assert 40.0 <= lat.min() < 40.15 and 45.85 < lat.max() <= 46.0
    with pytest.raises(ValueError, match="not fully inside"):
        B.select_box(f, B.resolve_box([45.0, 55.0, -40.0, -30.0]))   # file stops at 50 N


@pytest.mark.parametrize("shared_input", [False, True])
def test_shape_runs_on_a_custom_box(setup, shared_input):
    pred, stats, cache = setup
    f = pred / "predictions_20230226_step024.nc"
    _write_file(f, shared_input=shared_input)
    box, sel, cp = _premask(f, cache)

    out = runner.run(pred, {}, {"box": BOX, "cache_dir": str(cache), "stats_file": str(stats),
                                "steps": [24], "n_boot": 50},
                     output_dir=pred / "shape", run_label="model")
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["box_bounds"] == BOX and meta["box_points"] == sel.size
    assert meta["geom_cache"] == cp["geom_cache"] and meta["mask_cache"] == cp["mask_cache"]
    assert meta["files"][0]["members_share_input"] is shared_input
    rows = runner.read_rows(out / "rows.csv")
    # 3 members x 2 vars x 3 fields (draw, truth, driver) x 2 bands
    assert len(rows) == 3 * 2 * 3 * 2
    summary = json.loads((out / "summary.json").read_text())["rows"]
    assert {r["window"] for r in summary} == {"all"}
    ef = [r for r in summary if r["statistic"] == "elong_frac_gt3" and r["field"] == "draw"]
    assert ef and all(0.0 <= r["mean"] <= 1.0 for r in ef)


def test_a_mask_cache_of_another_box_is_refused(setup):
    pred, stats, cache = setup
    f = pred / "predictions_20230226_step024.nc"
    _write_file(f, shared_input=False)
    box, sel, cp = _premask(f, cache)
    z = dict(np.load(cp["mask_cache"]))
    np.savez(cp["mask_cache"], **{k: v[:-10] for k, v in z.items()})
    with pytest.raises((ValueError, IndexError)):
        runner.run(pred, {}, {"box": BOX, "cache_dir": str(cache), "stats_file": str(stats),
                              "steps": [24], "n_boot": 50}, output_dir=pred / "shape")


def test_module_constants_reset_between_runs(setup):
    from eval.evaluators.shape import instrument as S

    _, stats, cache = setup
    runner._configure({"regrid_deg": 0.05, "stats_file": str(stats)})
    assert S.REGRID_DEG == 0.05
    runner._configure({})
    assert S.REGRID_DEG == 0.09 and S.STATS != str(stats)
