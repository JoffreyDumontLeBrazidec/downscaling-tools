"""tp_maps evaluator: plan by date, stills per file, one video per date with several leads, auto panels."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

xr = pytest.importorskip("xarray")


def _write(tmp_path, date, step, *, input_tp_zero=True, truth_nan=False):
    from eval.tools.map_video.grid import OctahedralGrid

    g = OctahedralGrid.build(24)
    lat, lon = g.point_lat_lon()
    tp = np.maximum(np.sin(np.radians(lon) * 3 + step) * np.cos(np.radians(lat) * 2), 0) * 0.01
    pred = np.stack([tp, tp], -1)[None, None].astype(np.float32)
    x_interp = np.zeros_like(pred) if input_tp_zero else pred * 0.5
    y = np.full_like(pred[:, 0], np.nan) if truth_nan else pred[:, 0] * 1.2
    dims = ["sample", "ensemble_member", "grid_point_hres", "weather_state"]
    ds = xr.Dataset({"y_pred": (dims, pred), "x_interp": (dims, x_interp),
                     "y": (["sample", "grid_point_hres", "weather_state"], y),
                     "date": (["sample"], np.array([f"{date[:4]}-{date[4:6]}-{date[6:]}"])),
                     "lon_hres": (["grid_point_hres"], lon), "lat_hres": (["grid_point_hres"], lat)},
                    coords={"weather_state": ["tp", "cp"], "ensemble_member": [1]})
    ds.to_netcdf(tmp_path / f"predictions_{date}_step{step:03d}.nc")


def test_plan_groups_by_date_and_honours_dates_steps(tmp_path):
    from eval.evaluators.tp_maps.runner import plan

    for d in ("20230826", "20230827", "20230828", "20230829"):
        for s in (24, 48):
            (tmp_path / f"predictions_{d}_step{s:03d}.nc").touch()
    assert list(plan(tmp_path, {})) == ["20230826", "20230827", "20230828"]
    got = plan(tmp_path, {"dates": ["20230829"], "steps": [48]})
    assert list(got) == ["20230829"] and [p.name for p in got["20230829"]] == ["predictions_20230829_step048.nc"]


def test_auto_sources_drop_empty_input_and_nan_truth(tmp_path):
    from eval.tools.map_video.__main__ import auto_sources

    _write(tmp_path, "20230829", 24)
    assert auto_sources(tmp_path / "predictions_20230829_step024.nc", "tp") == ["pred", "truth"]
    _write(tmp_path, "20230830", 24, input_tp_zero=False, truth_nan=True)
    assert auto_sources(tmp_path / "predictions_20230830_step024.nc", "tp") == ["input", "pred"]


def test_run_writes_stills_crops_previews_and_one_video_per_date(tmp_path):
    from eval.evaluators.tp_maps import run

    pred = tmp_path / "pred"
    pred.mkdir()
    for s in (12, 24):
        _write(pred, "20230829", s)
    _write(pred, "20230830", 24)
    out = tmp_path / "out"
    cfg = {"res": 2.0, "crop_res": 2.0, "geography": None, "tween": 1, "crops": {"box": [-30, 40, 0, 60]}}
    run(pred, {}, cfg, output_dir=out, run_label="test")
    stills = sorted(p.name for p in (out / "stills").glob("*.png"))
    assert stills == ["tp_2023082900_+012h.png", "tp_2023082900_+024h.png", "tp_2023083000_+024h.png"]
    assert len(list((out / "crops" / "box" / "stills").glob("*.png"))) == 3
    has_ffmpeg = shutil.which("ffmpeg") is not None
    assert sorted(p.name for p in out.glob("*.mp4")) == (["tp_20230829.mp4"] if has_ffmpeg else [])
    previews = sorted(p.name for p in (out / "previews").iterdir())
    want = ["tp_20230829.png", "tp_20230829_box.png", "tp_20230830.png", "tp_20230830_box.png"]
    if has_ffmpeg:
        want += ["tp_20230829.mp4", "tp_20230829_box.mp4"]
    assert previews == sorted(want)
    assert all(p.stat().st_size < 1_000_000 for p in (out / "previews").iterdir())
    with pytest.raises(FileExistsError):
        run(pred, {}, {"res": 2.0}, output_dir=out)


def test_run_without_tp_writes_skipped_note(tmp_path):
    from eval.evaluators.tp_maps import run

    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred, "20230829", 24)
    out = run(pred, {}, {"variable": "2t"}, output_dir=tmp_path / "out")
    assert (out / "SKIPPED.txt").exists() and not (out / "stills").exists()


def test_crops_default_on_globe_only():
    from eval.evaluators.tp_maps.runner import GLOBAL_CROPS, crops_for

    assert crops_for({}, True) == GLOBAL_CROPS
    assert crops_for({}, False) == {}
    assert crops_for({"crops": False}, True) == {}
    assert crops_for({"crops": {"x": [0, 10, 0, 10]}}, False) == {"x": [0.0, 10.0, 0.0, 10.0]}


def test_poster_still_is_middle_lead(tmp_path):
    from eval.evaluators.tp_maps.previews import poster_still

    paths = [tmp_path / f"tp_2023082900_+{h:03d}h.png" for h in (72, 12, 24, 36, 48, 60)]
    assert poster_still(paths).name == "tp_2023082900_+036h.png"
    assert poster_still([]) is None


def test_tp_maps_joins_any_tp_selection():
    import argparse

    from eval.cli._selection import _resolve_evaluators

    lane = {"evaluator_groups": {"default": ["tc", "surface"], "diagnostics": ["precip_dist"]}}
    ns = lambda **kw: argparse.Namespace(**{"only": None, "include_diagnostics": False, "no_tp_maps": False, **kw})
    assert _resolve_evaluators(ns(only="precip_read"), lane) == ["precip_read", "tp_maps"]
    assert _resolve_evaluators(ns(only="surface"), lane) == ["surface"]
    assert _resolve_evaluators(ns(include_diagnostics=True), lane) == ["tc", "surface", "precip_dist", "tp_maps"]
    assert _resolve_evaluators(ns(only="precip_read,tp_maps", no_tp_maps=True), lane) == ["precip_read"]
