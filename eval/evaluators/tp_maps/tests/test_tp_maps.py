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


def test_run_writes_stills_and_one_video_per_date(tmp_path):
    from eval.evaluators.tp_maps import run

    pred = tmp_path / "pred"
    pred.mkdir()
    for s in (12, 24):
        _write(pred, "20230829", s)
    _write(pred, "20230830", 24)
    out = tmp_path / "out"
    run(pred, {}, {"res": 2.0, "geography": None, "tween": 1}, output_dir=out, run_label="test")
    stills = sorted(p.name for p in (out / "stills").glob("*.png"))
    assert stills == ["tp_2023082900_+012h.png", "tp_2023082900_+024h.png", "tp_2023083000_+024h.png"]
    videos = sorted(p.name for p in out.glob("*.mp4"))
    assert videos == (["tp_20230829.mp4"] if shutil.which("ffmpeg") else [])
    with pytest.raises(FileExistsError):
        run(pred, {}, {"res": 2.0}, output_dir=out)
