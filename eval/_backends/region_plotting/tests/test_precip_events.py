"""Unit tests for the heavy-precip event finder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import pytest

from eval._backends.region_plotting.precip_events import find_precip_events


def _write_nc(path: Path, *, date: str, step: int, peak_value: float,
              peak_lat: float, peak_lon: float, n_gp: int = 64):
    """Write a minimal predictions NC: y/y_pred with a single tp spike at one gridpoint.

    The spike gridpoint's lat/lon are pinned exactly to (peak_lat, peak_lon) so the
    finder's argmax location is deterministic and the bbox is centred there.
    """
    lats = np.linspace(-60, 60, n_gp).astype("float64")
    lons = np.linspace(-150, 150, n_gp).astype("float64")
    peak_idx = n_gp // 2
    lats[peak_idx] = peak_lat
    lons[peak_idx] = peak_lon
    y = np.full((1, 1, n_gp, 1), 1e-4, dtype="float32")
    y[0, 0, peak_idx, 0] = peak_value
    ds = xr.Dataset(
        {
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y * 0.5),
            "lat_hres": (("grid_point_hres",), lats),
            "lon_hres": (("grid_point_hres",), lons),
        },
        coords={"weather_state": ["tp"]},
    )
    ds.to_netcdf(path)


def test_ranks_by_truth_max_descending(tmp_path):
    _write_nc(tmp_path / "predictions_20250926_step024.nc",
              date="20250926", step=24, peak_value=0.05, peak_lat=20.0, peak_lon=-70.0)
    _write_nc(tmp_path / "predictions_20250927_step048.nc",
              date="20250927", step=48, peak_value=0.20, peak_lat=-10.0, peak_lon=30.0)
    _write_nc(tmp_path / "predictions_20250928_step072.nc",
              date="20250928", step=72, peak_value=0.10, peak_lat=40.0, peak_lon=-100.0)

    events = find_precip_events(tmp_path, n_events=8, dlat=12, dlon=15, rank_by="truth")

    assert [e.peak_value for e in events] == pytest.approx([0.20, 0.10, 0.05], abs=1e-6)
    assert events[0].date == "20250927" and events[0].step == 48
    e0 = events[0]
    assert e0.bbox == pytest.approx([-10.0 - 12, -10.0 + 12, 30.0 - 15, 30.0 + 15], abs=1.0)
    assert e0.label == "event01_20250927_step048"


def test_n_events_truncates(tmp_path):
    for i, pv in enumerate([0.05, 0.20, 0.10]):
        _write_nc(tmp_path / f"predictions_2025092{i+6}_step0{i+2}4.nc",
                  date=f"2025092{i+6}", step=(i + 2) * 10 + 4, peak_value=pv,
                  peak_lat=0.0, peak_lon=0.0)
    events = find_precip_events(tmp_path, n_events=2, dlat=12, dlon=15, rank_by="truth")
    assert len(events) == 2
    assert events[0].peak_value == pytest.approx(0.20, abs=1e-6)


def test_rank_by_pred_uses_y_pred(tmp_path):
    _write_nc(tmp_path / "predictions_20250926_step024.nc",
              date="20250926", step=24, peak_value=0.20, peak_lat=20.0, peak_lon=-70.0)
    events = find_precip_events(tmp_path, n_events=8, dlat=12, dlon=15, rank_by="pred")
    assert events[0].peak_value == pytest.approx(0.10, abs=1e-6)


def test_bbox_clamped_near_pole(tmp_path):
    _write_nc(tmp_path / "predictions_20250926_step024.nc",
              date="20250926", step=24, peak_value=0.20, peak_lat=60.0, peak_lon=150.0)
    events = find_precip_events(tmp_path, n_events=8, dlat=40, dlon=45, rank_by="truth")
    lat_min, lat_max, lon_min, lon_max = events[0].bbox
    assert lat_max <= 90.0 and lat_min >= -90.0
    assert lon_max <= 180.0 and lon_min >= -180.0


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_precip_events(tmp_path, n_events=8, dlat=12, dlon=15, rank_by="truth")
