"""Regression tests: a regional run must get truth on its own support from the first load.

A regional (box-cut) run predicts on a subset of the model's global grid while the
truth GRIB holds the full grid. `PrecipTruthSource` builds the support index inside
`verify_grid()`, so a caller that loaded truth before calling it received the full
grid for its first lead time: `precip_events` crashed with `rank_by: truth`, and the
`precip_dist` histogram counted the full grid for that lead time. These tests use a
small synthetic regional prediction file and a fake full-grid truth reader.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eval._backends.precip import sources
from eval._backends.precip.sources import PrecipTruthSource
from eval._backends.precip.tp_histogram_comparison import accumulate_tp_by_step
from eval._backends.region_plotting.precip_events import find_precip_events

N_FULL = 400
REGION = np.arange(150, 230)          # a contiguous box of 80 of the 400 rows
STEPS = (24, 48, 72)
OUTSIDE_PEAK = 0.500                  # largest truth value, OUTSIDE the box
INSIDE_PEAK = 0.200                   # largest truth value inside the box
INSIDE_PEAK_ROW = 200


def _full_grid():
    lats = np.linspace(-80.0, 80.0, N_FULL)
    lons = np.linspace(0.0, 358.0, N_FULL)
    return lats, lons


def _full_truth(step: int) -> np.ndarray:
    vals = np.full(N_FULL, 0.001, dtype=np.float32)
    vals[5] = OUTSIDE_PEAK
    vals[INSIDE_PEAK_ROW] = INSIDE_PEAK
    return vals


@pytest.fixture()
def fake_truth_grib(monkeypatch):
    """Make the default GRIB reader return the synthetic full-grid truth."""
    lats, lons = _full_grid()

    def fake_read(path, var):
        return {(0, s): _full_truth(s) for s in STEPS}, lats, lons

    monkeypatch.setattr(sources, "_read_grib_var", fake_read)
    return "/nowhere/truth_{date}.grib"


def _write_regional_prediction(path, *, date, step):
    lats, lons = _full_grid()
    n = REGION.size
    pred = np.full((1, 1, n, 1), 0.002, dtype=np.float32)
    nan_truth = np.full((1, 1, n, 1), np.nan, dtype=np.float32)
    ds = xr.Dataset(
        {
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), nan_truth),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), pred),
            "lat_hres": (("grid_point_hres",), lats[REGION]),
            "lon_hres": (("grid_point_hres",), lons[REGION]),
        },
        coords={"weather_state": ["tp"]},
    )
    ds.to_netcdf(path)


@pytest.fixture()
def regional_predictions(tmp_path):
    for step in STEPS:
        _write_regional_prediction(
            tmp_path / f"predictions_20250926_step{step:03d}.nc",
            date="20250926", step=step)
    return tmp_path


def test_truth_source_serves_the_region_from_the_first_load(fake_truth_grib):
    lats, lons = _full_grid()
    src = PrecipTruthSource(fake_truth_grib)
    # The order every caller now uses: declare the run's grid, then load.
    src.verify_grid(lats[REGION], lons[REGION])
    first = src.load("20250926", 24)
    assert first.shape == (REGION.size,)
    np.testing.assert_array_equal(first, _full_truth(24)[REGION])


def test_truth_source_warns_when_loaded_before_the_grid_is_declared(fake_truth_grib, caplog):
    src = PrecipTruthSource(fake_truth_grib)
    with caplog.at_level("WARNING", logger=sources.LOG.name):
        out = src.load("20250926", 24)
    assert out.shape == (N_FULL,)
    assert "before verify_grid" in caplog.text


def test_histogram_counts_only_the_region_for_every_lead_time(
        regional_predictions, fake_truth_grib):
    step_data = accumulate_tp_by_step(
        regional_predictions, truth_grib_tpl=fake_truth_grib)
    for step in STEPS:
        truth = step_data[step]["truth"]
        assert truth.n == REGION.size, f"step {step}: counted {truth.n} truth values"
        assert truth.max == pytest.approx(INSIDE_PEAK * 1000.0, rel=1e-3)
        assert step_data[step]["pred"].n == REGION.size


def test_precip_events_ranked_by_truth_on_a_regional_run(
        regional_predictions, fake_truth_grib):
    lats, lons = _full_grid()
    events = find_precip_events(
        regional_predictions, n_events=3, dlat=5, dlon=5, rank_by="truth",
        truth_grib_tpl=fake_truth_grib)
    assert len(events) == 3
    for e in events:
        assert e.peak_value == pytest.approx(INSIDE_PEAK, rel=1e-5)
        assert e.lat == pytest.approx(lats[INSIDE_PEAK_ROW])
        assert e.lon == pytest.approx(lons[INSIDE_PEAK_ROW])


def test_event_plot_data_uses_the_region_on_its_first_load(
        regional_predictions, fake_truth_grib):
    from eval._backends.region_plotting.plot_precip_events import _EventData
    from eval._backends.region_plotting.precip_events import Event

    data = _EventData(fake_truth_grib, "", "", "tp", 0)
    path = regional_predictions / "predictions_20250926_step024.nc"
    event = Event(nc_path=path, date="20250926", step=24, peak_value=0.0,
                  lat=0.0, lon=0.0, bbox=[-5, 5, -5, 5], label="e")
    lat, lon, truth, _base, pred = data.load(event)
    assert truth is not None and truth.size == lat.size == REGION.size
    assert np.nanmax(truth) == pytest.approx(INSIDE_PEAK * 1000.0, rel=1e-5)
