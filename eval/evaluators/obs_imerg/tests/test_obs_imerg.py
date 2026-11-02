"""obs_imerg on synthetic files: the IMERG cut from a tiny zarr store sums the right 6 h steps, run() scores
model, input and truth on the same points and refuses a different grid, plot() draws from the saved files."""
from __future__ import annotations

import json

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from eval.evaluators import obs_imerg

DATES = ["20230101", "20230111"]
STEPS = (12, 24)


def _grid():
    la, lo = np.meshgrid(np.arange(20.0, 65.0, 0.5), np.arange(-30.0, 30.0, 0.5), indexing="ij")
    return la.ravel(), lo.ravel()


def _truth(n, date, step):
    rng = np.random.default_rng(int(date[-2:]) * 100 + step)
    return rng.gamma(0.4, 4.0, n)  # mm


def _pred(path, date, step, members=4, noise=1.0, lat=None, lon=None):
    la, lo = _grid() if lat is None else (lat, lon)
    n = la.size
    t = _truth(n, date, step)
    rng = np.random.default_rng(step)
    with nc.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", members)
        d.createDimension("grid_point_hres", n)
        d.createDimension("weather_state", 1)
        d.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = la
        d.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = np.mod(lo, 360.0)
        ws = d.createVariable("weather_state", str, ("weather_state",))
        ws[0] = "tp"
        d.createVariable("y_pred", "f4", ("sample", "ensemble_member", "grid_point_hres", "weather_state"))
        d.createVariable("y", "f4", ("sample", "grid_point_hres", "weather_state"))
        d.createVariable("x_interp", "f4", ("sample", "grid_point_hres", "weather_state"))
        d["y"][0, :, 0] = t / 1000.0
        d["x_interp"][0, :, 0] = 0.8 * t / 1000.0
        for k in range(members):
            d["y_pred"][0, k, :, 0] = np.clip(t + noise * rng.standard_normal(n), 0, None) / 1000.0


def _imerg(path, lat=None, lon=None):
    la, lo = _grid() if lat is None else (lat, lon)
    valids = sorted({(pd.Timestamp(d) + pd.Timedelta(hours=s)).strftime("%Y%m%d%H") for d in DATES for s in STEPS})
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("valid", len(valids))
        ds.createDimension("point", la.size)
        ds.createVariable("lat", "f8", ("point",))[:] = la
        ds.createVariable("lon", "f8", ("point",))[:] = lo
        v = ds.createVariable("valid", str, ("valid",))
        tp = ds.createVariable("tp", "f4", ("valid", "point"))
        for k, val in enumerate(valids):
            v[k] = val
            d = DATES[0] if val < "2023011100" else DATES[1]
            s = int((pd.to_datetime(val, format="%Y%m%d%H") - pd.Timestamp(d)).total_seconds() // 3600)
            x = _truth(la.size, d, s) + 0.5
            x[:7] = np.nan  # IMERG holes are left out
            tp[k, :] = x
        ds.accum_hours = 12


@pytest.fixture()
def setup(tmp_path):
    pred = tmp_path / "pred"
    pred.mkdir()
    for d in DATES:
        for s in STEPS:
            _pred(pred / f"predictions_{d}_step{s:03d}.nc", d, s)
    _imerg(tmp_path / "imerg.nc")
    cfg = {"imerg": {"jupiter": str(tmp_path / "absent.nc"), "atos": str(tmp_path / "imerg.nc")},
           "regions": {"25-60N": [25, 60, -30, 30], "europe": [35, 60, -12.5, 30]}, "min_points": 100}
    return tmp_path, pred, cfg


def test_run_then_plot(setup):
    tmp, pred, cfg = setup
    out = obs_imerg.run(pred, {}, cfg, output_dir=tmp / "out", run_label="X")
    df = pd.read_csv(out / "scores_per_date.csv", dtype={"date": str})
    assert set(df.row) == {"model", "input", "truth"} and set(df.variant) == {"imerg"}
    assert (df.groupby(["region", "lead", "date"]).n.nunique() == 1).all()
    la, _ = _grid()
    n_region = int(((la >= 25) & (la <= 60)).sum())
    assert (df[df.region == "25-60N"].n == n_region).all()  # the NaN points sit at 20 N, outside the region
    # the truth is IMERG minus 0.5 mm everywhere: its fair CRPS (= MAE) is 0.5 where IMERG >= 0.5
    t = df[(df.row == "truth") & (df.region == "25-60N")]
    np.testing.assert_allclose(t.fcrps, 0.5, atol=1e-5)
    assert {"twcrps5", "brier10", "n_obs_gt20"} <= set(df.columns)
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["sample_unit"] == "grid points" and meta["members"] == [4]
    names = {p.name for p in map(__import__("pathlib").Path, obs_imerg.plot(out, {}, cfg))}
    for f in ("obs_tp_imerg_scores.png", "obs_imerg_tail_25-60N.png", "obs_imerg_rank_hist_europe.png",
              "obs_imerg_tp_pdfs.png"):
        assert f in names, f
    assert "IMERG" in (out / "summary.md").read_text()


def test_refuses_another_grid(setup):
    tmp, pred, cfg = setup
    la, lo = _grid()
    _imerg(tmp / "shifted.nc", la + 0.1, lo)
    with pytest.raises(ValueError, match="not the IMERG cut"):
        obs_imerg.run(pred, {}, {**cfg, "imerg": str(tmp / "shifted.nc")}, output_dir=tmp / "o2")


def test_fetch_sums_the_two_steps_ending_at_valid(tmp_path):
    zarr = pytest.importorskip("zarr")
    from eval.evaluators.obs_imerg.core.fetch import fetch_imerg

    lat = np.array([5.0, 15.0, 45.0, 69.9, 45.0])
    lon = np.array([0.0, 350.0, 30.0, 10.0, 60.0])  # 350 E = 10 W is inside, 60 E is outside the E2 box
    times = np.arange(np.datetime64("2023-01-01T06"), np.datetime64("2023-01-04T06"), np.timedelta64(6, "h"))
    data = np.arange(times.size)[:, None, None, None] * 0.001 + np.zeros((1, 1, 1, lat.size))
    try:  # the real store is zarr format 2
        g = zarr.open_group(str(tmp_path / "s.zarr"), mode="w", zarr_format=2)
    except TypeError:
        g = zarr.open_group(str(tmp_path / "s.zarr"), mode="w")
    for k, v in (("data", data.astype("f4")), ("dates", times.astype("datetime64[s]")), ("latitudes", lat),
                 ("longitudes", lon)):
        g.create_array(k, data=v) if hasattr(g, "create_array") else g.array(k, v)
    out = fetch_imerg(["20230101"], [12, 24], tmp_path / "c.nc", zarr_path=tmp_path / "s.zarr")
    with nc.Dataset(str(out)) as ds:
        assert list(ds["index_o1280"][:]) == [1, 2, 3]
        assert list(ds["valid"][:]) == ["2023010112", "2023010200"]
        # step index of 2023-01-01 06 is 0: window ending 12 UTC = steps 0 and 1 -> (0 + 1) mm
        np.testing.assert_allclose(ds["tp"][0, :], 1.0)
        np.testing.assert_allclose(ds["tp"][1, :], 2.0 + 3.0)
        np.testing.assert_allclose(ds["lon"][:], [-10.0, 30.0, 10.0])
