"""The temporal read on synthetic files: a prediction equal to the truth has ratio 1 in every
band; a prediction whose fine band is redrawn at every step has a fine-band ratio well above
1 and a lag-1 correlation near 0 while the large-scale band stays near the truth."""
from __future__ import annotations

import csv

import netCDF4 as nc
import numpy as np
import pytest

from eval.tools.temporal_consistency import read as tr

STATES = ["10u", "10v", "msl"]


def _fields(rng, LA, LO, t, fine_seed):
    """truth = slow large-scale wave advected in time + fine field that persists; the fine part of
    a 'flicker' prediction is redrawn from fine_seed."""
    large = np.sin(LA / 4.0 + 0.3 * t) + np.cos(LO / 5.0 - 0.2 * t)
    r = np.random.default_rng(fine_seed)
    fine = r.standard_normal(LA.shape)
    # keep only scales near 40 km: smooth white noise lightly (3-point box) so it is not pure grid noise
    fine = (fine + np.roll(fine, 1, 0) + np.roll(fine, 1, 1)) / 3.0
    return large, fine


def _write(path, *, step, mode, members=1):
    lat1 = np.arange(35.0, 45.0 + 1e-9, 0.08)
    lon1 = np.arange(-30.0, -15.0 + 1e-9, 0.08)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    t = step / 12.0
    large, fine_persist = _fields(None, LA, LO, t, fine_seed=7)
    n = LA.size
    with nc.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", members)
        d.createDimension("grid_point_hres", n)
        d.createDimension("weather_state", len(STATES))
        d.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = LA.ravel()
        d.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = np.mod(LO.ravel(), 360.0)
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(STATES):
            ws[i] = s
        d.createVariable("ensemble_member", "i4", ("ensemble_member",))[:] = np.arange(1, members + 1)
        dims = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        for name in ("y_pred", "y", "x_interp"):
            d.createVariable(name, "f4", dims)
        for k in range(members):
            truth = large + 0.3 * fine_persist
            if mode == "truth":
                pred = truth
            else:   # flicker: the fine part is redrawn at every step
                _, fine_new = _fields(None, LA, LO, t, fine_seed=100 + step + 10 * k)
                pred = large + 0.3 * fine_new
            msl_truth = 1010.0 - 20.0 * np.exp(-((LA - 40.0) ** 2 + (LO + 22.0 - 0.5 * t) ** 2) / 2.0)
            for name, fld in (("y", truth), ("y_pred", pred), ("x_interp", large)):
                arr = np.stack([fld.ravel(), fld.ravel() * 0.5,
                                (msl_truth if name != "y_pred" else msl_truth - 0.5 * (k + 1) * (step / 12.0 % 2)).ravel()], axis=1)
                d[name][0, k, :, :] = arr


def _run(tmp_path, mode):
    pdir = tmp_path / mode
    pdir.mkdir()
    for step in (12, 24, 36):
        _write(pdir / f"predictions_20230226_step{step:03d}.nc", step=step, mode=mode)
    return pdir


def test_truth_prediction_has_ratio_one(tmp_path):
    pdir = _run(tmp_path, "truth")
    out = tmp_path / "out"
    tr.main(["--run", f"t={pdir}", "--output-dir", str(out), "--variables", "10u,msl",
             "--storm-box", "36,44,-28,-16", "--storm-radius-deg", "3"])
    rows = list(csv.DictReader(open(out / "bands.csv")))
    pred = {(r["variable"], r["band"]): float(r["mean_abs_diff"]) for r in rows if r["field"] == "pred"}
    truth = {(r["variable"], r["band"]): float(r["mean_abs_diff"]) for r in rows if r["field"] == "truth"}
    for key in pred:
        assert pred[key] == pytest.approx(truth[key], rel=1e-6)
    assert (out / "read.md").exists()
    storms = list(csv.DictReader(open(out / "storms.csv")))
    assert any(r["quantity"] == "msl_min" and r["field"] == "truth" for r in storms)


def test_flicker_is_seen_in_the_fine_band_only(tmp_path):
    pdir = _run(tmp_path, "flicker")
    out = tmp_path / "out"
    tr.main(["--run", f"f={pdir}", "--output-dir", str(out), "--variables", "10u", "--storm-box", ""])
    rows = list(csv.DictReader(open(out / "bands.csv")))
    by = {(r["band"], r["field"]): r for r in rows if r["variable"] == "10u"}
    fine_ratio = float(by[("31-56km", "pred")]["mean_abs_diff"]) / float(by[("31-56km", "truth")]["mean_abs_diff"])
    large_ratio = float(by[("gt125km", "pred")]["mean_abs_diff"]) / float(by[("gt125km", "truth")]["mean_abs_diff"])
    assert fine_ratio > 2.0
    assert abs(large_ratio - 1.0) < 0.2
    assert float(by[("31-56km", "pred")]["lag1_corr"]) < 0.3
    assert float(by[("31-56km", "truth")]["lag1_corr"]) > 0.9
