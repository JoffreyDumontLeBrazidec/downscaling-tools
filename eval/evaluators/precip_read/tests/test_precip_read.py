"""precip_read on synthetic files: fair CRPS matches the pair formula, the scores are exact where
the model is the truth, run() writes every file and plot() draws every figure from them alone."""
from __future__ import annotations

import json

import netCDF4 as nc
import numpy as np
import pytest

from eval.evaluators import precip_read
from eval.evaluators.precip_read.core.metrics import fair_crps, pooled, score_sums


def _write(path, *, mode, members=3, seed=0, truth_members=True):
    rng = np.random.default_rng(seed)
    lat1 = np.arange(10.0, 60.0 + 1e-9, 0.5)
    lon1 = np.arange(-30.0, 20.0 + 1e-9, 0.5)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    n = LA.size
    states = ["2t", "tp"]
    y = rng.gamma(0.3, 0.004, n)
    xi = 0.7 * y + rng.gamma(0.3, 0.001, n)
    with nc.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", members)
        d.createDimension("grid_point_hres", n)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = LA.ravel()
        d.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = np.mod(LO.ravel(), 360.0)
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            ws[i] = s
        full = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        d.createVariable("y_pred", "f4", full)
        d.createVariable("y", "f4", full if truth_members else ("sample", "grid_point_hres", "weather_state"))
        d.createVariable("x_interp", "f4", full if truth_members else ("sample", "grid_point_hres", "weather_state"))
        for k in range(members):
            yp = {"truth": y, "noise": y * rng.lognormal(0, 0.5, n)}[mode]
            d["y_pred"][0, k, :, :] = np.c_[np.zeros(n), yp]
        if truth_members:
            for k in range(members):
                d["y"][0, k, :, :] = np.c_[np.zeros(n), y]
                d["x_interp"][0, k, :, :] = np.c_[np.zeros(n), xi]
        else:
            d["y"][0, :, :] = np.c_[np.zeros(n), y]
            d["x_interp"][0, :, :] = np.c_[np.zeros(n), xi]


def test_fair_crps_matches_pair_formula():
    rng = np.random.default_rng(1)
    e, o = rng.random((7, 11)), rng.random(11)
    brute = np.abs(e - o).mean(0) - np.abs(e[:, None] - e[None]).sum((0, 1)) / (2 * 7 * 6)
    np.testing.assert_allclose(fair_crps(e, o), brute)


def test_scores_exact_when_model_is_truth():
    rng = np.random.default_rng(2)
    t = rng.gamma(0.5, 3.0, 500)
    s = score_sums(np.stack([t, t, t]), 0.5 * t, t, (1.0,))
    p = pooled([{**s, "lead": 24, "date": "x", "region": "all"}])
    assert p["bias_model"] == pytest.approx(0, abs=1e-12)
    assert p["nrmse_model"] == pytest.approx(0, abs=1e-12)
    assert p["corr_model"] == pytest.approx(1)
    assert p["crps_model"] == pytest.approx(0, abs=1e-12)
    assert p["corr_input"] == pytest.approx(1)
    assert p["bias_input"] < 0


@pytest.mark.parametrize("truth_members", [True, False])
def test_run_then_plot(tmp_path, truth_members):
    pred = tmp_path / "pred"
    pred.mkdir()
    for i, date in enumerate(["20230101", "20230111", "20230122"]):
        for step in (24, 48):
            _write(pred / f"predictions_{date}_step{step:03d}.nc", mode="noise", seed=10 * i + step,
                   truth_members=truth_members)
    cfg = {"accum_hours": 12, "bootstrap": 50, "run_label": "test",
           "cases": [{"name": "c1", "label": "Case 1", "date": "20230111", "step": 24, "box": [40, 50, 0, 15]}],
           "map_groups": {"demo": ["c1"]}}
    out = precip_read.run(pred, {}, cfg, output_dir=tmp_path / "out")
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["steps"] == [24, 48] and meta["members"] == [3] and not meta["cases_missing"]
    z = np.load(out / "hist_step024.npz")
    assert z["model"].shape[:2] == (3, 3) and z["truth"].shape[0] == 3
    # every point of every field lands in exactly one bin
    assert (z["truth"].sum(-1)[:, 0] == z["npts"][0]).all()
    precip_read.plot(out, {}, cfg)
    for name in ("tp_pdf_step024.png", "tp_pdf_step048.png", "tp_scores_by_lead.png", "tp_heavy_area_by_lead.png",
                 "tp_maps_demo.png"):
        assert (out / name).exists(), name
