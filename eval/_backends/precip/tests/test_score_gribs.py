"""End-to-end test of the GRIB-route scorer with injected readers."""
from __future__ import annotations

import json

import numpy as np

from eval._backends.precip import score_gribs as SG
from eval._backends.precip.sources import LresInterpBaseline, PrecipTruthSource

N = 200
LATS = np.linspace(-10.0, 10.0, N)
LONS = np.linspace(100.0, 120.0, N)
RNG = np.random.default_rng(5)
TRUTH = {s: RNG.gamma(0.5, 0.002, size=N) for s in (6, 12)}


def _truth_reader(date):
    return {(0, s): v.copy() for s, v in TRUTH.items()}, LATS, LONS


def _model_reader(path, var):
    # model = truth + per-member bias (member 1: +1mm, member 2: -1mm)
    out = {}
    for member, shift in ((1, 0.001), (2, -0.001)):
        for s, v in TRUTH.items():
            out[(member, s)] = v + shift
    return out, LATS, LONS


def test_grib_route_matches_known_bias(tmp_path, monkeypatch):
    monkeypatch.setattr(SG, "_read_grib_var", _model_reader)
    monkeypatch.setattr(
        SG, "PrecipTruthSource",
        lambda tpl, var="tp": PrecipTruthSource(tpl, var=var, _reader=_truth_reader))

    out = SG.score_gribs(
        model_grib_tpl="/nowhere/model_{date}.grib",
        truth_grib_tpl="/nowhere/truth_{date}.grib",
        dates=["20250926"],
        out_dir=tmp_path / "out",
        run_label="grib-route test",
    )
    payload = json.loads((out / "scores.json").read_text())
    assert payload["meta"]["route"].startswith("grib")
    assert payload["meta"]["n_slices"] == 2
    row = payload["rows"][0]
    m1, m2 = row["members"]
    assert abs(m1["model"]["bias_mm"] - 1.0) < 1e-9
    assert abs(m2["model"]["bias_mm"] + 1.0) < 1e-9
    # +1mm and -1mm members average back to truth
    assert row["model_ens_mean"]["rmse_mm"] < 1e-9
    assert abs(payload["summary"]["model_rmse_mm"] - 1.0) < 1e-9
    assert (out / "plots" / "precip_scores.pdf").stat().st_size > 1024
