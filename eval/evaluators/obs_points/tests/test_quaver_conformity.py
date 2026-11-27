"""obs_points scores the quaver way (owner rule, 27 Nov 2026): quaver's QC limits (tp up to 900 mm), its
analysis-departure check on each row's own grid, each row on its own station set, density weights over every
observation of the valid time, quaver's spread (ddof 0, RMS over stations) and its means over dates ("wmo":
RMS for rmsef and spread, plain mean for fair CRPS). Synthetic checks here; ``test_matches_real_quaver`` checks
the numbers against quaver 3.8.2's own per-date domain means (fixture from the T1 conformity set, 20230101 +24 h)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.evaluators import obs_points
from eval.evaluators.obs_points.core.scores import ensemble_scores, fair_crps
from eval.evaluators.obs_points.core.stations import (
    analysis_column, density_weights, limits_of, load_cache, region_masks,
)
from eval.evaluators.obs_points.core.summary import summarise
from eval.evaluators.obs_points.runner import row_analysis_columns

from .test_obs_points import _cache, _grid, _pred

FIXTURE = Path(__file__).parent / "data" / "quaver_t1_20230101_L024.json.gz"


def test_quaver_hard_limits_keep_tp_up_to_900_mm(tmp_path):
    assert limits_of("tp") == limits_of("tp6") == limits_of("tp24") == (0.0, 900.0)
    assert limits_of("2t") == (173.0, 333.0) and limits_of("10ff") == (0.0, 100.0)
    pd.DataFrame({"valid": ["2023110300"] * 3, "lat": [48.3, 48.4, 48.5], "lon": [16.1, 16.2, 16.3],
                  "stnid": ["11030", "b", "c"], "elevation": [0.0] * 3,
                  "obs": [703.0, 901.0, -0.1]}).to_csv(tmp_path / "obs_tp.csv", index=False)
    obs = load_cache(tmp_path, ["tp"])[0]["tp"]
    assert obs.stnid.tolist() == ["11030"]  # quaver keeps the 703 mm report of 2023110300


def test_regions_include_their_bounds_like_quaver_domains():
    m = region_masks([69.0, 69.01, 11.0], [39.0, 0.0, -79.0], {"e2box": [11.0, 69.0, -79.0, 39.0]})["e2box"]
    assert m.tolist() == [True, False, True]


def test_spread_is_quavers_and_dates_average_the_wmo_way():
    rng = np.random.default_rng(0)
    ens, obs = rng.standard_normal((10, 500)), rng.standard_normal(500)
    w = rng.uniform(0.5, 1.5, 500)
    s = ensemble_scores(ens, obs, weights=w)
    wn = w / w.sum()
    assert s["spread"] == s["spread_q"] == pytest.approx(np.sqrt((wn * ens.var(0, ddof=0)).sum()), rel=1e-12)
    assert s["spread_rms"] == pytest.approx(np.sqrt((wn * ens.var(0, ddof=1)).sum()), rel=1e-12)
    assert s["spread_sd"] == pytest.approx((wn * ens.std(0, ddof=1)).sum(), rel=1e-12)
    assert s["rmse_mean"] == pytest.approx(np.sqrt((wn * (ens.mean(0) - obs) ** 2).sum()), rel=1e-12)
    assert s["fcrps"] == pytest.approx((wn * fair_crps(ens, obs)).sum(), rel=1e-12)
    keys = dict(var="2t", variant="raw", radius_km=0.0, region="all", lead=24, row="model", kind="ensemble",
                members=10, n=500)
    df = pd.DataFrame([{**keys, "date": d, "fcrps": f, "spread": sp, "spread_q": sp, "spread_rms": sp,
                        "rmse_mean": r, "ssr_corrected": 1.0}
                       for d, f, sp, r in (("a", 1.0, 1.0, 2.0), ("b", 3.0, 3.0, 4.0), ("c", 5.0, 2.0, 1.0))])
    q = summarise(df).iloc[0]
    assert q.fcrps == pytest.approx(3.0)  # plain mean
    assert q.spread == pytest.approx(np.sqrt((1 + 9 + 4) / 3)) and q.rmse_mean == pytest.approx(np.sqrt(21 / 3))


def test_each_row_reads_the_analysis_of_its_own_grid():
    assert analysis_column("O1280") == "an" and analysis_column("o320") == "an_o320"
    cols = row_analysis_columns({"model_height_column": "h_o1280"}, "h_o320_eefh")
    assert cols == {"model": "an", "truth": "an", "input": "an_o320", "hres": "an"}
    assert row_analysis_columns({"model_height_column": "h_o2560"}, "h_o1280")["model"] == "an_o2560"
    assert row_analysis_columns({"model_height_column": "h_o2560"}, "h_o1280")["input"] == "an"


def _with_native_input(pred, la, lo):
    import netCDF4 as nc

    for f in sorted(pred.glob("*.nc")):
        with nc.Dataset(str(f), "a") as d:
            d.createDimension("grid_point_lres", la.size)
            d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = la
            d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = np.mod(lo, 360.0)
            x = d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))
            rng = np.random.default_rng(1)
            base = np.zeros((la.size, 5)); base[:, 0] = 280.0; base[:, 3] = 101000.0 + 10.0 * la
            x[0] = np.stack([base + rng.standard_normal(base.shape) * [1, 1, 1, 50, 0] for _ in range(5)])


def test_rows_are_scored_on_their_own_stations_like_quaver(tmp_path):
    """quaver's europe cell: the model inside its box, a global input on the whole domain. A station the
    analysis check drops for the model grid (an) but keeps for the input grid (an_o320) is the input's only
    (station 60190, 2023050700, 10ff 42 m/s)."""
    cache = _cache(tmp_path)
    o = pd.read_csv(cache / "obs_msl.csv")
    extra = o[o.stnid == 1000].copy()
    extra["lat"], extra["lon"], extra["stnid"] = 65.0, 0.0, 77777  # outside the model grid (30-60 N), inside the input's
    o = pd.concat([o, extra], ignore_index=True)
    o["an"] = o.obs
    o["an_o320"] = o.obs
    o.loc[o.stnid == 1001, "an"] = o.loc[o.stnid == 1001, "obs"] + 5000.0  # gross for the model grid only
    o.to_csv(cache / "obs_msl.csv", index=False)
    pred = _pred(tmp_path, "pr", noise=1.0, seed=2)
    lat1, lon1 = np.arange(25.0, 70.0 + 1e-9, 1.0), np.arange(-25.0, 25.0 + 1e-9, 1.0)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    _with_native_input(pred, LA.ravel(), LO.ravel())
    cfg = {"cache": str(cache), "vars": ["msl"], "max_km": 50.0, "min_stations": 20, "radii": [0],
           "regions": {"all": None}, "height_classes": {}, "box_edge_deg": 0.0, "input_height_column": "h_o320"}
    df = pd.read_csv(obs_points.run(pred, {}, cfg, output_dir=tmp_path / "pr_out", run_label="P")
                     / "scores_per_date.csv")
    n = df.groupby("row").n.unique().to_dict()
    assert list(n["model"]) == [79] and list(n["truth"]) == [79]  # 80 minus 1001 (an); 77777 outside the grid
    assert list(n["input"]) == [81]  # every station, 77777 included, and 1001 passes on the O320 analysis
    dc = pd.read_csv(obs_points.run(pred, {}, {**cfg, "station_set": "common"}, output_dir=tmp_path / "c_out",
                                    run_label="C") / "scores_per_date.csv")
    assert set(dc.n) == {79}


def test_density_weights_run_over_every_observation_of_the_valid_time(tmp_path):
    """quaver's kernel support: all observations of the valid time (a global set), scored or not, then the
    scored stations' weights renormalised (T1: reproduces quaver's psv to 0 on 48 cases)."""
    rng = np.random.default_rng(5)
    lat, lon = rng.uniform(30, 60, 300), rng.uniform(-20, 20, 300)
    plat, plon = np.r_[lat, rng.uniform(30, 32, 200)], np.r_[lon, rng.uniform(-20, -18, 200)]
    w_all, w_sc = density_weights(lat, lon, plat, plon), density_weights(lat, lon)
    assert w_all.sum() == pytest.approx(1.0) and not np.allclose(w_all, w_sc)


@pytest.mark.skipif(not FIXTURE.exists(), reason="real-quaver fixture not in this checkout")
def test_matches_real_quaver():
    """quaver 3.8.2's own per-date domain means (psv) of the T1 set, 20230101 +24 h, GLB450e2_m and INPUT_PF,
    e2box and europe, 2t (height-corrected), 10ff, msl, tp 12 h: obs_points' scores, given quaver's per-station
    member values and observations and quaver's observation population, equal them. Fixture: q_per_station.csv of
    docs epics/misc/support/20261126_quaver_conformity/T1 (Atos nine/t1_20261126), cut to the stations each
    domain mean uses plus the population's coordinates."""
    with gzip.open(FIXTURE, "rt") as fh:
        fx = json.load(fh)
    for case in fx["cases"]:
        ens, obs = np.asarray(case["members"], np.float64), np.asarray(case["obs"], np.float64)
        w = density_weights(case["lat"], case["lon"], fx["population"][case["population"]]["lat"],
                            fx["population"][case["population"]]["lon"])
        s = ensemble_scores(ens, obs, weights=w)
        for score, col in (("fcrps", "fcrps"), ("spread", "spread")):
            assert s[col] == pytest.approx(case["psv"][score], rel=1e-8), (case["key"], score)
