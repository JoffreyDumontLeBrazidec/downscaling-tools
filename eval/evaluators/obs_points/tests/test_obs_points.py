"""obs_points on synthetic files: the ensemble scores match their brute-force definitions and the A0 kit,
run() writes every table, plot() draws every figure from them alone, and the two-run comparison and
the CLI behave."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from eval.evaluators import obs_points
from eval.evaluators.obs_points.core.scores import ensemble_scores, fair_brier, fair_crps, rank_counts
from eval.evaluators.obs_points.core.stations import StationMatcher, resolve_cache
from eval.evaluators.obs_points.core.summary import compare_runs

DATES = ["20230101", "20230111", "20230122"]
STEPS = (24, 48)
KIT = Path(__file__).resolve().parents[4] / "kits" / "20261102-obs-a0" / "score_a0.py"


def _grid():
    lat1 = np.arange(30.0, 60.0 + 1e-9, 0.25)
    lon1 = np.arange(-20.0, 20.0 + 1e-9, 0.25)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    return LA.ravel(), LO.ravel()


def _truth(la, lo, date, step):
    k = (int(date[-2:]) + step) / 10.0
    return {"2t": 280.0 + 10 * np.sin(np.radians(la * 3 + k)), "10u": 5 * np.cos(np.radians(lo * 4 + k)),
            "10v": 3.0 + 0 * la, "msl": 101000.0 + 800 * np.sin(np.radians(lo * 5 - k)),
            "tp": np.clip(0.004 * np.sin(np.radians(la * 7 + lo * 3 + 10 * k)), 0, None)}


def _write(path, date, step, *, noise, members=5, seed=0):
    rng = np.random.default_rng(seed)
    la, lo = _grid()
    n = la.size
    states = ["2t", "10u", "10v", "msl", "tp"]
    t = _truth(la, lo, date, step)
    scale = {"2t": 1.0, "10u": 1.0, "10v": 1.0, "msl": 100.0, "tp": 0.001}
    with nc.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", members)
        d.createDimension("grid_point_hres", n)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = la
        d.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = np.mod(lo, 360.0)
        ws = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            ws[i] = s
        full = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        single = ("sample", "grid_point_hres", "weather_state")
        d.createVariable("y_pred", "f4", full)
        d.createVariable("y", "f4", single)
        d.createVariable("x_interp", "f4", single)
        d["y"][0] = np.stack([t[s] for s in states], -1)
        d["x_interp"][0] = np.stack([t[s] + 0.5 * scale[s] * rng.standard_normal(n) for s in states], -1)
        for k in range(members):
            d["y_pred"][0, k] = np.stack([t[s] + noise * scale[s] * rng.standard_normal(n) for s in states], -1)


def _cache(tmp_path, *, n_st=80, seed=3):
    rng = np.random.default_rng(seed)
    la, lo = _grid()
    out = tmp_path / "cache"
    out.mkdir()
    slat, slon = rng.uniform(31, 59, n_st), rng.uniform(-19, 19, n_st)
    m = StationMatcher(la, lo)
    idx, _ = m.match(slat, slon, 100.0)
    rows = {v: [] for v in ("2t", "10ff", "msl", "tp")}
    hres = []
    for date in DATES:
        for step in STEPS:
            valid = (pd.Timestamp(date) + pd.Timedelta(hours=step)).strftime("%Y%m%d%H")
            t = _truth(la[idx], lo[idx], date, step)
            vals = {"2t": t["2t"], "10ff": np.hypot(t["10u"], t["10v"]), "msl": t["msl"], "tp": t["tp"] * 1000.0}
            for v, x in vals.items():
                noise = {"2t": 1.0, "10ff": 1.0, "msl": 100.0, "tp": 1.0}[v]
                obs = x + noise * rng.standard_normal(n_st)
                if v == "tp":
                    obs = np.clip(obs, 0, None)
                for k in range(n_st):
                    rows[v].append({"valid": valid, "lat": slat[k], "lon": slon[k], "stnid": 1000 + k,
                                    "elevation": 50.0, "obs": obs[k], "h_o1280": 50.0 + (k % 3) * 200.0,
                                    "h_o320": 60.0})
            for k in range(n_st - 5):  # five stations without HRES: they leave every row's sample
                hres.append({"date": date, "lead": step, "stnid": 1000 + k, "hres": vals["2t"][k] + 0.3,
                             "hres_elevation": 55.0})
    rows["2t"].append({"valid": "2023010200", "lat": 45, "lon": 0, "stnid": 9999, "elevation": 0, "obs": 999.0,
                       "h_o1280": 0, "h_o320": 0})  # outside quaver's hard limits: dropped
    for v, r in rows.items():
        pd.DataFrame(r).to_csv(out / f"obs_{v}.csv", index=False)
    pd.DataFrame(hres).to_csv(out / "hres_2t.csv", index=False)
    (out / "cache_meta.json").write_text(json.dumps({"cache_version": 1, "dates": DATES}))
    return out


def _pred(tmp_path, name, noise, seed):
    pred = tmp_path / name
    pred.mkdir()
    for i, date in enumerate(DATES):
        for step in STEPS:
            _write(pred / f"predictions_{date}_step{step:03d}.nc", date, step, noise=noise, seed=seed + 10 * i + step)
    return pred


# ----------------------------------------------------------------------------- scores
def test_fair_scores_match_brute_force():
    rng = np.random.default_rng(1)
    e, o = rng.gamma(0.7, 3.0, (6, 40)), rng.gamma(0.7, 3.0, 40)
    brute = np.abs(e - o).mean(0) - np.abs(e[:, None] - e[None]).sum((0, 1)) / (2 * 6 * 5)
    np.testing.assert_allclose(fair_crps(e, o), brute)
    t = 2.0
    p, y = (e > t).mean(0), (o > t).astype(float)
    np.testing.assert_allclose(fair_brier(e, o, t), (p - y) ** 2 - p * (1 - p) / 5)
    s = ensemble_scores(e, o, (0.0,))
    assert s["twcrps0"] == pytest.approx(s["fcrps"])  # threshold below every value: twCRPS is the CRPS
    one = ensemble_scores(o[None] + 1.0, o, (2.0,))
    assert one["fcrps"] == pytest.approx(1.0) and one["spread"] == 0.0 and np.isnan(one["ssr_corrected"])
    w, s2 = ensemble_scores(e, o, (2.0,), weights=np.ones(40)), ensemble_scores(e, o, (2.0,))
    for k in ("fcrps", "spread", "brier2", "twcrps2", "rmse_mean", "corr_mean"):
        assert w[k] == pytest.approx(s2[k]), k


def test_calibrated_ensemble_has_unit_spread_skill_and_flat_ranks():
    rng = np.random.default_rng(7)
    mu = rng.standard_normal(200000)
    ens = mu + rng.standard_normal((9, mu.size))
    obs = mu + rng.standard_normal(mu.size)
    s = ensemble_scores(ens, obs)
    assert s["ssr_corrected"] == pytest.approx(1.0, abs=0.01)
    c = rank_counts(ens, obs)
    assert c.sum() == mu.size and c.size == 10
    assert np.abs(c / c.mean() - 1).max() < 0.05


@pytest.mark.skipif(not KIT.exists(), reason="A0 kit not in this checkout")
def test_matches_the_a0_kit():
    spec = importlib.util.spec_from_file_location("score_a0", KIT)
    a0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a0)
    rng = np.random.default_rng(4)
    ens, o = rng.gamma(0.5, 6.0, (10, 300)), rng.gamma(0.5, 6.0, 300)
    mine = ensemble_scores(ens, o, a0.TP_THRESHOLDS)
    ref = {**a0.scores(ens.mean(0), o, ens), **a0.tail_scores(ens.mean(0), o, ens)}
    pairs = {"fcrps": "crps", "spread": "spread", "ssr_corrected": "ssr_corr", "bias": "bias", "rmse_mean": "rmse",
             "mae_mean": "mae", "mae_member": "member_mae", "corr_mean": "corr"}
    pairs.update({k: k for k in ref if k.startswith(("twcrps", "brier", "n_obs_gt"))})
    for k, r in pairs.items():
        assert mine[k] == pytest.approx(ref[r], rel=1e-12), k
    single = ensemble_scores(o[None] * 0.9, o, a0.TP_THRESHOLDS)
    ref1 = {**a0.scores(o * 0.9, o), **a0.tail_scores(o * 0.9, o)}
    for k in ("twcrps5", "brier10", "n_obs_gt20"):
        assert single[k] == pytest.approx(ref1[k], rel=1e-12)
    assert single["fcrps"] == pytest.approx(ref1["crps"])


# ----------------------------------------------------------------------------- run, plot, compare
@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("obs_points")
    cache = _cache(tmp)
    cfg = {"cache": {"jupiter": str(tmp / "absent"), "atos": str(cache)}, "max_km": 100.0, "min_stations": 20,
           "radii": [0, 40], "regions": {"all": None, "north": [45, 60, -20, 20]}}
    outs = {}
    for name, noise, seed in (("good", 1.0, 0), ("bad", 3.0, 100)):
        pred = _pred(tmp, name, noise, seed)
        outs[name] = obs_points.run(pred, {}, cfg, output_dir=tmp / f"out_{name}", run_label=name.upper())
    return tmp, cfg, outs


def test_run_writes_tables(two_runs):
    _, _, outs = two_runs
    out = outs["good"]
    for f in ("scores_per_date.csv", "rank_hist.csv", "summary.csv", "paired.csv", "summary.md", "run_meta.json"):
        assert (out / f).exists(), f
    df = pd.read_csv(out / "scores_per_date.csv", dtype={"date": str})
    assert set(df.row) == {"model", "input", "truth", "hres"}
    assert set(df[df.row != "model"].kind) == {"single field"} and set(df[df.row == "model"].kind) == {"ensemble"}
    assert set(df[df.row == "hres"]["var"]) == {"2t"}
    assert {"raw", "lapse"} <= set(df[df["var"] == "2t"].variant)
    assert set(df[df["var"] == "tp"].radius_km) == {0.0, 40.0}
    # one station sample per (var, variant, radius, region, lead, date): the same n for every row
    assert (df.groupby(["var", "variant", "radius_km", "region", "lead", "date"]).n.nunique() == 1).all()
    assert (df[(df["var"] == "2t") & (df.region == "all")].n == 75).all()  # 80 minus 5 without HRES; 999 K dropped
    assert (df[(df["var"] == "msl") & (df.region == "all")].n == 80).all()
    # single fields: fair CRPS is the MAE, spread 0
    single = df[df.kind == "single field"]
    np.testing.assert_allclose(single.fcrps, single.mae_mean)
    assert (single.spread == 0).all()
    tp = df[(df["var"] == "tp") & (df.row == "model")]
    assert {"twcrps5", "brier5", "n_obs_gt5", "twcrps20"} <= set(tp.columns)
    rk = pd.read_csv(out / "rank_hist.csv")
    assert set(rk.row) == {"model"}
    np.testing.assert_array_equal(rk[[f"r{i}" for i in range(6)]].sum(1), rk.n)
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["run_label"] == "GOOD" and meta["members"] == [5] and meta["cache_dir"].endswith("cache")
    md = (out / "summary.md").read_text()
    assert "single field" in md and "fair CRPS" in md


def test_plot_from_saved_files_only(two_runs):
    _, cfg, outs = two_runs
    out = outs["good"]
    written = obs_points.plot(out, {}, cfg)
    names = {Path(p).name for p in written}
    for stem in ("obs_2t_raw_scores", "obs_2t_lapse_scores", "obs_10ff_raw_scores", "obs_msl_raw_scores",
                 "obs_tp_raw_scores", "obs_tp_raw_r40km_scores", "obs_tp_tail", "obs_tp_tail_r40km",
                 "obs_rank_hist_2t_lapse", "obs_rank_hist_tp_raw"):
        assert f"{stem}.png" in names, stem


def test_good_ensemble_beats_noisy_one_and_compare(two_runs, tmp_path):
    _, _, outs = two_runs
    a = pd.read_csv(outs["good"] / "scores_per_date.csv", dtype={"date": str})
    b = pd.read_csv(outs["bad"] / "scores_per_date.csv", dtype={"date": str})
    tab, checks = compare_runs(a, b)
    assert checks["only_in_a"] == 0 and checks["n_mismatch"] == 0
    assert checks["truth_fcrps_max_abs_diff"] == pytest.approx(0.0, abs=1e-6)
    f = tab[(tab.metric == "fcrps") & (tab["var"] == "msl") & (tab.region == "all")]
    assert (f["mean"] < 0).all() and (f.wins == f.dates).all() and (f.dates == 3).all()

    from eval.cli import main
    main(["obs", "compare", "--a", str(outs["good"]), "--b", str(outs["bad"]), "--out", str(tmp_path / "cmp")])
    for name in ("paired_runs.csv", "paired_runs.md", "compare_meta.json", "paired_runs.png"):
        assert (tmp_path / "cmp" / name).exists(), name
    meta = json.loads((tmp_path / "cmp" / "compare_meta.json").read_text())
    assert meta["label_a"] == "GOOD" and meta["label_b"] == "BAD"
    main(["obs", "compare", "--a", "x", "--b", "y", "--out", str(tmp_path / "cmp"), "--plot-only"])


def test_station_density_weighting_runs(two_runs, tmp_path):
    tmp, cfg, _ = two_runs
    out = obs_points.run(tmp / "good", {}, {**cfg, "weighting": "station_density", "vars": ["msl"]},
                         output_dir=tmp_path / "w")
    df = pd.read_csv(out / "scores_per_date.csv")
    assert set(df["var"]) == {"msl"} and np.isfinite(df.fcrps).all()
    with pytest.raises(ValueError):
        obs_points.run(tmp / "good", {}, {**cfg, "weighting": "area"}, output_dir=tmp_path / "w2")


def test_resolve_cache_takes_the_first_existing(tmp_path):
    (tmp_path / "b").mkdir()
    assert resolve_cache([str(tmp_path / "a"), str(tmp_path / "b")]) == tmp_path / "b"
    with pytest.raises(FileNotFoundError):
        resolve_cache({"x": str(tmp_path / "nope")})


def test_obs_fetch_refuses_off_atos(monkeypatch, tmp_path):
    import socket

    from eval.cli import main
    monkeypatch.setattr(socket, "gethostname", lambda: "jpbl-s01-02")
    with pytest.raises(SystemExit, match="Atos only"):
        main(["obs", "fetch", "--dates", "20230101", "--out", str(tmp_path / "c")])
