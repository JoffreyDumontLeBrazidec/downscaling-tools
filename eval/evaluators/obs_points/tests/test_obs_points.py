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
    # audit 5 Nov 2026: the file's rows carry no input or truth members, so the competitor rows are single fields
    # and say so; the model members are draws from one input (x_interp has no member axis), so the provenance
    # block and the log say "single-input draws"
    prov = meta["provenance"]
    assert prov["against_observations"] and prov["status"] == "headline"
    assert prov["ensemble_construction"].startswith("single-input draws")
    assert prov["competitor_rows"]["input"].startswith("single field")
    assert any("single-input" in w or "ONE input" in w for w in prov["warnings"])
    assert any("single field" in w for w in meta["warnings"])
    assert prov["date_spacing"]["n_dates"] == 3 and not prov["date_spacing"]["consecutive"]
    assert "Provenance" in md and "Verifying field: station observations" in md


def _write_member_mode(path, date, step, *, members=5, seed=0, truth_members=False):
    """A file of the ENS-member kind (4 Nov 2026): member k conditioned on input member k, x_interp_members with
    the members, member_mode=pf; optionally y_members (the truth ensemble)."""
    _write(path, date, step, noise=1.0, members=members, seed=seed)
    rng = np.random.default_rng(seed + 7)
    la, lo = _grid()
    n = la.size
    states = ["2t", "10u", "10v", "msl", "tp"]
    t = _truth(la, lo, date, step)
    scale = {"2t": 1.0, "10u": 1.0, "10v": 1.0, "msl": 100.0, "tp": 0.001}
    with nc.Dataset(str(path), "r+") as d:
        full = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        d.createVariable("x_interp_members", "f4", full)
        for k in range(members):
            d["x_interp_members"][0, k] = np.stack([t[s] + 0.5 * scale[s] * rng.standard_normal(n) for s in states], -1)
        if truth_members:
            d.createVariable("y_members", "f4", full)
            for k in range(members):
                d["y_members"][0, k] = np.stack([t[s] + 0.3 * scale[s] * rng.standard_normal(n) for s in states], -1)
        d.setncattr("member_mode", "pf")
        d.setncattr("seed_rule", "per_case")


def test_input_and_truth_ensembles_are_scored_as_ensembles(tmp_path):
    """Audit 5 Nov 2026 (D4, rule 34(e)): when the files carry the input members (x_interp_members) and the
    truth members (y_members), those rows are scored as ensembles on the same stations (fair CRPS, spread, rank
    counts), labelled so, and the model row is 'one draw per input member'."""
    cache = _cache(tmp_path)
    pred = tmp_path / "pred"
    pred.mkdir()
    for i, date in enumerate(DATES):
        for step in STEPS:
            _write_member_mode(pred / f"predictions_{date}_step{step:03d}.nc", date, step, seed=10 * i + step,
                               truth_members=True)
    cfg = {"cache": str(cache), "vars": ["2t", "msl"], "max_km": 100.0, "min_stations": 20, "radii": [0],
           "regions": {"all": None}, "height_classes": {}}
    out = obs_points.run(pred, {}, cfg, output_dir=tmp_path / "out", run_label="M")
    df = pd.read_csv(out / "scores_per_date.csv", dtype={"date": str})
    kinds = df.groupby("row").kind.first()
    assert kinds["model"] == "ensemble" and kinds["hres"] == "single field"
    assert kinds["input"] == "ensemble (input members)" and kinds["truth"] == "ensemble (truth members)"
    ens = df[df.row.isin(["input", "truth"])]
    assert (ens.members == 5).all() and (ens.spread > 0).all()
    assert not np.allclose(ens.fcrps, ens.mae_mean)  # an ensemble's fair CRPS is not its mean's MAE
    rk = pd.read_csv(out / "rank_hist.csv")
    assert set(rk.row) == {"model", "input", "truth"}
    meta = json.loads((out / "run_meta.json").read_text())
    prov = meta["provenance"]
    assert prov["ensemble_construction"].startswith("one draw per input member")
    assert prov["competitor_rows"]["input"].startswith("ensemble") and prov["competitor_rows"]["truth"].startswith("ensemble")
    assert not any("single-input" in w for w in prov["warnings"])
    md = (out / "summary.md").read_text()
    assert "input (ensemble)" in md and "truth (ensemble)" in md and "hres (single field)" in md
    # the lapse variant applies the O320 height to every input member
    lapse = df[(df["var"] == "2t") & (df.variant == "lapse") & (df.row == "input")]
    assert len(lapse) == len(DATES) * len(STEPS)
    # the opt-out scores the control fields as before
    out2 = obs_points.run(pred, {}, {**cfg, "competitor_ensembles": False}, output_dir=tmp_path / "out2")
    df2 = pd.read_csv(out2 / "scores_per_date.csv", dtype={"date": str})
    assert set(df2[df2.row.isin(["input", "truth"])].kind) == {"single field"}
    # the two-run comparison still pairs the model rows
    tab, checks = compare_runs(df, df2)
    assert checks["only_in_a"] == 0 and len(tab)


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
    (tmp_path / "cmp" / "paired_runs.png").unlink()
    main(["obs", "compare", "--out", str(tmp_path / "cmp"), "--plot-only"])
    assert (tmp_path / "cmp" / "paired_runs.png").exists()  # redrawn from paired_runs.csv alone
    with pytest.raises(SystemExit, match="--a and --b"):
        main(["obs", "compare", "--out", str(tmp_path / "cmp2")])


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


# ----------------------------------------------------------------------------- SEEPS
def test_seeps_matches_quaver_indexing():
    from eval.evaluators.obs_points.core.scores import seeps

    smat = np.arange(9, dtype=float)[None].repeat(3, 0)  # entry k = k, so the result is the index
    t1, t2 = np.full(3, 0.2), np.full(3, 5.0)
    obs = np.array([0.0, 1.0, 9.0])          # dry, light, heavy
    fc = np.array([[0.24, 0.26, 5.0],        # rounds to 0.2 (dry), 0.3 (light), 5.0 (heavy, >= t2)
                   [7.0, 0.0, 1.0]])
    np.testing.assert_array_equal(seeps(fc, obs, t1, t2, smat), [[0, 4, 8], [6, 1, 5]])


def test_seeps_on_24h_windows(tmp_path):
    rng = np.random.default_rng(9)
    la, lo = _grid()
    cache = tmp_path / "cache"
    cache.mkdir()
    n_st = 60
    slat, slon = rng.uniform(31, 59, n_st), rng.uniform(-19, 19, n_st)
    date = "20230101"
    rows, rows24, clim = [], [], []
    for step in (12, 24):
        valid = (pd.Timestamp(date) + pd.Timedelta(hours=step)).strftime("%Y%m%d%H")
        for k in range(n_st):
            rows.append({"valid": valid, "lat": slat[k], "lon": slon[k], "stnid": k, "elevation": 0.0,
                         "obs": rng.gamma(0.5, 4.0), "h_o1280": 0.0, "h_o320": 0.0})
    valid24 = "2023010200"
    for k in range(n_st):
        rows24.append({"valid": valid24, "lat": slat[k], "lon": slon[k], "stnid": k, "elevation": 0.0,
                       "obs": rng.gamma(0.5, 8.0), "h_o1280": 0.0, "h_o320": 0.0})
        p1 = 0.05 if k < 10 else 0.5  # ten stations fail quaver's seeps_filter
        clim.append({"valid": valid24, "stnid": k, "lat": slat[k], "lon": slon[k], "p1": p1, "t1": 0.2, "t2": 4.0,
                     **{f"s{j}": float(j != 0 and j != 4 and j != 8) for j in range(9)}})
    pd.DataFrame(rows).to_csv(cache / "obs_tp.csv", index=False)
    pd.DataFrame(rows24).to_csv(cache / "obs_tp24.csv", index=False)
    pd.DataFrame(clim).to_csv(cache / "seeps_clim_tp24.csv", index=False)
    pred = tmp_path / "pred"
    pred.mkdir()
    for step in (12, 24):
        _write(pred / f"predictions_{date}_step{step:03d}.nc", date, step, noise=1.0, seed=step)
    cfg = {"cache": str(cache), "vars": ["tp"], "max_km": 100.0, "min_stations": 20, "radii": [0],
           "regions": {"all": None}, "height_classes": {}}
    out = obs_points.run(pred, {}, cfg, output_dir=tmp_path / "out", run_label="S")
    df = pd.read_csv(out / "scores_per_date.csv")
    t24 = df[df["var"] == "tp24"]
    assert set(t24.lead) == {24} and (t24.n == 50).all()  # the ten filtered stations are left out
    assert t24.seeps.between(0, 1).all() and df[df["var"] == "tp"].seeps.isna().all()
    # recompute the truth row's SEEPS by hand: 24 h = the +12 and +24 files summed
    from eval.evaluators.obs_points.runner import read_fields
    f12 = read_fields(pred / f"predictions_{date}_step012.nc", ["tp"])
    f24 = read_fields(pred / f"predictions_{date}_step024.nc", ["tp"])
    m = StationMatcher(f12[0], f12[1])
    idx, _ = m.match(slat[10:], slon[10:], 100.0)
    truth = (f12[2]["tp"]["truth"] + f24[2]["tp"]["truth"])[idx]
    ob = np.round(np.array([r["obs"] for r in rows24[10:]]), 1)
    fc = np.round(truth, 1)
    cat = lambda x: (x > 0.2).astype(int) + (x >= 4.0).astype(int)  # noqa: E731
    expected = (cat(fc) != cat(ob)).mean()
    assert t24[t24.row == "truth"].seeps.iloc[0] == pytest.approx(expected)
    obs_points.plot(out, {}, cfg)
    assert (out / "obs_tp24_raw_scores.png").exists()


def test_seeps_warns_when_a_12h_file_is_missing(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = [{"valid": v, "lat": 45.0 + k * 0.1, "lon": 0.0, "stnid": k, "elevation": 0.0, "obs": 1.0,
             "h_o1280": 0.0, "h_o320": 0.0} for v in ("2023010112", "2023010212") for k in range(40)]
    pd.DataFrame(rows).to_csv(cache / "obs_tp.csv", index=False)
    pd.DataFrame(rows).to_csv(cache / "obs_tp24.csv", index=False)
    pd.DataFrame([{"valid": "2023010212", "stnid": k, "lat": 45.0, "lon": 0.0, "p1": 0.5, "t1": 0.2, "t2": 4.0,
                   **{f"s{j}": 0.0 for j in range(9)}} for k in range(40)]).to_csv(cache / "seeps_clim_tp24.csv",
                                                                                  index=False)
    pred = tmp_path / "pred"
    pred.mkdir()
    for step in (12, 36):  # +24 h is missing: no 24 h window ending at +36 h
        _write(pred / f"predictions_20230101_step{step:03d}.nc", "20230101", step, noise=1.0, seed=step)
    out = obs_points.run(pred, {}, {"cache": str(cache), "vars": ["tp"], "radii": [0], "min_stations": 10,
                                    "regions": {"all": None}, "height_classes": {}}, output_dir=tmp_path / "out")
    warn = json.loads((out / "run_meta.json").read_text())["warnings"]
    assert any("no +24 h file" in w for w in warn) and any("not 12 h apart" in w for w in warn)


def test_seeps_climatology_is_joined_by_station(monkeypatch):
    """The 13 climatology fields are matched on station id, not on row position."""
    import sys
    import types

    from eval.evaluators.obs_points.core import fetch

    class F:
        def __init__(self, ids, vals):
            self.df = pd.DataFrame({"latitude": 45.0, "longitude": 5.0, "stnid": ids, "elevation": 0.0,
                                    "value_0": vals})

        def to_dataframe(self):
            return self.df.copy()

    class FS(list):
        header = {"number": list(range(13))}

    ids = ["a", "b", "c"]
    fields = FS(F(ids if k != 3 else ids[::-1], [10.0 * k + i for i in range(3)][::(-1 if k == 3 else 1)])
                for k in range(13))
    vm = types.SimpleNamespace(stvl_retrieve=lambda **kw: fields)
    monkeypatch.setitem(sys.modules, "vtb", types.SimpleNamespace(media=vm))
    monkeypatch.setitem(sys.modules, "vtb.media", vm)
    import datetime as dt
    c = fetch.seeps_clim_at(dt.datetime(2023, 1, 2), (0, 90, -180, 180)).set_index("stnid")
    assert list(c.t2) == [30.0, 31.0, 32.0] and list(c.t1) == [10.0, 11.0, 12.0] and list(c.s0) == [40.0, 41.0, 42.0]


# ----------------------------------------------------------------------------- A0 audits
@pytest.mark.parametrize("swap", [False, True])
def test_tp_window_check(tmp_path, swap):
    """Station tp equal to the file's 12 h tp (+0.5 mm) in the window ending at valid time, noise in the window
    before; with swap the two are exchanged. The two windows are reported by different station sets (60 common,
    20 only at valid time, 15 only the window before): both correlations use the 60 common stations."""
    from eval.evaluators.obs_points.runner import read_fields

    date, step = "20230101", 24
    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / f"predictions_{date}_step{step:03d}.nc", date, step, noise=0.5, seed=1)
    la, lo, f, _ = read_fields(pred / f"predictions_{date}_step{step:03d}.nc", ["tp"])
    rng = np.random.default_rng(2)
    n_st = 95
    slat, slon = rng.uniform(31, 59, n_st), rng.uniform(-19, 19, n_st)
    idx, _ = StationMatcher(la, lo).match(slat, slon, 100.0)
    truth = f["tp"]["truth"][idx]
    exact, noise = truth + 0.5, rng.gamma(0.5, 2.0, n_st)
    at_valid, at_prev = np.arange(0, 80), np.r_[np.arange(0, 60), np.arange(80, 95)]
    rows = []
    for valid, vals, which in (("2023010200", noise if swap else exact, at_valid),
                               ("2023010112", exact if swap else noise, at_prev)):
        rows += [{"valid": valid, "lat": slat[k], "lon": slon[k], "stnid": k, "elevation": 0.0, "obs": vals[k],
                  "h_o1280": 0.0, "h_o320": 0.0} for k in which]
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame(rows).to_csv(cache / "obs_tp.csv", index=False)
    pd.DataFrame([{"date": date, "lead": step, "stnid": k, "hres": 0.001, "hres_elevation": 0.0}
                  for k in at_valid]).to_csv(cache / "hres_tp.csv", index=False)
    out = obs_points.run(pred, {}, {"cache": str(cache), "vars": ["tp"], "radii": [0], "max_km": 100.0,
                                    "min_stations": 10, "regions": {"all": None}, "height_classes": {}},
                         output_dir=tmp_path / "out")
    w = pd.read_csv(out / "tp_window_check.csv")
    t = w[w.row == "truth"].iloc[0]
    common = np.arange(60)
    assert set(w.row) == {"truth", "input", "model"} and t.n == 60
    assert t.corr_valid == pytest.approx(np.corrcoef(truth[common], (noise if swap else exact)[common])[0, 1])
    assert t.corr_prev == pytest.approx(np.corrcoef(truth[common], (exact if swap else noise)[common])[0, 1])
    assert (t.corr_valid == pytest.approx(1.0)) != swap and bool(t.prev_better) == swap
    warn = json.loads((out / "run_meta.json").read_text())["warnings"]
    assert any("WARN tp window" in x and "truth" in x for x in warn) == swap
    assert any("looks like metres" in x for x in warn)  # HRES tp of 0.001 against mm observations
    s = pd.read_csv(out / "tp_window_summary.csv")
    assert (s.dates == 1).all()


def test_tp_window_check_skips(tmp_path):
    from eval.evaluators.obs_points.runner import tp_window_check

    la, lo = _grid()
    m = StationMatcher(la, lo)
    fields = {"truth": np.arange(la.size, dtype=float), "input": np.ones(la.size), "model": np.ones((2, la.size))}
    cfg = {"max_km": 100.0, "min_stations": 5}
    obs = pd.DataFrame({"valid": ["2023010200"] * 8 + ["2023010112"] * 8, "stnid": list(range(8)) * 2,
                        "lat": list(np.linspace(35, 55, 8)) * 2, "lon": 0.0, "obs": np.r_[np.arange(8.0), np.arange(8.0)[::-1]]})
    log = []
    rows = tp_window_check(fields, obs, m, date="20230101", lead=24, cfg=cfg, log=log)
    assert [r["row"] for r in rows] == ["truth"]  # input and model are constant: skipped
    assert sum("constant series" in x for x in log) == 2
    log = []
    assert tp_window_check(fields, obs[obs.valid == "2023010200"], m, date="20230101", lead=24, cfg=cfg, log=log) == []
    assert "window audit skipped" in log[0] and "no station tp at 2023010112" in log[0]
    log = []
    assert tp_window_check(fields, obs, m, date="20230101", lead=24, cfg={**cfg, "min_stations": 20}, log=log) == []
    assert "8 stations report both windows" in log[0]


def test_hres_metres_rule():
    from eval.evaluators.obs_points.runner import hres_tp_in_metres

    obs = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 8.0, 15.0])
    assert hres_tp_in_metres(obs / 1000.0, obs)
    assert not hres_tp_in_metres(obs * 0.9, obs)
    assert not hres_tp_in_metres(np.full(7, np.nan), obs)
    dry = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.4, 0.0])  # median 0 and no report above 1 mm: nothing to decide
    assert not hres_tp_in_metres(dry, dry) and not hres_tp_in_metres(dry / 1000.0, dry)
    mostly_dry = np.r_[np.zeros(50), [5.0, 12.0]]  # a dry median does not make correct mm HRES look like metres
    assert not hres_tp_in_metres(mostly_dry * 0.8, mostly_dry)


def test_analysis_departure_check_matches_quaver_toss(tmp_path):
    """quaver toss_with_analysis: |obs - an| <= 30 K / 40 m/s / 1000 Pa kept; caches without 'an' untouched."""
    from eval.evaluators.obs_points.core.stations import analysis_departure_ok, load_cache

    o = pd.DataFrame({"valid": ["2024110112"] * 4, "lat": [41.1, 50, 51, 52], "lon": [23.2, 0, 1, 2],
                      "stnid": ["a", "b", "c", "d"], "elevation": [50.0] * 4,
                      "obs": [90840.0, 101000.0, 101500.0, 100000.0], "an": [101200.0, 101100.0, 100500.0, np.nan]})
    assert analysis_departure_ok(o, "msl").tolist() == [False, True, True, True]
    assert analysis_departure_ok(o.drop(columns="an"), "msl").all()
    assert analysis_departure_ok(o, "tp").all()
    w = pd.DataFrame({"obs": [80.0, 5.0], "an": [3.0, 4.0]})
    assert analysis_departure_ok(w, "10ff").tolist() == [False, True]
    o.to_csv(tmp_path / "obs_msl.csv", index=False)
    obs, _, _ = load_cache(tmp_path, ["msl"])
    assert obs["msl"].stnid.tolist() == ["b", "c", "d"]


def test_input_is_scored_on_its_native_grid(tmp_path):
    """Rule 41: with a native ``x`` on lat_lres/lon_lres the input row is read there (nearest native point),
    never from x_interp; without it the run warns that the input row is interpolated."""
    from eval.evaluators.obs_points.runner import input_height_column, read_fields

    pred = _pred(tmp_path, "native", noise=0.5, seed=7)
    f = sorted(pred.glob("*.nc"))[0]
    lat1, lon1 = np.arange(30.0, 60.0 + 1e-9, 1.0), np.arange(-20.0, 20.0 + 1e-9, 1.0)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    la, lo = LA.ravel(), LO.ravel()
    with nc.Dataset(str(f), "a") as d:
        d.createDimension("grid_point_lres", la.size)
        d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = la
        d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = np.mod(lo, 360.0)
        x = d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))
        base = np.zeros((la.size, 5)); base[:, 0] = 250.0  # native 2t = 250 K everywhere, x_interp is ~280 K
        x[0] = np.stack([base + k for k in range(5)])
    _, _, fields, prov = read_fields(f, ["2t"])
    assert prov["input_grid"]["source"].startswith("x (native")
    assert fields["2t"]["input"].shape == (5, la.size)
    np.testing.assert_allclose(fields["2t"]["input"][:, 0], 250.0 + np.arange(5))
    _, _, fields_old, prov_old = read_fields(f, ["2t"], native_input=False)
    assert "interpolated" in prov_old["input_grid"]["source"]
    assert fields_old["2t"]["input"].shape[-1] == fields_old["2t"]["model"].shape[-1]
    assert input_height_column(la, lo) == "h_o320"  # 1-degree spacing: the coarse column


def test_native_input_row_drops_the_box_edge_strip_stations_too(tmp_path):
    """Rule 41 with the box-edge strip (docs ablation note s21): stations dropped for the strip of the output
    grid are dropped for the native input row as well, so input and model rows share one station sample."""
    cache = _cache(tmp_path)
    pred = _pred(tmp_path, "native", noise=1.0, seed=5)
    lat1, lon1 = np.arange(29.0, 61.0 + 1e-9, 0.5), np.arange(-21.0, 21.0 + 1e-9, 0.5)
    LA, LO = np.meshgrid(lat1, lon1, indexing="ij")
    la, lo = LA.ravel(), LO.ravel()
    for f in sorted(pred.glob("*.nc")):
        with nc.Dataset(str(f), "a") as d:
            d.createDimension("grid_point_lres", la.size)
            d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = la
            d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = np.mod(lo, 360.0)
            x = d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))
            base = np.zeros((la.size, 5)); base[:, 0] = 280.0; base[:, 3] = 101000.0
            x[0] = np.stack([base] * 5)
    cfg = {"cache": str(cache), "vars": ["msl"], "max_km": 100.0, "min_stations": 20, "radii": [0],
           "regions": {"all": None}, "height_classes": {}, "box_edge_deg": 3.0}  # wide strip: stations sit >= 1 deg in
    out = obs_points.run(pred, {}, cfg, output_dir=tmp_path / "o1", run_label="N")
    df = pd.read_csv(out / "scores_per_date.csv", dtype={"date": str})
    assert (df.groupby(["var", "variant", "radius_km", "region", "lead", "date"]).n.nunique() == 1).all()
    n_strip = int(df[df.row == "input"].n.iloc[0])
    out0 = obs_points.run(pred, {}, {**cfg, "box_edge_deg": 0.0}, output_dir=tmp_path / "o0", run_label="N0")
    n_all = int(pd.read_csv(out0 / "scores_per_date.csv").query("row == \"input\"").n.iloc[0])
    assert n_strip < n_all == 80   # the strip drops stations, for the input row as for the model row
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["input_source"] == ["x (native input grid)"]


def test_tp_obs_var_scores_tp_against_the_named_series(tmp_path):
    """tp_obs_var: tp6 scores the file's tp against obs_tp6.csv (the 4 km lane's 6 h windows), not obs_tp.csv."""
    import shutil

    cache = _cache(tmp_path)
    pred = _pred(tmp_path, "p6", noise=0.5, seed=11)
    cfg = {"vars": ["tp"], "radii": [0], "min_stations": 10, "max_km": 100.0}
    ref = obs_points.run(pred, {}, {**cfg, "cache": str(cache)}, output_dir=tmp_path / "ref", run_label="R")
    c6 = tmp_path / "cache6"
    shutil.copytree(cache, c6)
    (c6 / "obs_tp.csv").rename(c6 / "obs_tp6.csv")
    o = pd.read_csv(c6 / "obs_tp6.csv")
    o["obs"] = o["obs"] * 7.0 + 3.0  # a different 12 h series that must NOT be used
    o.to_csv(c6 / "obs_tp.csv", index=False)
    out = obs_points.run(pred, {}, {**cfg, "cache": str(c6), "tp_obs_var": "tp6"}, output_dir=tmp_path / "o6",
                         run_label="S")
    a = pd.read_csv(ref / "scores_per_date.csv")
    b = pd.read_csv(out / "scores_per_date.csv")
    num = [c for c in a.columns if a[c].dtype.kind == "f"]
    pd.testing.assert_frame_equal(a[num].reset_index(drop=True), b[num].reset_index(drop=True))
    assert json.loads((out / "run_meta.json").read_text())["tp_obs_var"] == "tp6"
