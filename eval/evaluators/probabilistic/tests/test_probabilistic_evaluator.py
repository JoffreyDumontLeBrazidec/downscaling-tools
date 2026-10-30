from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from eval.evaluators.probabilistic.core.scoring import crps_ensemble_components
from eval.evaluators.probabilistic.runner import run
from eval.evaluators.probabilistic.scorer import score
from eval.evaluators.probabilistic.plotter import plot
from eval.jobs.compare_probabilistic_reference import compare


def _brute_crps(forecasts: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = forecasts.shape[0]
    term1 = np.mean(np.abs(forecasts - truth[None, :]), axis=0)
    pair_sum = np.abs(forecasts[:, None, :] - forecasts[None, :, :]).sum(axis=(0, 1))
    crps = term1 - pair_sum / (2 * m * m)
    fcrps = term1 - pair_sum / (2 * m * (m - 1))
    return crps, fcrps


def test_crps_components_match_bruteforce() -> None:
    forecasts = np.array([[0.0, 2.0, 3.0], [1.0, 4.0, 5.0], [2.0, 6.0, 7.0]])
    truth = np.array([1.0, 3.0, 10.0])
    got = crps_ensemble_components(forecasts, truth)
    crps, fcrps = _brute_crps(forecasts, truth)
    np.testing.assert_allclose(got["crps"], crps)
    np.testing.assert_allclose(got["fcrps"], fcrps)
    np.testing.assert_allclose(got["spread"], np.std(forecasts, axis=0, ddof=1))


def _write_prediction(path: Path, *, date: str, step: int) -> None:
    members = np.array([1, 2, 3])
    weather = np.array(["2t", "10u", "10v"], dtype=object)
    lat = np.array([30.0, 40.0, -10.0, -35.0], dtype=np.float32)
    lon = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    base_truth = np.array(
        [
            [280.0, 3.0, 4.0],
            [281.0, 4.0, 0.0],
            [282.0, 0.0, 5.0],
            [283.0, 8.0, 6.0],
        ],
        dtype=np.float32,
    )
    y = np.broadcast_to(base_truth, (1, members.size, 4, weather.size)).copy()
    offsets = np.array([-1.0, 0.0, 1.0], dtype=np.float32)[None, :, None, None]
    y_pred = y + offsets
    x = np.zeros((1, members.size, 2, weather.size), dtype=np.float32)
    ds = xr.Dataset(
        {
            "x": (("sample", "ensemble_member", "grid_point_lres", "weather_state"), x),
            "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y_pred),
            "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
            "lat_hres": (("grid_point_hres",), lat),
            "lon_hres": (("grid_point_hres",), lon),
            "lat_lres": (("grid_point_lres",), np.array([0.0, 1.0], dtype=np.float32)),
            "lon_lres": (("grid_point_lres",), np.array([0.0, 1.0], dtype=np.float32)),
        },
        coords={
            "sample": [0],
            "ensemble_member": members,
            "grid_point_lres": [0, 1],
            "grid_point_hres": np.arange(4),
            "weather_state": weather,
        },
        attrs={"init_date": date, "lead_step_hours": step},
    )
    ds.to_netcdf(path)


def test_probabilistic_evaluator_writes_outputs(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_prediction(pred_dir / "predictions_20230816_step024.nc", date="20230816", step=24)
    _write_prediction(pred_dir / "predictions_20230817_step024.nc", date="20230817", step=24)

    out_dir = tmp_path / "evaluators" / "probabilistic"
    eval_config = {"weather_states": ["2t", "10ff"], "domains": ["global", "n.hem"], "steps": [24]}
    run(pred_dir, {}, eval_config, output_dir=out_dir, overwrite=True)
    records = score(out_dir, {}, eval_config)
    plot(out_dir, {}, eval_config)

    assert (out_dir / "scores_by_lead.csv").exists()
    assert (out_dir / "summary_by_lead.csv").exists()
    assert (out_dir / "probabilistic_summary.json").exists()
    assert (out_dir / "plots" / "probabilistic_scores.pdf").exists()
    assert any(record["metric"].startswith("probabilistic_2t_global_crps") for record in records)
    with (out_dir / "summary_by_lead.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["weather_state"] for row in rows} == {"2t", "10ff"}
    # the historical schema is untouched
    assert list(rows[0]) == ["step", "weather_state", "domain", "metric", "mean", "std", "stderr", "n_dates", "n_points_total"]
    assert {row["metric"] for row in rows} == {"crps", "fcrps", "spread", "rmse_ens_mean"}
    # the calibration read and the rank histograms are there (full band only: no scale split)
    with (out_dir / "calibration_by_lead.csv").open(newline="") as f:
        cal = list(csv.DictReader(f))
    assert {r["band"] for r in cal} == {"full"}
    r = next(r for r in cal if r["weather_state"] == "2t" and r["domain"] == "global")
    assert r["n_members"] == "3" and r["n_dates"] == "2" and r["n_points_total"] == "8"
    # members at -1, 0, +1 around the truth: variance 1, error 0 (ratio undefined), fair CRPS exactly 0
    assert r["spread_rms"] == "1.0" and float(r["crps_mean"]) > 0 and float(r["fcrps_mean"]) == 0.0
    with (out_dir / "rank_histograms.csv").open(newline="") as f:
        hist = list(csv.DictReader(f))
    h = next(r for r in hist if r["weather_state"] == "2t" and r["domain"] == "global")
    # truth equals member 2 at every point: one member below, one tied, so the rank is 1 or 2 at random
    assert h["n_bins"] == "4" and int(h["bin_1"]) + int(h["bin_2"]) == 8 and h["bin_0"] == "0" and h["bin_3"] == "0"
    assert (out_dir / "plots" / "rank_histograms.pdf").exists()
    assert not (out_dir / "plots" / "band_scores.pdf").exists()
    payload = json.loads((out_dir / "probabilistic_summary.json").read_text())
    assert payload["schema_version"] == "1.1" and payload["scale_split"]["enabled"] is False
    assert payload["input_members"]["n_members"] == 0           # x on the coarse grid only, no x_interp
    assert payload["rank_histogram"]["n_bins"] == 4


def _write_box_prediction(path: Path, *, date: str, step: int, n_members: int = 10, with_input: bool = True,
                          seed: int = 0) -> dict:
    """A 60 km box (20-50 N, -40..10 E) with 2t, 10u, 10v and tp; smooth large scales plus fine noise."""
    rng = np.random.default_rng(seed)
    lat1 = np.arange(20.0, 50.0, 0.54)
    lon1 = np.arange(-40.0, 10.0, 0.7)
    lon_g, lat_g = np.meshgrid(lon1, lat1)
    lat, lon = lat_g.ravel().astype(np.float32), lon_g.ravel().astype(np.float32)
    n = lat.size
    x_km = (lon - lon.mean()) * 111.2 * np.cos(np.deg2rad(lat))
    large = np.sin(2 * np.pi * x_km / 1500.0)
    weather = np.array(["2t", "10u", "10v", "tp"], dtype=object)
    # calibrated: truth and members are the same smooth base plus independent unit noise
    base = np.stack([280.0 + 5 * large, 5 * large, np.zeros(n), 2 * large], axis=-1)
    truth = base + rng.standard_normal(base.shape)
    truth[:, 3] = np.maximum(truth[:, 3], 0.0)
    members = np.stack([base + rng.standard_normal(base.shape) for _ in range(n_members)])
    members[..., 3] -= 0.3          # negative rain in the members, to be clipped (and dry ties after)
    y_pred = members[None].astype(np.float32)
    y = np.broadcast_to(truth, (1, n_members, n, 4)).astype(np.float32)
    data = {
        "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y_pred),
        "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
        "lat_hres": (("grid_point_hres",), lat),
        "lon_hres": (("grid_point_hres",), lon),
    }
    if with_input:
        inputs = np.stack([truth + np.stack([3 * large * 0.1 * rng.standard_normal(), np.zeros(n), np.zeros(n), np.zeros(n)],
                                            axis=-1) for _ in range(n_members)])
        data["x_interp"] = (("sample", "ensemble_member", "grid_point_hres", "weather_state"), inputs[None].astype(np.float32))
    ds = xr.Dataset(data, coords={"sample": [0], "ensemble_member": np.arange(1, n_members + 1),
                                  "grid_point_hres": np.arange(n), "weather_state": weather},
                    attrs={"init_date": date, "lead_step_hours": step})
    ds.to_netcdf(path)
    return {"n_points": n, "truth_tp": truth[:, 3], "members_tp": members[..., 3]}


def test_scale_split_tp_clipping_and_input_spread(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    facts = _write_box_prediction(pred_dir / "predictions_20230816_step024.nc", date="20230816", step=24)
    _write_box_prediction(pred_dir / "predictions_20230817_step024.nc", date="20230817", step=24, seed=1)
    out_dir = tmp_path / "evaluators" / "probabilistic"
    eval_config = {"weather_states": ["2t", "10ff", "tp"], "domains": ["global"], "steps": [24],
                   "scale_split": True, "scale_cut_km": 400, "scale_edge_margin_km": 300}
    run(pred_dir, {}, eval_config, output_dir=out_dir, overwrite=True)
    plot(out_dir, {}, eval_config)

    with (out_dir / "calibration_by_lead.csv").open(newline="") as f:
        cal = {(r["weather_state"], r["band"]): r for r in csv.DictReader(f)}
    assert {b for _, b in cal} == {"full", "large", "fine"}
    full, large, fine = cal[("2t", "full")], cal[("2t", "large")], cal[("2t", "fine")]
    # truth and members = base + independent N(0,1): calibrated, so the corrected ratio reads about 1 in
    # every band (the large band has few independent 400 km blobs in the box, hence the wider tolerance)
    for r, tol in ((full, 0.1), (large, 0.3), (fine, 0.1)):
        assert abs(float(r["spread_skill_ratio_corrected"]) - 1.0) < tol, r
        assert abs(float(r["spread_skill_ratio"]) * 1.0488 - float(r["spread_skill_ratio_corrected"])) < 1e-3
        assert r["n_members"] == "10" and r["n_dates"] == "2"
    # the band split of white noise on a 60 km lattice: the 53 km Gaussian keeps about d^2/(4 pi sigma^2) = 10 %
    # of the noise variance (rms 0.32), the rest is fine
    assert 0.25 < float(large["spread_rms"]) < 0.40 and float(fine["spread_rms"]) > 2 * float(large["spread_rms"])
    assert int(large["n_points_total"]) < int(full["n_points_total"])     # the edge margin is dropped
    # the input members carry spread in 2t only in the large band (a scaled smooth pattern)
    assert float(large["spread_input_rms"]) > 0 and large["input_n_members"] == "10"
    assert float(large["spread_input_share"]) > 0
    assert float(cal[("10ff", "large")]["spread_input_rms"]) < 1e-10      # identical input winds: no spread
    # tp is clipped at zero before scoring: the pooled error must be the error of the clipped members
    tp_full = cal[("tp", "full")]
    clipped = np.maximum(facts["members_tp"], 0.0)
    assert float(tp_full["rmse_ens_mean"]) < np.sqrt(np.mean((facts["members_tp"].mean(0) - facts["truth_tp"]) ** 2))
    assert float(tp_full["spread_rms"]) < np.sqrt(np.mean(np.var(facts["members_tp"], axis=0, ddof=1)))
    assert np.isfinite(float(cal[("tp", "fine")]["fcrps_mean"]))
    with (out_dir / "band_scores_by_lead.csv").open(newline="") as f:
        band_rows = list(csv.DictReader(f))
    assert {r["metric"] for r in band_rows} >= {"fcrps", "crps", "spread_rms", "rmse_ens_mean", "ms_spread", "ms_err"}
    assert {r["date"] for r in band_rows} == {"20230816", "20230817"}
    # rank histogram of tp: dry truth with several dry members is a tie, spread over the bins
    with (out_dir / "rank_histograms.csv").open(newline="") as f:
        hist = {r["weather_state"]: r for r in csv.DictReader(f)}
    counts = np.array([int(hist["tp"][f"bin_{i}"]) for i in range(11)])
    assert counts.sum() == 2 * facts["n_points"] and counts[0] / counts.sum() < 0.3
    payload = json.loads((out_dir / "probabilistic_summary.json").read_text())
    assert payload["scale_split"]["enabled"] and payload["scale_split"]["cut_km"] == 400.0
    assert payload["scale_split"]["sigma_km"] == round(53.0, 0) or abs(payload["scale_split"]["sigma_km"] - 53.0) < 0.1
    assert payload["precip_clipped_at_zero"] == ["tp"]
    assert payload["input_members"]["n_members"] == 10
    assert (out_dir / "plots" / "band_scores.pdf").exists()
    assert (out_dir / "plots" / "rank_histograms.pdf").exists()


def test_single_input_member_is_reported_not_scored(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_box_prediction(pred_dir / "predictions_20230816_step024.nc", date="20230816", step=24, with_input=False)
    out_dir = tmp_path / "evaluators" / "probabilistic"
    run(pred_dir, {}, {"weather_states": ["2t"], "domains": ["global"], "scale_split": True}, output_dir=out_dir)
    payload = json.loads((out_dir / "probabilistic_summary.json").read_text())
    assert payload["input_members"]["n_members"] == 0 and "no x_interp" in payload["input_members"]["note"]
    with (out_dir / "calibration_by_lead.csv").open(newline="") as f:
        cal = list(csv.DictReader(f))
    assert all(r["spread_input_rms"] in ("", "nan") for r in cal)


def test_probabilistic_reference_compare_writes_outputs(tmp_path: Path) -> None:
    local = tmp_path / "local.csv"
    reference = tmp_path / "reference.csv"
    local.write_text(
        "step,weather_state,domain,metric,mean\n"
        "24,2t,n.hem,fcrps,1.0\n"
        "48,2t,n.hem,fcrps,2.0\n"
    )
    reference.write_text(
        "step,weather_state,domain,metric,value\n"
        "24,2t,n.hem,fcrps,1.5\n"
        "48,2t,n.hem,fcrps,1.0\n"
    )
    summary = compare(local, reference, tmp_path / "cmp")
    assert summary["matched_rows"] == 2
    assert summary["mean_abs_diff_by_metric"]["fcrps"] == 0.75
    assert (tmp_path / "cmp" / "probabilistic_reference_comparison.csv").exists()
    assert (tmp_path / "cmp" / "probabilistic_reference_comparison.json").exists()
    assert (tmp_path / "cmp" / "probabilistic_reference_overlay.pdf").exists()
