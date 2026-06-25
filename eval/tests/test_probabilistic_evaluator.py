from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import xarray as xr

from eval._backends.probabilistic.scoring import crps_ensemble_components
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
