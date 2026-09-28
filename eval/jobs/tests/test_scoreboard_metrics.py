from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from eval.jobs import scoreboard_metrics as metrics


def test_load_sigma_losses_from_csv_normalizes_float_sigma_labels(tmp_path):
    csv_path = tmp_path / "sample_sigma_eval.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sigma", "loss"])
        writer.writeheader()
        writer.writerow({"sigma": "1.0", "loss": "0.1"})
        writer.writerow({"sigma": "5.0", "loss": "0.2"})
        writer.writerow({"sigma": "10", "loss": "0.3"})

    result = metrics.load_sigma_losses_from_csv(csv_path)

    assert result == {
        "sigma_1": pytest.approx(0.1),
        "sigma_5": pytest.approx(0.2),
        "sigma_10": pytest.approx(0.3),
    }


def test_load_spectra_metrics_falls_back_to_raw_surface_fields(tmp_path):
    spectra_dir = tmp_path / "spectra_step120_5dates_m10_ecmwf"
    ref_root = tmp_path / "reference"
    spectra_dir.mkdir()
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(
        json.dumps(
            {
                "dates": [20230826],
                "steps_hours": [120],
                "ensemble_members": [1],
                "template_root": str(ref_root),
            }
        )
    )

    for field_dir in metrics.RAW_FIELD_DIRS.values():
        run_field_dir = spectra_dir / field_dir
        ref_field_dir = ref_root / field_dir
        run_field_dir.mkdir()
        ref_field_dir.mkdir()
        run_curve = np.ones(200, dtype=np.float64)
        ref_curve = np.ones(200, dtype=np.float64)
        run_curve[149] = 2.0
        np.save(run_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", run_curve)
        np.save(ref_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", ref_curve)

    result = metrics.load_spectra_metrics(spectra_dir)

    assert result["10u"] is not None
    assert result["10v"] is not None
    assert result["2t"] is not None
    # Mean requires at least 3 fields with data
    assert result["mean"] is not None
    assert result["n_curves"] == 1


def test_load_spectra_metrics_ignores_differences_below_high_wavenumber_threshold(tmp_path):
    spectra_dir = tmp_path / "spectra_step120_5dates_m10_ecmwf"
    ref_root = tmp_path / "reference"
    spectra_dir.mkdir()
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(
        json.dumps(
            {
                "dates": [20230826],
                "steps_hours": [120],
                "ensemble_members": [1],
                "template_root": str(ref_root),
            }
        )
    )

    wvn = np.arange(1.0, 201.0, dtype=np.float64)
    ref_curve = np.ones(200, dtype=np.float64)
    run_curve = ref_curve.copy()
    run_curve[49] = 5.0  # wavenumber 50 should be ignored by the scoreboard metric

    for field_dir in metrics.RAW_FIELD_DIRS.values():
        run_field_dir = spectra_dir / field_dir
        ref_field_dir = ref_root / field_dir
        run_field_dir.mkdir()
        ref_field_dir.mkdir()
        np.save(run_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", run_curve)
        np.save(ref_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", ref_curve)
        np.save(run_field_dir / f"wvn_20230826_120_{field_dir}_1_n1.npy", wvn)
        np.save(ref_field_dir / f"wvn_20230826_120_{field_dir}_1_n1.npy", wvn)

    result = metrics.load_spectra_metrics(spectra_dir)

    assert result["10u"] == pytest.approx(0.0)
    assert result["10v"] == pytest.approx(0.0)
    assert result["2t"] == pytest.approx(0.0)
    assert result["mean"] == pytest.approx(0.0)
    assert result["score_wavenumber_min_exclusive"] == pytest.approx(100.0)


def test_load_spectra_metrics_prefers_raw_arrays_over_stale_summary(tmp_path):
    spectra_dir = tmp_path / "spectra_step120_5dates_m10_ecmwf"
    ref_root = tmp_path / "reference"
    spectra_dir.mkdir()
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(
        json.dumps(
            {
                "dates": [20230826],
                "steps_hours": [120],
                "ensemble_members": [1],
                "template_root": str(ref_root),
            }
        )
    )
    (spectra_dir / "spectra_summary.json").write_text(
        json.dumps(
            {
                "method": "ecmwf_mean_curve_reference_l2",
                "10u": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "10v": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "2t": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "msl": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "t_850": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "z_500": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
            }
        )
    )

    wvn = np.arange(1.0, 201.0, dtype=np.float64)
    curve = np.ones(200, dtype=np.float64)
    for field_dir in metrics.RAW_FIELD_DIRS.values():
        run_field_dir = spectra_dir / field_dir
        ref_field_dir = ref_root / field_dir
        run_field_dir.mkdir()
        ref_field_dir.mkdir()
        np.save(run_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", curve)
        np.save(ref_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", curve)
        np.save(run_field_dir / f"wvn_20230826_120_{field_dir}_1_n1.npy", wvn)
        np.save(ref_field_dir / f"wvn_20230826_120_{field_dir}_1_n1.npy", wvn)

    result = metrics.load_spectra_metrics(spectra_dir)

    assert result["mean"] == pytest.approx(0.0)


def test_load_spectra_metrics_falls_back_to_comparison_step_means(tmp_path):
    run_root = tmp_path / "manual_step_mean"
    spectra_dir = run_root / "spectra_proxy10_subset_ecmwf"
    ref_root = run_root / "enfo_o320"
    spectra_dir.mkdir(parents=True)
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(
        json.dumps(
            {
                "dates": [20230827],
                "steps_hours": [24],
                "ensemble_members": [1],
                "template_root": str(ref_root),
            }
        )
    )
    (spectra_dir / "spectra_summary.json").write_text(
        json.dumps(
            {
                "weather_states": {
                    "10u": {"status": "missing"},
                    "10v": {"status": "missing"},
                    "2t": {"status": "missing"},
                    "sp": {"status": "missing"},
                    "t_850": {"status": "missing"},
                    "z_500": {"status": "missing"},
                }
            }
        )
    )

    comparison_summary = {
        "base_dir": str(run_root),
        "output_dir": str(spectra_dir),
        "prefer_step": 24,
        "models": [
            {"name": spectra_dir.name, "path": str(spectra_dir), "exists": True, "steps_with_wvn": [24], "chosen_step": 24},
            {"name": ref_root.name, "path": str(ref_root), "exists": True, "steps_with_wvn": [24], "chosen_step": 24},
        ],
        "per_param": {},
    }

    wvn = np.arange(1.0, 201.0, dtype=np.float64)
    ref_curve = np.ones(200, dtype=np.float64)
    run_curve = ref_curve.copy()
    run_curve[149] = 2.0
    expected = metrics.relative_l2_weighted(run_curve, ref_curve, wavenumbers=wvn)

    field_dirs = {
        "10u_sfc": "10u_sfc",
        "10v_sfc": "10v_sfc",
        "2t_sfc": "2t_sfc",
        "sp_sfc": "sp_sfc",
        "t_850": "t_850",
        "z_500": "z_500",
    }
    for field_dir in field_dirs.values():
        (spectra_dir / field_dir).mkdir()
        (ref_root / field_dir).mkdir()
        np.save(spectra_dir / field_dir / f"wvn_20230827_24_{field_dir}_1_n1.npy", wvn)
        np.save(spectra_dir / field_dir / f"ampl_20230827_24_{field_dir}_1_n1.npy", run_curve)
        np.save(ref_root / field_dir / f"wvn_20240201_24_{field_dir}_1_n1.npy", wvn)
        np.save(ref_root / field_dir / f"ampl_20240201_24_{field_dir}_1_n1.npy", ref_curve)
        comparison_summary["per_param"][field_dir] = {
            spectra_dir.name: {"status": "plotted", "step": 24, "files": 1},
            ref_root.name: {"status": "plotted", "step": 24, "files": 1},
        }

    (spectra_dir / "comparison_summary.json").write_text(json.dumps(comparison_summary))

    result = metrics.load_spectra_metrics(spectra_dir)

    assert result["10u"] == pytest.approx(expected)
    assert result["10v"] == pytest.approx(expected)
    assert result["2t"] == pytest.approx(expected)
    assert result["msl"] == pytest.approx(expected)
    assert result["t_850"] == pytest.approx(expected)
    assert result["z_500"] == pytest.approx(expected)
    assert result["mean"] == pytest.approx(expected)
    assert result["count_label"] == "curves"
    assert result["source_path"] == str(spectra_dir / "comparison_summary.json")


def test_load_surface_loss_metrics_uses_truth_std_normalization_fallback(tmp_path):
    summary_path = tmp_path / "surface_loss_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "weighted_surface_mse": 10.0,
                "variables": {
                    "msl": {"mean_mse": 10.0, "normalized_weight": 0.6},
                    "sp": {"mean_mse": 5.0, "normalized_weight": 0.3},
                    "2t": {"mean_mse": 2.0, "normalized_weight": 0.1},
                },
            }
        )
    )

    result = metrics.load_surface_loss_metrics(
        summary_path,
        truth_std_by_variable={"msl": 10.0, "sp": 5.0, "2t": 2.0},
    )

    assert result["weighted_mse"] == pytest.approx(10.0)
    assert result["weighted_nmse"] == pytest.approx(0.6 * 0.1 + 0.3 * 0.2 + 0.1 * 0.5)
    assert [entry["variable"] for entry in result["top_contributors"]] == ["msl", "sp"]
    assert metrics.format_surface_loss_for_scoreboard(result) == "0.1700"


def test_infer_eval_sampler_min_from_run_root_prefers_experiment_config(tmp_path):
    run_root = tmp_path / "manual_piecewise"
    run_root.mkdir()
    (run_root / "EXPERIMENT_CONFIG.yaml").write_text(
        json.dumps(
            {
                "sampling_config_json": json.dumps(
                    {
                        "schedule_type": "experimental_piecewise",
                        "num_steps": 30,
                        "sigma_max": 100000.0,
                        "sigma_min": 0.03,
                    }
                )
            }
        ),
        encoding="utf-8",
    )

    assert metrics.infer_eval_sampler_min_from_run_root(run_root) == "piecewise30"


def test_infer_eval_sampler_min_from_run_root_falls_back_to_logs(tmp_path):
    run_root = tmp_path / "manual_karras"
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "predict25_manual_karras_123.out").write_text(
        "noise_scheduler_params: {'schedule_type': 'karras', 'num_steps': 40, 'sigma_max': 1000.0, 'sigma_min': 0.03, 'rho': 7.0}\n",
        encoding="utf-8",
    )

    assert metrics.infer_eval_sampler_min_from_run_root(run_root) == "karras40"


def test_infer_eval_sampler_min_from_run_root_falls_back_to_run_id(tmp_path):
    run_root = tmp_path / "manual_83edcda0_multi_o96_o320_20260331_heun40_sigmax1000"
    run_root.mkdir()
    # No EXPERIMENT_CONFIG.yaml and no logs with schedule info

    assert metrics.infer_eval_sampler_min_from_run_root(run_root) == "heun40"


def test_finite_positive_mask_adaptive_threshold_low_lmax():
    """When max wavenumber <= 100, threshold drops to max_wvn/3."""
    wvn = np.arange(0, 96, dtype=np.float64)  # lmax=95
    arr = np.ones_like(wvn) * 10.0
    mask = metrics.finite_positive_mask(arr, wavenumbers=wvn)
    # Effective threshold = 95/3 ≈ 31.67, so wavenumbers > 31.67 should pass
    assert mask.sum() > 0, "Should score wavenumbers when lmax < 100"
    assert mask[0] is np.False_, "Wavenumber 0 should be excluded"
    assert mask[32] is np.True_, "Wavenumber 32 should be included"


def test_finite_positive_mask_standard_threshold_high_lmax():
    """When max wavenumber > 100, threshold stays at 100."""
    wvn = np.arange(0, 321, dtype=np.float64)  # lmax=320
    arr = np.ones_like(wvn) * 10.0
    mask = metrics.finite_positive_mask(arr, wavenumbers=wvn)
    # Threshold stays at 100
    assert mask[100] is np.False_, "Wavenumber 100 should be excluded (> not >=)"
    assert mask[101] is np.True_, "Wavenumber 101 should be included"
    assert mask[50] is np.False_, "Wavenumber 50 should be excluded"


def test_build_run_scoreboard_metrics_sigma_fallback_to_run_root(tmp_path):
    """When SIGMA_RUN_ID is blank, sigma loads from <run_root>/sigma_eval_table.csv."""
    run_id = "manual_abc123_new_o48_o96_20260422_full_eval"
    run_root = tmp_path / run_id
    run_root.mkdir()

    # Write sigma CSV to the run root (not scoreboards/sigma/)
    sigma_csv = run_root / "sigma_eval_table.csv"
    with sigma_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma", "loss"])
        writer.writeheader()
        writer.writerow({"sigma": "1.0", "loss": "0.098"})
        writer.writerow({"sigma": "5.0", "loss": "0.139"})
        writer.writerow({"sigma": "10.0", "loss": "0.162"})
        writer.writerow({"sigma": "100.0", "loss": "0.211"})

    # Empty TC/spectra/surface stubs
    tc_path = run_root / "tc_stats.json"
    tc_path.write_text("{}", encoding="utf-8")
    spectra_dir = run_root / "spectra"
    spectra_dir.mkdir()
    surface_path = run_root / "surface_loss.json"
    surface_path.write_text("{}", encoding="utf-8")

    result = metrics.build_run_scoreboard_metrics(
        run_id=run_id,
        output_root=tmp_path,
        sigma_run_id="",
        tc_stats_path=tc_path,
        spectra_dir=spectra_dir,
        surface_json_path=surface_path,
    )

    assert result["sigma_losses"]["sigma_1"] == pytest.approx(0.098)
    assert result["sigma_losses"]["sigma_5"] == pytest.approx(0.139)
    assert result["sigma_losses"]["sigma_10"] == pytest.approx(0.162)
    assert result["sigma_losses"]["sigma_100"] == pytest.approx(0.211)


def test_rescore_from_curve_summary_low_lmax(tmp_path):
    """_rescore_from_curve_summary rescores when lmax < 100."""
    spectra_dir = tmp_path / "spectra_o48_o96"
    spectra_dir.mkdir()

    # Build a curve summary with lmax=95 and identical pred/truth means
    wvn = list(range(96))  # 0..95
    pred_mean = [float(i + 1) for i in range(96)]
    truth_mean = [float(i + 1) for i in range(96)]

    weather_states = {}
    for field in ["10u", "10v", "2t", "msl", "t_850", "z_500"]:
        weather_states[field] = {
            "status": "ok",
            "scopes": {
                "residual": {
                    "status": "ok",
                    "n_curves": 25,
                    "wavenumbers": wvn,
                    "prediction_mean": pred_mean,
                    "truth_mean": truth_mean,
                }
            },
        }

    (spectra_dir / "spectra_curve_summary.json").write_text(
        json.dumps({
            "score_wavenumber_min_exclusive": 100.0,
            "weather_states": weather_states,
        }),
        encoding="utf-8",
    )

    result = metrics._rescore_from_curve_summary(spectra_dir)

    assert result["mean"] is not None
    assert result["mean"] == pytest.approx(0.0)
    assert result["coverage"] == "25 curves"
    for field in ["10u", "10v", "2t", "msl", "t_850", "z_500"]:
        assert result[field] == pytest.approx(0.0)


def _raw_extremes_stats() -> dict:
    """A minimal TC stats payload: model, OPER analysis, ENFO and EEFO rows for one event."""
    def row(exp, mslp_min, mslp_p001, wind_max, wind_p9999):
        return {"exp": exp, "mslp_min": mslp_min, "mslp_p001": mslp_p001,
                "wind_max": wind_max, "wind_p9999": wind_p9999}

    return {"events": {"idalia": {"extreme_tail": {"rows": [
        row("OPER_O320_0001", 960.0, 965.0, 40.0, 35.0),
        row("ENFO_O320_0001", 970.0, 975.0, 32.0, 28.0),
        row("EEFO_O96", 985.0, 990.0, 24.0, 20.0),
        row("manual_0c446b41_new_o96_o320", 965.0, 970.0, 36.0, 32.0),
    ]}}}}


def test_load_tc_extreme_scores_emits_raw_extremes_per_source(tmp_path):
    """The raw-extremes contract: the four extremes for the model and for each reference row, nothing else."""
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_raw_extremes_stats()))

    result = metrics.load_tc_extreme_scores_from_json(stats_path, run_id="manual_0c446b41_new_o96_o320")

    assert result["idalia_mslp_min"] == 965.0 and result["idalia_wind_max"] == 36.0
    assert result["idalia_mslp_p001"] == 970.0 and result["idalia_wind_p9999"] == 32.0
    assert result["idalia_oper_mslp_min"] == 960.0
    assert result["idalia_enfo_wind_max"] == 32.0
    assert result["idalia_eefo_mslp_p001"] == 990.0
    assert len(result) == 16  # 4 extremes x (model + OPER + ENFO + EEFO); no score, ratio or anchor keys


def test_load_tc_extreme_scores_ignores_the_retired_anchor_arguments(tmp_path):
    """The anchor arguments are still accepted, for old callers, and change nothing."""
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(_raw_extremes_stats()))
    plain = metrics.load_tc_extreme_scores_from_json(stats_path, run_id="manual_0c446b41_new_o96_o320")
    with_anchors = metrics.load_tc_extreme_scores_from_json(
        stats_path, run_id="manual_0c446b41_new_o96_o320",
        canonical_analysis_by_event={"idalia": {}}, canonical_eefo_by_event={"idalia": {}},
        extreme_reference_expid="ENFO_O320_0001",
    )
    assert plain == with_anchors


def test_load_tc_extreme_scores_supports_custom_event_names(tmp_path):
    payload = _raw_extremes_stats()
    payload["events"]["humberto"] = payload["events"].pop("idalia")
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(json.dumps(payload))

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path, run_id="manual_0c446b41_new_o96_o320", event_names=("humberto",),
    )

    assert result["humberto_mslp_min"] == 965.0
    assert not any(key.startswith("idalia") for key in result)
