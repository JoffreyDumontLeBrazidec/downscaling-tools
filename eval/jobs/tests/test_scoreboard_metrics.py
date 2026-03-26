from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from eval.jobs import generate_enfo_o320_scoreboard as scoreboard
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


def test_load_tc_extreme_scores_falls_back_to_normalized_tail_fractions(tmp_path):
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "events": {
                    "idalia": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.1, "wind_gt_25_fraction": 0.1},
                                {"exp": "manual_0c446b41_new_o96_o320", "mslp_980_990_fraction": 0.8, "wind_gt_25_fraction": 0.6},
                            ]
                        }
                    },
                    "franklin": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.2, "wind_gt_25_fraction": 0.2},
                                {"exp": "manual_0c446b41_new_o96_o320", "mslp_980_990_fraction": 0.5, "wind_gt_25_fraction": 0.8},
                            ]
                        }
                    },
                }
            }
        )
    )

    result = metrics.load_tc_extreme_scores_from_json(stats_path, run_id="manual_0c446b41_new_o96_o320")

    assert result["idalia"] == pytest.approx(1.0)
    assert result["franklin"] == pytest.approx(1.0)


def test_load_tc_extreme_scores_supports_custom_event_names(tmp_path):
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "events": {
                    "humberto": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.1, "wind_gt_25_fraction": 0.1},
                                {"exp": "manual_95a07500_new_o48_o96", "mslp_980_990_fraction": 0.9, "wind_gt_25_fraction": 0.8},
                            ]
                        }
                    }
                }
            }
        )
    )

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_95a07500_new_o48_o96",
        event_names=("humberto",),
    )

    assert result == {"humberto": pytest.approx(1.0)}


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


def test_build_scoreboard_rows_merges_prefix_related_run_ids():
    rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={},
        surface_loss_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": 123.0},
    )

    assert len(rows) == 1
    assert rows[0]["short_id"] == "39991df8"
    assert rows[0]["display_run_id"].startswith("manual_39991df_")
    assert rows[0]["sigma_1"] == pytest.approx(0.1)
    assert rows[0]["idalia_extreme"] == pytest.approx(0.9)
    assert rows[0]["surface_loss"] == pytest.approx(123.0)
