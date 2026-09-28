"""Tests of the retired generate_enfo_o320_scoreboard.py, moved out of
eval/jobs/tests/test_scoreboard_metrics.py when that job was quarantined on
2026-09-28. Not collected (pytest.ini norecursedirs)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from eval.jobs import generate_enfo_o320_scoreboard as scoreboard
from eval.jobs import scoreboard_metrics as metrics


def test_build_scoreboard_rows_merges_prefix_related_run_ids():
    rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={},
        surface_loss_data={
            "manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {
                "weighted_nmse": 0.123,
                "variables": {
                    "10v": {"mean_nmse": 0.2},
                    "2t": {"mean_nmse": 0.01},
                    "msl": {"mean_nmse": 0.05},
                    "sp": {"mean_nmse": 0.02},
                },
            }
        },
        inference_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": "piecewise30"},
    )

    assert len(rows) == 1
    assert rows[0]["short_id"] == "39991df8"
    assert rows[0]["display_run_id"].startswith("manual_39991df_")
    assert rows[0]["inference"] == "piecewise30"
    assert rows[0]["sigma_1"] == pytest.approx(0.1)
    assert rows[0]["idalia_extreme"] == pytest.approx(0.9)
    assert rows[0]["surface_loss"] == pytest.approx(0.123)
    assert rows[0]["surface_10v"] == pytest.approx(0.2)
    assert rows[0]["surface_2t"] == pytest.approx(0.01)
    assert rows[0]["surface_msl"] == pytest.approx(0.05)
    assert rows[0]["surface_sp"] == pytest.approx(0.02)


def test_generate_scoreboard_markdown_includes_inference_column():
    rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={},
        surface_loss_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": 123.0},
        inference_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": "piecewise30"},
    )

    markdown = scoreboard.generate_scoreboard_markdown(rows)

    assert "| Ckpt | Inference | Run ID |" in markdown
    assert "| 39991df8 | piecewise30 | manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100 |" in markdown
    assert "- **Inference**: Schedule-plus-step label inferred from run metadata or prediction logs" in markdown


def test_generate_scoreboard_markdown_prefers_surface_nmse():
    rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": 0.25},
        surface_loss_data={
            "manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {
                "weighted_nmse": 0.17,
                "variables": {
                    "10v": {"mean_nmse": 0.2463},
                    "2t": {"mean_nmse": 0.0082},
                    "msl": {"mean_nmse": 0.0494},
                    "sp": {"mean_nmse": 0.0264},
                },
            }
        },
        inference_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": "piecewise30"},
    )

    markdown = scoreboard.generate_scoreboard_markdown(rows)

    assert "| Ckpt | Inference | Run ID | σ=1 loss | σ=5 loss | σ=10 loss | σ=100 loss | TC MSLP p0.01 | TC MSLP min | TC wind p99.99 | TC wind max | Spectra L2 | Sfc nMSE | 10v nMSE | 2t nMSE | MSLP nMSE | SP nMSE |" in markdown
    assert "| 39991df8 | piecewise30 | manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100 | 0.1000 | na | na | na | na | na | na | na | 0.2500 | 0.1700 | 0.2463 | 0.0082 | 0.0494 | 0.0264 |" in markdown
    assert "- **Sfc nMSE**: Area-weighted and variable-weighted surface MSE after per-variable truth-std normalization over the fixed Aug 26-30 evaluation contract" in markdown
    assert "- **10v / 2t / MSLP / SP nMSE**: Per-variable truth-std-normalized surface MSE for the named field, using the same fixed evaluation contract as the aggregate surface score" in markdown


def test_load_context_baseline_rows_reads_eefo_o96_input_baseline(tmp_path):
    source_csv = tmp_path / "scoreboard.csv"
    source_csv.write_text(
        "\n".join(
            [
                "tc_rank,label,checkpoint_short,eval_sampler_min,table_group,contract_status,idalia_tc_extreme_score,franklin_tc_extreme_score,spectra_10u_score_vs_reference,spectra_10v_score_vs_reference,spectra_2t_score_vs_reference,spectra_mean_score_vs_reference,spectra_10u_distance,spectra_10v_distance,spectra_2t_distance,spectra_mean_distance,surface_weighted_mse,validation_loss,role,spectra_coverage,spectra_n_curves,surface_loss_source,note,dossier",
                "-,enfo_o320,na,na,context,eligible,0.859539,na,1.000000,1.000000,1.000000,1.000000,0.000000,0.000000,0.000000,0.000000,na,na,truth-reference,reference by definition,na,na,Reference baseline.,/tmp/enfo.md",
                "-,eefo_o96,na,na,context,eligible,0.000000,na,na,na,na,na,na,na,na,na,na,na,input-baseline,missing,na,na,Context baseline (off-grid input).,/tmp/eefo.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = scoreboard.load_context_baseline_rows(source_csv)

    assert len(rows) == 1
    assert rows[0]["short_id"] == "x_interp"
    assert rows[0]["display_run_id"] == "eefo_o96"
    assert rows[0]["idalia_extreme"] == pytest.approx(0.0)
    assert math.isnan(rows[0]["sigma_1"])
    assert math.isnan(rows[0]["spectra_l2"])
    assert math.isnan(rows[0]["surface_loss"])


def test_load_context_baseline_rows_overrides_eefo_surface_with_real_xinterp_metrics(tmp_path):
    source_csv = tmp_path / "scoreboard.csv"
    source_csv.write_text(
        "\n".join(
            [
                "tc_rank,label,checkpoint_short,eval_sampler_min,table_group,contract_status,idalia_tc_extreme_score,franklin_tc_extreme_score,spectra_10u_score_vs_reference,spectra_10v_score_vs_reference,spectra_2t_score_vs_reference,spectra_mean_score_vs_reference,spectra_10u_distance,spectra_10v_distance,spectra_2t_distance,spectra_mean_distance,surface_weighted_mse,validation_loss,role,spectra_coverage,spectra_n_curves,surface_loss_source,note,dossier",
                "-,eefo_o96,na,na,context,eligible,0.000000,na,na,na,na,na,na,na,na,na,na,na,input-baseline,missing,na,na,Context baseline (off-grid input).,/tmp/eefo.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = scoreboard.load_context_baseline_rows(
        source_csv,
        context_surface_metrics={
            "eefo_o96": {
                "weighted_nmse": 0.1026,
                "variables": {
                    "10v": {"mean_nmse": 0.2},
                    "2t": {"mean_nmse": 0.01},
                    "msl": {"mean_nmse": 0.04},
                    "sp": {"mean_nmse": 0.03},
                },
            }
        },
    )

    assert rows[0]["surface_loss"] == pytest.approx(0.1026)
    assert rows[0]["surface_loss_text"] == "0.1026"
    assert rows[0]["surface_10v"] == pytest.approx(0.2)
    assert rows[0]["surface_2t"] == pytest.approx(0.01)
    assert rows[0]["surface_msl"] == pytest.approx(0.04)
    assert rows[0]["surface_sp"] == pytest.approx(0.03)


def test_generate_scoreboard_markdown_appends_eefo_o96_context_row():
    experiment_rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={},
        surface_loss_data={},
        inference_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": "piecewise30"},
    )
    experiment_rows.append(
        {
            "row_key": "eefo_o96",
            "display_run_id": "eefo_o96",
            "short_id": "x_interp",
            "inference": "na",
            "row_group": "context",
            "sigma_1": float("nan"),
            "sigma_5": float("nan"),
            "sigma_10": float("nan"),
            "sigma_100": float("nan"),
            "idalia_extreme": 0.0,
            "franklin_extreme": float("nan"),
            "enfo_deviation": float("nan"),
            "enfo_match": float("nan"),
            "mslp_reach": float("nan"),
            "wind_reach": float("nan"),
            "mslp_p001_ratio": float("nan"),
            "mslp_min_ratio": float("nan"),
            "wind_p9999_ratio": float("nan"),
            "wind_max_ratio": float("nan"),
            "spectra_l2": float("nan"),
            "surface_loss": float("nan"),
            "surface_10v": float("nan"),
            "surface_2t": float("nan"),
            "surface_msl": float("nan"),
            "surface_sp": float("nan"),
        }
    )

    markdown = scoreboard.generate_scoreboard_markdown(experiment_rows)

    assert "| x_interp | na | eefo_o96 | na | na | na | na | na | na | na | na | na | na | na | na | na | na |" in markdown
    assert markdown.index("| 39991df8 | piecewise30 | manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100 |") < markdown.index("| x_interp | na | eefo_o96 |")
    assert "- **Context baseline rows**: Curated comparison rows appended after experiment runs; `x_interp` is sourced from the docs-side `eefo_o96` input baseline" in markdown


def test_choose_xinterp_context_predictions_dir_prefers_matching_o96_o320_contract(tmp_path, monkeypatch):
    other_run = tmp_path / "manual_181be03e_new_o320_o1280_20260421_manual_eval"
    other_predictions = other_run / "predictions"
    other_predictions.mkdir(parents=True)
    (other_predictions / "predictions_20230827_step024.nc").write_text("", encoding="utf-8")
    (other_run / "EXPERIMENT_CONFIG.yaml").write_text(
        json.dumps(
            {
                "lane": "o320_o1280",
                "source": {
                    "bundle_scope": {
                        "dates": [20230827, 20230828, 20230829, 20230830],
                        "steps_hours": [24, 48, 72],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    matching_run = tmp_path / "manual_59e4_300k"
    matching_predictions = matching_run / "predictions"
    matching_predictions.mkdir(parents=True)
    (matching_predictions / "predictions_20230826_step024.nc").write_text("", encoding="utf-8")
    (matching_run / "EXPERIMENT_CONFIG.yaml").write_text(
        json.dumps(
            {
                "lane": "o96_o320",
                "source": {
                    "bundle_scope": {
                        "dates": [20230826, 20230827, 20230828, 20230829, 20230830],
                        "steps_hours": [24, 48, 72, 96, 120],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        scoreboard,
        "_predictions_support_xinterp",
        lambda path: path == matching_predictions,
    )

    chosen = scoreboard.choose_xinterp_context_predictions_dir(tmp_path)

    assert chosen == matching_predictions
