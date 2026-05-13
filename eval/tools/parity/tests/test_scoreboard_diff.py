"""Tests for the scoreboard-level parity diff."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def _write_csv(path: Path, rows: list[dict]) -> Path:
    """Write a minimal scoreboard CSV."""
    import csv

    fieldnames = ["evaluator", "metric", "value", "unit"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def test_diff_scoreboards_identical(tmp_path):
    from eval.tools.parity import diff_scoreboards

    rows = [
        {"evaluator": "surface", "metric": "surface_weighted_nmse", "value": 0.1, "unit": "nmse"},
        {"evaluator": "spectra", "metric": "spectra_mean_score", "value": 0.98, "unit": "score_0_1"},
    ]
    left = _write_csv(tmp_path / "left.csv", rows)
    right = _write_csv(tmp_path / "right.csv", rows)

    report = diff_scoreboards(left, right)
    assert report.shared_count == 2
    assert all(d.abs_diff == 0.0 for d in report.shared)
    assert all(d.rel_diff == 0.0 for d in report.shared)
    assert report.only_left == []
    assert report.only_right == []


def test_diff_scoreboards_computes_abs_and_rel(tmp_path):
    from eval.tools.parity import diff_scoreboards

    left = _write_csv(tmp_path / "left.csv", [
        {"evaluator": "surface", "metric": "surface_10u_nmse", "value": 0.1, "unit": "nmse"},
    ])
    right = _write_csv(tmp_path / "right.csv", [
        {"evaluator": "surface", "metric": "surface_10u_nmse", "value": 0.15, "unit": "nmse"},
    ])

    report = diff_scoreboards(left, right)
    assert report.shared_count == 1
    d = report.shared[0]
    assert d.abs_diff == pytest.approx(0.05)
    # rel = |0.05| / ((0.1 + 0.15) / 2) = 0.05 / 0.125 = 0.4
    assert d.rel_diff == pytest.approx(0.4)


def test_diff_scoreboards_handles_zero_values(tmp_path):
    from eval.tools.parity import diff_scoreboards

    left = _write_csv(tmp_path / "left.csv", [
        {"evaluator": "surface", "metric": "zero_metric", "value": 0.0, "unit": "x"},
    ])
    right = _write_csv(tmp_path / "right.csv", [
        {"evaluator": "surface", "metric": "zero_metric", "value": 0.0, "unit": "x"},
    ])
    report = diff_scoreboards(left, right)
    d = report.shared[0]
    # denom would divide by zero; should fall back to 0.0 rel_diff cleanly.
    assert d.abs_diff == 0.0
    assert d.rel_diff == 0.0


def test_diff_scoreboards_partitions_only_left_only_right(tmp_path):
    from eval.tools.parity import diff_scoreboards

    left = _write_csv(tmp_path / "left.csv", [
        {"evaluator": "surface", "metric": "only_left_a", "value": 0.1, "unit": "u"},
        {"evaluator": "surface", "metric": "shared", "value": 0.2, "unit": "u"},
    ])
    right = _write_csv(tmp_path / "right.csv", [
        {"evaluator": "surface", "metric": "shared", "value": 0.3, "unit": "u"},
        {"evaluator": "tc", "metric": "only_right_b", "value": 0.4, "unit": "u"},
    ])
    report = diff_scoreboards(left, right)
    assert [(d.evaluator, d.metric) for d in report.shared] == [("surface", "shared")]
    assert report.only_left == [("surface", "only_left_a")]
    assert report.only_right == [("tc", "only_right_b")]


def test_tolerance_absolute_filters_outside(tmp_path):
    from eval.tools.parity import diff_scoreboards

    left = _write_csv(tmp_path / "left.csv", [
        {"evaluator": "surface", "metric": "a", "value": 1.0, "unit": "u"},
        {"evaluator": "surface", "metric": "b", "value": 1.0, "unit": "u"},
    ])
    right = _write_csv(tmp_path / "right.csv", [
        {"evaluator": "surface", "metric": "a", "value": 1.001, "unit": "u"},  # within
        {"evaluator": "surface", "metric": "b", "value": 2.0, "unit": "u"},    # outside
    ])
    report = diff_scoreboards(left, right, tolerance=0.01, tolerance_kind="absolute")
    assert report.n_within == 1
    assert report.n_outside == 1
    outside = report.outside_tolerance()
    assert len(outside) == 1
    assert outside[0].metric == "b"


def test_tolerance_relative_uses_relative_diff(tmp_path):
    from eval.tools.parity import diff_scoreboards

    left = _write_csv(tmp_path / "left.csv", [
        {"evaluator": "x", "metric": "a", "value": 100.0, "unit": "u"},
    ])
    right = _write_csv(tmp_path / "right.csv", [
        {"evaluator": "x", "metric": "a", "value": 101.0, "unit": "u"},
    ])
    # abs_diff = 1.0, rel_diff = 1 / 100.5 ≈ 0.00995. Tolerance 0.5 abs would say outside;
    # tolerance 0.05 rel says within.
    report = diff_scoreboards(left, right, tolerance=0.5, tolerance_kind="absolute")
    assert report.n_outside == 1

    report = diff_scoreboards(left, right, tolerance=0.05, tolerance_kind="relative")
    assert report.n_outside == 0


def test_tolerance_kind_validated(tmp_path):
    from eval.tools.parity import diff_scoreboards

    left = _write_csv(tmp_path / "left.csv", [{"evaluator": "x", "metric": "a", "value": 0.0}])
    right = _write_csv(tmp_path / "right.csv", [{"evaluator": "x", "metric": "a", "value": 0.0}])
    with pytest.raises(ValueError, match="Unsupported tolerance_kind"):
        diff_scoreboards(left, right, tolerance=0.1, tolerance_kind="silly")


def test_missing_columns_raises(tmp_path):
    from eval.tools.parity import diff_scoreboards

    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    good = _write_csv(tmp_path / "good.csv", [{"evaluator": "x", "metric": "y", "value": 0.0}])
    with pytest.raises(ValueError, match="missing required columns"):
        diff_scoreboards(bad, good)


def test_per_evaluator_summary(tmp_path):
    from eval.tools.parity import diff_scoreboards

    left = _write_csv(tmp_path / "left.csv", [
        {"evaluator": "surface", "metric": "a", "value": 1.0, "unit": "u"},
        {"evaluator": "surface", "metric": "b", "value": 1.0, "unit": "u"},
        {"evaluator": "tc",      "metric": "c", "value": 5.0, "unit": "u"},
    ])
    right = _write_csv(tmp_path / "right.csv", [
        {"evaluator": "surface", "metric": "a", "value": 1.5, "unit": "u"},
        {"evaluator": "surface", "metric": "b", "value": 1.05, "unit": "u"},
        {"evaluator": "tc",      "metric": "c", "value": 5.05, "unit": "u"},
    ])
    report = diff_scoreboards(left, right, tolerance=0.1, tolerance_kind="absolute")
    summary = report.per_evaluator_summary()
    assert set(summary.keys()) == {"surface", "tc"}
    assert summary["surface"]["n"] == 2
    assert summary["surface"]["max_abs"] == pytest.approx(0.5)
    assert summary["surface"]["mean_abs"] == pytest.approx(0.275)
    assert summary["surface"]["n_outside"] == 1  # only metric `a` (abs=0.5 > 0.1) is outside
    assert summary["tc"]["n_outside"] == 0


def test_render_markdown_report_contains_key_sections(tmp_path):
    from eval.tools.parity import diff_scoreboards, render_markdown_report

    left = _write_csv(tmp_path / "left.csv", [
        {"evaluator": "surface", "metric": "surface_weighted_nmse", "value": 0.10, "unit": "nmse"},
        {"evaluator": "surface", "metric": "surface_10u_nmse",      "value": 0.13, "unit": "nmse"},
    ])
    right = _write_csv(tmp_path / "right.csv", [
        {"evaluator": "surface", "metric": "surface_weighted_nmse", "value": 0.12, "unit": "nmse"},
        {"evaluator": "surface", "metric": "surface_10u_nmse",      "value": 0.15, "unit": "nmse"},
    ])
    report = diff_scoreboards(left, right, tolerance=0.01)
    md = render_markdown_report(report, left_label="manual", right_label="prepml")

    assert "manual" in md and "prepml" in md
    assert "Scoreboard diff" in md
    assert "Per-evaluator summary" in md
    # Top-N table includes shared metrics
    assert "surface_weighted_nmse" in md
    assert "surface_10u_nmse" in md
    assert "shared metrics: 2" in md
    assert "tolerance (absolute): 0.01" in md


def test_render_markdown_report_only_outside_filter(tmp_path):
    from eval.tools.parity import diff_scoreboards, render_markdown_report

    left = _write_csv(tmp_path / "left.csv", [
        {"evaluator": "surface", "metric": "within_band", "value": 1.0},
        {"evaluator": "surface", "metric": "outside_band", "value": 1.0},
    ])
    right = _write_csv(tmp_path / "right.csv", [
        {"evaluator": "surface", "metric": "within_band", "value": 1.001},
        {"evaluator": "surface", "metric": "outside_band", "value": 5.0},
    ])
    report = diff_scoreboards(left, right, tolerance=0.01)
    md_all = render_markdown_report(report, only_outside=False, top_n=10)
    md_outside = render_markdown_report(report, only_outside=True, top_n=10)
    assert "within_band" in md_all
    assert "outside_band" in md_all
    assert "within_band" not in md_outside
    assert "outside_band" in md_outside
