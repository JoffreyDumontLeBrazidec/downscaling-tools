"""Tests for the lean run-root layout projection (eval.lean_layout)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import lean_layout


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _fake_run(root: Path, *, nested: bool = False) -> Path:
    """Build a fake evaluator tree. Returns the output_dir passed to the harness.

    nested=False: output_dir IS the run root (canonical).
    nested=True:  output_dir is <run_root>/eval_plots with run-root siblings.
    """
    run_root = root
    output_dir = run_root / "eval_plots" if nested else run_root
    if nested:
        # run-root siblings that mark `root` as the real run root
        (run_root / "manual").mkdir(parents=True, exist_ok=True)
        (run_root / "prepml").mkdir(parents=True, exist_ok=True)

    ev = output_dir / "evaluators"
    # spectra: declared rename + per-var plots subdir + metrics
    _write(ev / "spectra" / "all_spectra_proxy.pdf")
    _write(ev / "spectra" / "plots" / "spectra_2t.pdf")
    _write(ev / "spectra" / "metrics.json", json.dumps({"spectra_mean_relative_l2": 0.12}))
    # tc: deep consolidated PDF under plots/ + member_maps + stats
    _write(ev / "tc" / "plots" / "all_tc_distributions.pdf")
    _write(ev / "tc" / "member_maps" / "tc_members_idalia.pdf")
    _write(ev / "tc" / "stats.json", "{}")
    _write(ev / "tc" / "metrics.json", json.dumps({"tc_extreme_score": 0.45}))
    # region_plot: convention only (root-level pdf, no spec)
    _write(ev / "region_plot" / "all_regions_plots.pdf")
    # surface: data-only, no plot/pdf
    _write(ev / "surface" / "surface_loss.json", "{}")
    return output_dir


# --------------------------------------------------------------------------- #
# resolve_run_root
# --------------------------------------------------------------------------- #
def test_resolve_run_root_canonical(tmp_path: Path):
    (tmp_path / "evaluators").mkdir()
    assert lean_layout.resolve_run_root(tmp_path) == tmp_path


def test_resolve_run_root_data_subdir(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    assert lean_layout.resolve_run_root(data) == tmp_path


def test_resolve_run_root_nested_analysis_dir(tmp_path: Path):
    """eval_plots/ with run-root siblings must resolve to the PARENT run root."""
    output_dir = _fake_run(tmp_path, nested=True)
    assert output_dir.name == "eval_plots"
    assert lean_layout.resolve_run_root(output_dir) == tmp_path


def test_resolve_run_root_rejects_unrelated(tmp_path: Path):
    with pytest.raises(ValueError):
        lean_layout.resolve_run_root(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# project_lean_layout
# --------------------------------------------------------------------------- #
def test_projection_canonical_layout(tmp_path: Path):
    output_dir = _fake_run(tmp_path, nested=False)
    run_root = lean_layout.project_lean_layout(output_dir)
    assert run_root == tmp_path

    # Top-level deliverables: declared renames + convention promotions.
    assert (run_root / "spectra_proxy.pdf").is_symlink()
    assert (run_root / "tc_pdf_distributions.pdf").is_symlink()
    assert (run_root / "all_regions_plots.pdf").is_symlink()  # convention
    # The renamed source must NOT also leak under its raw name.
    assert not (run_root / "all_spectra_proxy.pdf").exists()

    # plots/<name>/ views (default plots/ + extra member_maps subdir for tc).
    assert (run_root / "plots" / "spectra").is_symlink()
    assert (run_root / "plots" / "tc").is_symlink()
    assert (run_root / "plots" / "tc_member_maps").is_symlink()

    # data/<name> -> whole evaluator dir.
    assert (run_root / "data" / "surface").is_symlink()
    assert (run_root / "data" / "spectra" / "metrics.json").exists()

    # Assembled metrics.json merges per-evaluator metrics.
    metrics = json.loads((run_root / "metrics.json").read_text())
    assert metrics["schema_version"] == "1.0"
    assert metrics["evaluators"]["spectra"]["spectra_mean_relative_l2"] == 0.12
    assert metrics["evaluators"]["tc"]["tc_extreme_score"] == 0.45
    assert "spectra_proxy.pdf" in metrics["top_level_deliverables"]


def test_projection_nested_writes_to_true_run_root(tmp_path: Path):
    output_dir = _fake_run(tmp_path, nested=True)
    run_root = lean_layout.project_lean_layout(output_dir)
    assert run_root == tmp_path
    # plots/ and data/ at the TRUE run root, not buried under eval_plots/.
    assert (tmp_path / "plots" / "spectra").is_symlink()
    assert not (output_dir / "plots").exists()
    assert (tmp_path / "tc_pdf_distributions.pdf").is_symlink()


def test_projection_is_idempotent(tmp_path: Path):
    output_dir = _fake_run(tmp_path, nested=False)
    lean_layout.project_lean_layout(output_dir)
    first = sorted(p.name for p in tmp_path.iterdir())
    lean_layout.project_lean_layout(output_dir)
    second = sorted(p.name for p in tmp_path.iterdir())
    assert first == second
    # Re-projection produces a clean snapshot (no stale duplicate links).
    assert (tmp_path / "plots" / "spectra").is_symlink()


def test_projection_refuses_to_clobber_real_top_level_file(tmp_path: Path):
    output_dir = _fake_run(tmp_path, nested=False)
    # A hand-placed real file with the same name as a convention promotion.
    real = tmp_path / "all_regions_plots.pdf"
    real.write_text("HUMAN")
    lean_layout.project_lean_layout(output_dir)
    assert not real.is_symlink()
    assert real.read_text() == "HUMAN"


def test_projection_no_evaluators_is_noop(tmp_path: Path):
    (tmp_path / "predictions").mkdir()
    # Should not raise even with nothing to project.
    assert lean_layout.project_lean_layout(tmp_path) == tmp_path
