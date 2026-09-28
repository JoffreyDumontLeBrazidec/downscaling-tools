"""The finalize_lean_eval_layout.sbatch template (kept in eval/jobs/templates/).

Split out of test_o48_o96_flow_helper.py on 2026-09-28, when the archived
submit_o48_o96_manual_eval_flow.sh helper and its tests were quarantined."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FINALIZE_TEMPLATE = ROOT / "eval/jobs/templates/finalize_lean_eval_layout.sbatch"


def test_finalize_lean_layout_moves_o48_outputs_into_data_without_symlink_clutter(tmp_path: Path):
    run_root = tmp_path / "manual_test_o48"
    run_root.mkdir()
    (run_root / "predictions").mkdir()
    (run_root / "predictions" / "predictions_20250928_step024.nc").write_text("stub\n", encoding="utf-8")
    (run_root / "logs").mkdir()
    (run_root / "bundles_with_y").mkdir()
    (run_root / "EXPERIMENT_CONFIG.yaml").write_text("lane: o48_o96\n", encoding="utf-8")
    (run_root / "o48_o96_metrics.json").write_text("{}\n", encoding="utf-8")
    (run_root / "surface_loss_summary.json").write_text("{}\n", encoding="utf-8")
    for step in ("024", "120"):
        plot_dir = run_root / f"local_plots_regions_step{step}"
        plot_dir.mkdir()
        (plot_dir / "all_regions_plots.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

    # The template reads RUN_ROOT, RUN_ID and CREATE_BACKCOMPAT_SYMLINKS from the environment
    # (each has a "${VAR:-default}" form), so the test sets them instead of editing the text.
    env = {**os.environ, "RUN_ROOT": str(run_root), "RUN_ID": "manual_test_o48",
           "CREATE_BACKCOMPAT_SYMLINKS": "0"}
    result = subprocess.run(
        ["bash", str(FINALIZE_TEMPLATE)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert (run_root / "data" / "predictions").is_dir()
    assert (run_root / "data" / "bundles_with_y").is_dir()
    assert (run_root / "data" / "logs").is_dir()
    assert (run_root / "data" / "o48_o96_metrics.json").is_file()
    assert (run_root / "data" / "surface_loss_summary.json").is_file()
    assert (run_root / "data" / "local_plots_regions_step024").is_dir()
    assert (run_root / "data" / "local_plots_regions_step120").is_dir()
    assert not (run_root / "predictions").exists()
    assert not (run_root / "bundles_with_y").exists()
    assert not (run_root / "logs").exists()
    assert not (run_root / "o48_o96_metrics.json").exists()
    assert not (run_root / "surface_loss_summary.json").exists()
    assert not (run_root / "local_plots_regions_step024").exists()
    assert not (run_root / "local_plots_regions_step120").exists()
    assert (run_root / "local_plots_step024.pdf").is_file()
    assert (run_root / "local_plots_step120.pdf").is_file()
    assert (run_root / "EXPERIMENT_CONFIG.yaml").is_file()
