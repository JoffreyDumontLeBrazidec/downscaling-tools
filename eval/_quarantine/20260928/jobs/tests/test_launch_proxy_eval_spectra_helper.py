"""Test of eval/jobs/templates/predictions_dir_spectra.py, moved out of
eval/jobs/tests/test_predictions_jobs.py on 2026-09-28. That helper was deleted when the
spectra templates were archived (commit 71046dd), so the test can no longer run. It is
kept here, not collected (pytest.ini norecursedirs), instead of being deleted."""
from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]


def test_launch_proxy_eval_dry_run_uses_repo_owned_spectra_helper(tmp_path: Path):
    import subprocess
    import uuid

    run_id = f"proxy_{uuid.uuid4().hex[:8]}"
    script = ROOT / "eval/archive/jobs/launch_proxy_eval.sh"
    eval_root = tmp_path / "eval_root"
    run_dir = eval_root / run_id
    generated_dir = run_dir / "jobs"

    out = subprocess.check_output(
        [
            str(script),
            "--run-id",
            run_id,
            "--eval-root",
            str(eval_root),
            "--ckpt-id",
            "4a5b2f1b24b84c52872bfcec1410b00f",
            "--write-scoreboard-artifacts",
            "--dry-run",
        ],
        text=True,
    )
    assert "Dry run. Scripts written" in out

    evl = generated_dir / f"eval_proxy_{run_id}.sbatch"
    assert evl.exists()

    evl_text = evl.read_text(encoding="utf-8")
    helper_path = ROOT / "eval/jobs/templates/predictions_dir_spectra.py"
    assert str(helper_path) in evl_text
    assert "/dev/docs/scratch/predictions_dir_spectra.py" not in evl_text
    assert "preserving scheduler success" not in evl_text
    assert "Proxy TC comparison failed with exit code" in evl_text
