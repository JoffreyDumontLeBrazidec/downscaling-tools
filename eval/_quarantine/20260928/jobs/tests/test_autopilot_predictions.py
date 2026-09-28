"""Tests of the retired autopilot_predictions.py, moved out of
eval/jobs/tests/test_predictions_jobs.py when the autopilot jobs were quarantined
on 2026-09-28. Not collected (pytest.ini norecursedirs)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_autopilot_write_state(tmp_path: Path):
    mod = _load_module(
        "autopred_state",
        ROOT / "eval/_quarantine/20260928/jobs/autopilot_predictions.py",
    )
    state_file = tmp_path / "state.json"
    jobs = {
        "predict25": mod.JobTrack(
            name="predict25",
            script=Path("/tmp/predict.sbatch"),
            job_id="111",
            state="RUNNING",
            retries=0,
            max_retries=1,
        ),
        "eval25": mod.JobTrack(
            name="eval25",
            script=Path("/tmp/eval.sbatch"),
            dependency="predict25",
            job_id="222",
            state="PENDING",
            retries=0,
            max_retries=1,
        ),
    }
    mod._write_state(state_file, "manualabcd", mod.PHASE_PROXY, jobs)
    text = state_file.read_text(encoding="utf-8")
    assert "\"run_id\": \"manualabcd\"" in text
    assert f"\"phase\": \"{mod.PHASE_PROXY}\"" in text
    assert "\"predict25\"" in text
    assert "\"eval25\"" in text


def test_autopilot_rejects_unsafe_run_id(monkeypatch):
    mod = _load_module(
        "autopred_reject_unsafe_id",
        ROOT / "eval/_quarantine/20260928/jobs/autopilot_predictions.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autopilot_predictions.py",
            "--run-id",
            "bad/run",
        ],
    )
    with pytest.raises(SystemExit, match="unsafe characters"):
        mod.main()


def test_autopilot_rejects_eval_root_that_looks_like_run(monkeypatch, tmp_path: Path):
    mod = _load_module(
        "autopred_reject_eval_root",
        ROOT / "eval/_quarantine/20260928/jobs/autopilot_predictions.py",
    )
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autopilot_predictions.py",
            "--run-id",
            "manualabcd",
            "--eval-root",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit, match="looks like a run directory"):
        mod.main()
