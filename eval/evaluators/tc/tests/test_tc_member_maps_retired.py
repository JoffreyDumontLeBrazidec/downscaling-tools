"""The tc per-member maps were retired on 2026-09-30 (eval/_quarantine/20260930/).

A lane that still sets tc.member_maps.enabled must run unedited: tc warns once,
draws nothing and does not fail. The heavy parts of run() (prediction discovery,
curve loading, contracts) are stubbed; what is exercised is run() itself.
"""
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from eval.evaluators.tc import runner
from eval.evaluators.tc.core import workflows


def _stub_heavy_parts(monkeypatch, tmp_path):
    pred = tmp_path / "predictions_20230828_step024.nc"
    files = [(pred, 20230828, 24)]
    curve = SimpleNamespace(support_mode="native", support_signature="stub")
    monkeypatch.setattr(runner, "_pred_files_as_tuples", lambda _d: files)
    monkeypatch.setattr(runner, "select_prediction_files_for_event", lambda _f, _e: files)
    monkeypatch.setattr(runner, "build_prediction_contract", lambda **_k: {"stub": True})
    monkeypatch.setattr(runner, "load_curves_for_event", lambda *_a, **_k: {"model": curve})
    monkeypatch.setattr(runner, "load_prediction_curves", lambda *_a, **_k: curve)
    monkeypatch.setattr(runner, "validate_curve_support_contract", lambda *_a: None)
    monkeypatch.setattr(runner, "event_days_steps", lambda _f: ([28], [24]))


def test_enabled_member_maps_block_warns_once_and_draws_nothing(tmp_path, monkeypatch, caplog):
    _stub_heavy_parts(monkeypatch, tmp_path)
    eval_config = {
        "events": ["idalia"],
        "support_mode": "native",
        "member_maps": {
            "enabled": True,
            "event_dates": {"idalia": "20230828"},
            "steps": [24],
            "combined_pdf": True,
        },
    }
    out = tmp_path / "evaluators" / "tc"
    with caplog.at_level(logging.WARNING, logger=runner.LOG.name):
        result = runner.run(tmp_path, {}, eval_config, output_dir=out, run_label="demo")

    assert result == out
    assert (out / "stats.json").is_file()
    assert not (out / "member_maps").exists()
    assert not list(tmp_path.rglob("tc_members_*"))
    retired = [r for r in caplog.records if "2026-09-30" in r.getMessage()]
    assert len(retired) == 1
    assert "zoom_maps" in retired[0].getMessage()
    assert not hasattr(runner, "run_member_maps")
    assert not hasattr(workflows, "run_member_maps")


@pytest.mark.parametrize("block", [None, {}, {"enabled": False}])
def test_absent_or_disabled_block_is_silent(block, caplog):
    config = {} if block is None else {"member_maps": block}
    with caplog.at_level(logging.WARNING, logger=runner.LOG.name):
        assert runner._warn_if_member_maps_requested(config) is False
    assert not caplog.records


def test_member_maps_subcommand_is_a_tombstone(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "workflows.py", "member-maps", "--predictions-dir", "/x", "--outdir", "/y",
        "--run-label", "demo", "--date", "20230828",
    ])
    with pytest.raises(SystemExit) as exc:
        workflows.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "2026-09-30" in err and "zoom_maps" in err
