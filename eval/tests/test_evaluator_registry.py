"""The evaluator registry is the one list of evaluators; the CLI and the
scoreboard aggregator derive from it, and retired names are tombstones."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pytest

from eval.evaluators import registry

EVALUATORS_DIR = Path(__file__).resolve().parents[1] / "evaluators"
# Packages under eval/evaluators/ that are not evaluators `evaluate` can run.
NOT_EVALUATORS = {"tctracks"}


def test_every_entry_has_a_group_and_a_question():
    for entry in registry.REGISTRY.values():
        assert entry.group in registry.GROUPS
        assert entry.question.strip().endswith(("?", ".)")), entry.name
        if entry.group == registry.RETIRED:
            assert entry.retired_on
        else:
            assert entry.retired_on is None and entry.replacement is None


def test_only_scored_evaluators_feed_the_scoreboard():
    assert set(registry.scoreboard_names()) == {
        "tc", "surface", "spectra_ecmwf_v2", "precip_scores", "sigma_loss",
    }
    assert set(registry.names(registry.SCORED)) == set(registry.scoreboard_names())


def test_replacements_point_at_runnable_evaluators():
    runnable = set(registry.runnable_names())
    for entry in registry.REGISTRY.values():
        if entry.replacement:
            assert entry.replacement in runnable, entry.name


def test_cli_and_aggregator_derive_from_the_registry():
    from eval import cli
    from eval.scoreboard import aggregator

    assert cli.ALL_EVALUATORS == registry.runnable_names()
    assert aggregator.SCOREBOARD_EVALUATORS == registry.scoreboard_names()
    assert not hasattr(aggregator, "KNOWN_EVALUATORS")


def test_every_runnable_evaluator_imports_and_has_no_duplicate_fields():
    for name in registry.runnable_names():
        mod = importlib.import_module(f"eval.evaluators.{name}")
        spec = getattr(mod, "EVALUATOR_SPEC", {})
        # role fields live in the registry only
        assert "default_enabled" not in spec, name
        assert "scoreboard" not in spec, name
        assert callable(getattr(mod, "run", None)), name


def test_every_evaluator_package_is_registered():
    packages = {
        p.name for p in EVALUATORS_DIR.iterdir()
        if p.is_dir() and (p / "__init__.py").exists()
    }
    assert packages - NOT_EVALUATORS == set(registry.runnable_names())


def test_retired_packages_are_quarantined_and_not_importable():
    quarantine = EVALUATORS_DIR.parent / "_quarantine"
    for name in registry.names(registry.RETIRED):
        entry = registry.get(name)
        assert not (EVALUATORS_DIR / name).exists(), name
        assert (quarantine / entry.retired_on / name / "__init__.py").exists(), name
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"eval.evaluators.{name}")


def test_only_retired_evaluator_is_a_tombstone(tmp_path, capsys):
    from eval import cli

    with pytest.raises(SystemExit) as exc:
        cli.main([
            "evaluate", "--dry-run", "--lane", "o96_o320",
            "--predictions-dir", str(tmp_path), "--only", "spectra",
        ])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "retired" in err and "spectra_ecmwf_v2" in err


def test_lane_group_retired_name_is_skipped_with_warning(caplog):
    import argparse

    from eval import cli

    lane = {"evaluator_groups": {"default": ["tc", "spectra", "surface"],
                                 "diagnostics": ["sigma", "mlflow"]}}
    args = argparse.Namespace(only=None, include_diagnostics=True, expver=None)
    with caplog.at_level(logging.WARNING, logger="eval.cli"):
        resolved = cli._resolve_evaluators(args, lane)
    assert resolved == ["tc", "surface", "mlflow"]
    assert "spectra_ecmwf_v2" in caplog.text and "sigma_loss" in caplog.text


def test_expver_adds_quaver_only():
    import argparse

    from eval import cli

    lane = {"evaluator_groups": {"default": ["tc"]}}
    args = argparse.Namespace(only=None, include_diagnostics=False, expver="abcd")
    assert cli._resolve_evaluators(args, lane) == ["tc", "quaver"]


def test_host_constrained_evaluator_is_skipped_off_host(monkeypatch, tmp_path):
    import socket

    from eval import cli

    monkeypatch.setattr(socket, "gethostname", lambda: "ag6-001")
    assert cli._host_mismatch("spectra_ecmwf_v2")
    assert cli._host_mismatch("tc") is None
    # skipped, not failed, and not counted as a declared-but-missing evaluator
    ran = cli._run_evaluators(tmp_path / "predictions", {}, ["spectra_ecmwf_v2"], tmp_path)
    assert ran == []
    monkeypatch.setattr(socket, "gethostname", lambda: "ac6-100")
    assert cli._host_mismatch("spectra_ecmwf_v2") is None


def test_lane_loader_allows_every_registered_section():
    from eval.config import loader

    assert set(registry.names()) <= loader._LANE_ALLOWED_KEYS
