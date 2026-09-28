"""``eval.cli list`` and ``eval.cli describe`` are the way an agent finds out what can be evaluated."""
from __future__ import annotations

import json

import pytest

from eval import cli
from eval.evaluators import registry


def _run(argv, capsys):
    cli.main(argv)
    return capsys.readouterr().out


def test_top_level_help_lists_every_command_in_groups(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in cli.commands():
        assert name in out, name
    for title in ("Discovery", "Pipeline", "Comparison", "Tropical cyclone tracks", "Figures", "Maintenance"):
        assert title in out, title


def test_every_command_has_a_group_and_a_summary():
    from eval.cli._common import GROUP_TITLES

    for name, command in cli.commands().items():
        assert command.group in GROUP_TITLES, name
        assert command.summary.strip().endswith("."), name


def test_list_names_every_registered_evaluator_once(capsys):
    out = _run(["list"], capsys)
    for name in registry.names():
        assert f"\n  {name}\n" in out, name
    assert "Atos AC only" in out  # the host constraint of spectra_ecmwf_v2 is shown
    assert "replacement: spectra_ecmwf_v2" in out  # retired names show what replaces them


def test_list_json_matches_the_registry(capsys):
    rows = json.loads(_run(["list", "--json"], capsys))
    assert [r["name"] for r in rows] == registry.names()
    by_name = {r["name"]: r for r in rows}
    assert by_name["tc"]["feeds_scoreboard"] is True
    assert by_name["spectra_ecmwf_v2"]["host_prefix"] == "ac"
    assert by_name["spectra"]["replacement"] == "spectra_ecmwf_v2"


@pytest.mark.parametrize("name", registry.runnable_names())
def test_describe_works_for_every_runnable_evaluator(name, capsys):
    info = json.loads(_run(["describe", name, "--json"], capsys))
    assert info["question"] == registry.get(name).question
    assert info["method"], f"{name} has no package docstring"
    assert info["requires"] and info["example"].startswith("python -m eval.cli evaluate")
    text = _run(["describe", name], capsys)
    for heading in ("Method", "Inputs it needs", "Outputs it writes", "Lane configuration it reads", "Example"):
        assert heading in text, (name, heading)


def test_describe_a_retired_evaluator_shows_the_replacement(capsys):
    text = _run(["describe", "spectra"], capsys)
    assert "retired" in text and "spectra_ecmwf_v2" in text


def test_describe_unknown_name_lists_the_known_ones():
    with pytest.raises(SystemExit) as exc:
        cli.main(["describe", "not_an_evaluator"])
    assert "Known evaluators" in str(exc.value)


def test_describe_reads_lane_keys_from_the_lane_files(capsys):
    info = json.loads(_run(["describe", "tc", "--json"], capsys))
    assert "support_mode" in info["lane_keys_set"]
    assert "support_mode" in info["config_keys_read"]


@pytest.mark.parametrize("name", sorted(cli.commands()))
def test_every_command_renders_its_help(name, capsys):
    """--help must print and exit 0 for every command (membermaps once crashed on a bare percent sign)."""
    with pytest.raises(SystemExit) as exc:
        cli.main([name, "--help"])
    assert exc.value.code == 0
    assert "usage: python -m eval.cli " + name in capsys.readouterr().out
