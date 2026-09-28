"""storm_maps runner: the box and lead times are configurable, and the defaults are unchanged."""
from __future__ import annotations

import pytest

from eval.evaluators.storm_maps.core import render as backend
from eval.evaluators.storm_maps import runner

# The values that were hard-coded before the keys became configurable.
TODAY_EVENT_BOX = (5.0, 35.0, -100.0, -40.0)
TODAY_STORM_BOX = (10.0, 35.0, -100.0, -80.0)
TODAY_STEP = "072"


@pytest.fixture()
def calls(monkeypatch):
    seen = []

    def fake_render(predictions_dir, out_dir, **kwargs):
        seen.append({"out_dir": out_dir, **kwargs})
        return out_dir

    monkeypatch.setattr(runner, "render", fake_render)
    return seen


def _run(tmp_path, lane, eval_config, **kwargs):
    pred = tmp_path / "predictions"
    pred.mkdir(exist_ok=True)
    out = tmp_path / "out"
    runner.run(pred, lane, eval_config, output_dir=out, overwrite=True, **kwargs)
    return out


def test_defaults_are_the_values_that_were_hard_coded(tmp_path, calls):
    out = _run(tmp_path, {}, {})
    assert len(calls) == 1
    call = calls[0]
    assert call["out_dir"] == out
    assert tuple(call["event_box"]) == TODAY_EVENT_BOX
    assert tuple(call["storm_box"]) == TODAY_STORM_BOX
    assert call["step"] == TODAY_STEP
    assert call["event_name"] == "storm"


def test_an_empty_or_missing_block_changes_nothing(tmp_path, calls):
    _run(tmp_path, {}, {})
    _run(tmp_path, {"storm_maps": {}}, {})
    _run(tmp_path, {}, {"box": None, "steps": None, "storm_box": None})
    keys = ("event_box", "storm_box", "step", "event_name")
    for other in calls[1:]:
        assert {k: other[k] for k in keys} == {k: calls[0][k] for k in keys}


def test_the_tc_block_still_sets_the_storm_search_box(tmp_path, calls):
    _run(tmp_path, {"tc": {"events": ["idalia"], "storm_box": [20, 30, -90, -80]}}, {})
    assert tuple(calls[0]["storm_box"]) == (20.0, 30.0, -90.0, -80.0)
    assert tuple(calls[0]["event_box"]) == TODAY_EVENT_BOX
    assert calls[0]["event_name"] == "idalia"


def test_the_backend_command_line_defaults_match_the_runner_defaults(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(backend, "render", lambda pdir, out, **kw: seen.update(kw) or out)
    backend.main([str(tmp_path / "predictions"), "--out", str(tmp_path / "o")])
    assert tuple(seen["event_box"]) == runner._DEFAULT_BOX == TODAY_EVENT_BOX
    assert tuple(seen["storm_box"]) == runner._DEFAULT_STORM == TODAY_STORM_BOX
    assert seen["step"] == TODAY_STEP == runner._DEFAULT_STEPS[0]


def test_the_backend_function_defaults_match_the_runner_defaults():
    import inspect

    params = inspect.signature(backend.render).parameters
    assert tuple(params["event_box"].default) == TODAY_EVENT_BOX
    assert params["step"].default == TODAY_STEP


def test_box_can_be_given_as_a_mapping_or_a_list(tmp_path, calls):
    _run(tmp_path, {}, {"box": {"lat_min": 0, "lat_max": 30, "lon_min": -80, "lon_max": -50}})
    _run(tmp_path, {}, {"box": [1, 31, -81, -51], "storm_box": [2, 20, -75, -60]})
    assert tuple(calls[0]["event_box"]) == (0.0, 30.0, -80.0, -50.0)
    assert tuple(calls[0]["storm_box"]) == TODAY_STORM_BOX
    assert tuple(calls[1]["event_box"]) == (1.0, 31.0, -81.0, -51.0)
    assert tuple(calls[1]["storm_box"]) == (2.0, 20.0, -75.0, -60.0)


def test_one_configured_step_writes_in_the_evaluator_folder(tmp_path, calls):
    out = _run(tmp_path, {}, {"steps": [24]})
    assert [c["step"] for c in calls] == ["024"]
    assert calls[0]["out_dir"] == out


def test_several_steps_get_one_folder_each(tmp_path, calls):
    out = _run(tmp_path, {}, {"steps": [48, "072", 48]})
    assert [c["step"] for c in calls] == ["048", "072"]
    assert [c["out_dir"] for c in calls] == [out / "step048", out / "step072"]


@pytest.mark.parametrize("bad", [
    {"box": {"lat_min": 30, "lat_max": 0, "lon_min": -80, "lon_max": -50}},
    {"box": [1, 2, 3]},
    {"box": {"lat_min": 0}},
    {"storm_box": "north atlantic"},
    {"steps": []},
    {"steps": ["day two"]},
])
def test_a_malformed_setting_is_refused_with_a_clear_message(tmp_path, calls, bad):
    with pytest.raises(ValueError, match="storm_maps"):
        _run(tmp_path, {}, bad)
    assert calls == []


def test_describe_lists_the_new_keys(capsys):
    from eval import cli

    cli.main(["describe", "storm_maps"])
    text = capsys.readouterr().out
    for key in ("box", "storm_box", "steps"):
        assert key in text
    assert "lat_min" in text
