from __future__ import annotations

from pathlib import Path

import pytest

from eval._backends.tctracker.pipeline import TCTrackerConfig
from eval._backends.tctracker.sources import (
    default_source_specs,
    expand_months,
    parse_sources_arg,
    resolve_source_configs,
    source_id_for,
)

LANE_CONFIG = {
    "tctracker": {"grid": 320},
    "prepml": {
        "input": {"class": "od", "stream": "eefo", "type": "pf", "grid": "O320"},
        "output": {"class": "rd", "stream": "enfo", "type": "pf"},
    },
}


def test_expand_months_daily():
    dates = expand_months("202509")
    assert len(dates) == 30
    assert dates[0] == "20250901" and dates[-1] == "20250930"
    both = expand_months("202508,202509")
    assert len(both) == 61


def test_expand_months_rejects_bad_token():
    with pytest.raises(SystemExit):
        expand_months("2025-09")


def test_default_source_specs_derive_from_prepml_blocks():
    specs = default_source_specs(LANE_CONFIG)
    assert specs["input"] == {"class": "od", "stream": "eefo", "type": "pf", "expver": "0001"}
    # target = OPERATIONAL archive of the output stream, class od
    assert specs["target"]["class"] == "od"
    assert specs["target"]["stream"] == "enfo"
    assert specs["target"]["expver"] == "0001"


def test_lane_tctracker_sources_override_wins():
    lane = dict(LANE_CONFIG)
    lane["tctracker"] = {"grid": 320, "sources": {"target": {"members": [1, 2]}}}
    specs = default_source_specs(lane)
    assert specs["target"]["members"] == [1, 2]
    assert specs["target"]["stream"] == "enfo"


def test_parse_sources_arg():
    assert parse_sources_arg("model,ctrl=j95z,target,input") == {
        "model": None, "ctrl": "j95z", "target": None, "input": None,
    }
    assert parse_sources_arg(None) == {}


def test_resolve_source_configs_share_support_and_split_roots(tmp_path):
    base = TCTrackerConfig(
        lane="o320_o1280_prepml_pw40_season", expver="j9f3",
        output_dir=tmp_path / "eval" / "o320_o1280" / "tctracker" / "j9f3",
        dates=("20250901",), members=(1, 2), grid=320,
    )
    host = {"scratch_root": str(tmp_path)}
    roles = parse_sources_arg("model,ctrl=j95z,target,input")
    resolved = resolve_source_configs(base, roles, LANE_CONFIG, host)
    by_role = {role: (sid, cfg) for role, sid, cfg in resolved}

    assert by_role["model"][1] is base
    sid, ctrl = by_role["ctrl"]
    assert sid == "j95z" and ctrl.fdb_class == "rd" and ctrl.expver == "j95z"
    assert str(ctrl.output_dir).endswith("eval/o320_o1280/tctracker/j95z")

    sid, target = by_role["target"]
    assert sid == "od_enfo_0001"
    assert target.fdb_class == "od" and target.stream == "enfo"
    assert str(target.output_dir).endswith("eval/tcrefs/od_enfo_0001_o320")

    sid, inp = by_role["input"]
    assert inp.stream == "eefo"
    assert str(inp.output_dir).endswith("eval/tcrefs/od_eefo_0001_o320")

    # one support: every source inherits the base tracker settings
    for _, cfg in by_role.values():
        assert cfg.grid == 320 and cfg.dates == base.dates and cfg.members == base.members


def test_resolve_source_configs_missing_expver_is_explicit(tmp_path):
    base = TCTrackerConfig(
        lane="l", expver="x", output_dir=tmp_path, dates=("20250901",), members=(1,),
    )
    with pytest.raises(SystemExit):
        resolve_source_configs(base, {"ctrl": None}, {"tctracker": {}}, {"scratch_root": str(tmp_path)})


def test_source_id_for_rd_is_bare_expver():
    assert source_id_for("ctrl", {"class": "rd", "expver": "j95z"}) == "j95z"
    assert source_id_for("target", {"class": "od", "stream": "enfo", "expver": "0001"}) == "od_enfo_0001"
