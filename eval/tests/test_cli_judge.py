"""eval.cli judge: the read skeleton, the shape-box choice, and the small CLI helpers of the
run-judging protocol (--opt, zoom_maps --region)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from eval.cli import judge
from eval.cli._common import parse_opts
from eval.cli.zoom_maps import apply_region

# the part headings and banned phrases of docs scripts/check_run_read.py
PART_RE = re.compile(r"^### (\d)\.", re.M)
BANNED = (r"40\s*-\s*400\s*km distance", r"band[- ]averag")


def _arm(tmp_path: Path, label: str, n_points=1000) -> dict:
    return {"label": label, "dir": tmp_path / label, "eval": tmp_path / label / "evaluators",
            "status": {"shape": "failed: no stats file"},
            "facts": {"n_members": 10, "n_points": n_points, "checkpoint_id": f"ck_{label}",
                      "sampling_config_json": '{"num_steps": 30, "S_churn": 0}'}}


def test_read_without_outputs_has_four_parts_and_says_why(tmp_path):
    arms = {"member": 1, "run": _arm(tmp_path, "h9dm"), "reference": _arm(tmp_path, "donor")}
    text = judge.compose_read(arms, step=24, dates=["20230226"], shape_box="north_atlantic",
                              maps={"Alps": ([], "no file")}, domain="europe", weights=None,
                              out_dir=tmp_path, commit="abc")
    assert text.startswith("## Run read: h9dm")
    assert PART_RE.findall(text) == ["1", "2", "3", "4"]
    for part in re.split(r"^### \d\..*$", text, flags=re.M)[1:]:
        assert "Not available:" in part
    assert "Reference: donor" in text and "S_churn" in text and "Instrument:" in text
    assert not any(re.search(b, text, re.I) for b in BANNED)
    assert "WARNING" not in text


def test_read_warns_on_unequal_instruments(tmp_path):
    arms = {"member": 1, "run": _arm(tmp_path, "a", 6_599_680), "reference": _arm(tmp_path, "b", 1_200_000)}
    arms["reference"]["facts"]["sampling_config_json"] = '{"num_steps": 40}'
    text = judge.compose_read(arms, step=24, dates=["20230226"], shape_box="e2", maps={},
                              domain="europe", weights="EMA-10000 / raw", out_dir=tmp_path, commit="abc")
    assert "different point sets" in text and "sampler settings" in text


@pytest.mark.parametrize("lat,lon,box", [
    ([10.0, 70.0], [280.0, 40.0], "north_atlantic"),     # E2 file, 0..360 longitudes
    ([-89.9, 89.9], [0.0, 359.9], "north_atlantic"),     # global file
    ([10.0, 40.0], [260.0, 302.0], "tc_atlantic"),       # regional TC file
])
def test_shape_box_auto(lat, lon, box):
    assert judge.choose_shape_box("auto", {"lat_range": lat, "lon_range": lon}) == box
    assert judge.choose_shape_box("e2", {}) == "e2"


def test_parse_label_and_opts():
    assert judge.parse_label("h9dm=/x/y", "--run") == ("h9dm", Path("/x/y"))
    with pytest.raises(SystemExit):
        judge.parse_label("/x/y", "--run")
    assert parse_opts(["shape.box=north_atlantic", "zoom_maps.steps=[24]"]) == [
        ("shape", "box", "north_atlantic"), ("zoom_maps", "steps", [24])]
    with pytest.raises(SystemExit):
        parse_opts(["box=north_atlantic"])


def test_zoom_maps_region_sets_the_protocol_extent():
    args = argparse.Namespace(region="alps", extent=[0, 1, 2, 3], region_tag="europe-cutout")
    apply_region(args)
    assert args.extent == [4.0, 16.0, 43.0, 49.0] and args.region_tag == "alps"
