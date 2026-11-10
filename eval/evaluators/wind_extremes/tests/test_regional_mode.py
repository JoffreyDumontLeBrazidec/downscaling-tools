"""Regional (cut-graph) files: input wind from x_interp, boxes clipped to the file,
boxes outside it skipped; statistics identical to the global run on a box inside."""
from __future__ import annotations

import json

import pytest

from eval.evaluators.regional_test_support import make_world, write_prediction
from eval.evaluators.wind_extremes import runner

pytest.importorskip("netCDF4")
pytest.importorskip("zarr")

INNER = [15.0, 35.0, -95.0, -63.0]   # inside the cut with room for the disk padding


def _payload(out_dir):
    return json.loads((out_dir / "wind_extremes.json").read_text())


def test_regional_matches_global_inside_the_cut(tmp_path):
    world = make_world(tmp_path / "world")
    gdir, rdir = tmp_path / "global", tmp_path / "regional"
    gdir.mkdir()
    rdir.mkdir()
    write_prediction(gdir / "predictions_20230830_step024.nc", world, 2, seed=1, cut=False)
    write_prediction(rdir / "predictions_20230830_step024.nc", world, 2, seed=1, cut=True)
    paths = {"up_matrix": world["paths"]["up_matrix"]}
    cfg = {"boxes": {"inner": INNER}, "radii_km": [120.0, 240.0], "adjacency_km": 120.0,
           "paths": paths}
    g = _payload(runner.run(gdir, {}, cfg, output_dir=gdir / "wx"))
    r = _payload(runner.run(rdir, {}, cfg, output_dir=rdir / "wx"))
    assert g["config"]["regional"] is False
    assert r["config"]["regional"] is True and r["config"]["input_source"] == "x_interp"
    assert g["config"]["point_area_km2"] == pytest.approx(r["config"]["point_area_km2"])
    assert g["boxes"]["inner"]["n_core_points"] == r["boxes"]["inner"]["n_core_points"]

    def key(s):
        return (s["date"], s["step"], s["member"], s["box"])

    gs, rs = {key(s): s for s in g["samples"]}, {key(s): s for s in r["samples"]}
    assert set(gs) == set(rs) and len(gs) == 2
    for k in gs:
        for source in ("model", "truth", "input"):
            a, b = gs[k][source], rs[k][source]
            assert a["peak"] == pytest.approx(b["peak"], rel=1e-5), (k, source)
            assert a["peak_lat"] == b["peak_lat"] and a["peak_lon"] == b["peak_lon"]
            assert a["patch_points_90pct"] == b["patch_points_90pct"]
            for rad, val in a["retention"].items():
                assert val == pytest.approx(b["retention"][rad], rel=1e-5), (k, source, rad)


def test_regional_skips_boxes_outside_and_clips_partial_ones(tmp_path):
    world = make_world(tmp_path / "world")
    rdir = tmp_path / "regional"
    rdir.mkdir()
    write_prediction(rdir / "predictions_20230830_step024.nc", world, 1, seed=2, cut=True)
    cfg = {"radii_km": [120.0], "adjacency_km": 120.0,
           "paths": {"up_matrix": world["paths"]["up_matrix"]}}   # default boxes
    r = _payload(runner.run(rdir, {}, cfg, output_dir=rdir / "wx"))
    assert sorted(r["config"]["boxes_skipped"]) == ["europe", "open_north_atlantic"]
    assert list(r["boxes"]) == ["west_tropical_atlantic"]
    # [10, 32, -85, -55] clipped to the cut's east edge at -58: 22 rows x 27 columns.
    assert r["boxes"]["west_tropical_atlantic"]["n_core_points"] == 22 * 27
