"""Regional (cut-graph) files give the same texture statistics as the global run
on the same interior points."""
from __future__ import annotations

import json

import numpy as np
import pytest

from eval.evaluators.regional_test_support import make_world, write_prediction
from eval.evaluators.texture import runner

pytest.importorskip("netCDF4")
pytest.importorskip("zarr")


def _samples(results_dir, stratum, state="10u"):
    payload = json.loads((results_dir / "texture.json").read_text())
    rows = [r for r in payload["samples"] if r["stratum"] == stratum and r["state"] == state]
    assert rows, f"no samples for stratum {stratum!r}"
    return payload, rows


@pytest.mark.slow
def test_regional_matches_global_on_interior(tmp_path):
    world = make_world(tmp_path / "world")
    margin = 8.0   # > one coarse cell (4 deg) + the interpolation stencil
    gdir, rdir = tmp_path / "global", tmp_path / "regional"
    gdir.mkdir()
    rdir.mkdir()
    write_prediction(gdir / "predictions_20230830_step024.nc", world, 2, seed=1, cut=False)
    sel = write_prediction(rdir / "predictions_20230830_step024.nc", world, 2, seed=1, cut=True)

    lat_h, lon_h = world["lat_h"], world["lon_h"]
    lon180 = np.where(lon_h > 180.0, lon_h - 360.0, lon_h)
    lat_lo, lat_hi = float(lat_h[sel].min()), float(lat_h[sel].max())
    lon_lo, lon_hi = float(lon180[sel].min()), float(lon180[sel].max())
    interior = [lat_lo + margin, lat_hi - margin, lon_lo + margin, lon_hi - margin]

    common = {"weather_states": ["10u"], "paths": world["paths"], "nn_count": 6}
    g_out = runner.run(gdir, {}, {**common, "regions": {"interior": interior}},
                       output_dir=gdir / "evaluators" / "texture")
    r_out = runner.run(rdir, {}, {**common, "regions": {}, "edge_margin_deg": margin},
                       output_dir=rdir / "evaluators" / "texture")

    g_pay, g_rows = _samples(g_out, "interior")
    r_pay, r_rows = _samples(r_out, "all")
    assert g_pay["regional"] is False and r_pay["regional"] is True
    assert r_pay["driver_source"] == "x_interp"
    assert r_pay["grid"]["regional"]["n_global"] == lat_h.size
    assert r_pay["grid"]["n_points"] == sel.size

    def key(r):
        return (r["date"], r["step"], r["member"])

    g_by, r_by = {key(r): r for r in g_rows}, {key(r): r for r in r_rows}
    assert set(g_by) == set(r_by) and len(g_by) == 2
    for k in g_by:
        for side in ("truth", "model"):
            g, r = g_by[k][side], r_by[k][side]
            assert g["n_points"] == r["n_points"]
            for stat in runner.STAT_NAMES:
                assert g[stat] == pytest.approx(r[stat], rel=2e-5, abs=1e-7), (k, side, stat)


def test_regional_file_without_x_interp_is_refused(tmp_path):
    import netCDF4

    world = make_world(tmp_path / "world")
    rdir = tmp_path / "regional"
    rdir.mkdir()
    f = rdir / "predictions_20230830_step024.nc"
    write_prediction(f, world, 1, seed=3, cut=True)
    # Strip x_interp by rewriting the file without it.
    src = netCDF4.Dataset(f)
    g = rdir / "tmp.nc"
    with netCDF4.Dataset(g, "w") as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim))
        for name, var in src.variables.items():
            if name == "x_interp":
                continue
            v = dst.createVariable(name, var.dtype, var.dimensions)
            v[:] = var[:]
        dst.setncattr("lead_step_hours", 24)
    src.close()
    g.replace(f)
    with pytest.raises(RuntimeError, match="x_interp"):
        runner.run(rdir, {}, {"weather_states": ["10u"], "paths": world["paths"], "regions": {}},
                   output_dir=rdir / "evaluators" / "texture")
