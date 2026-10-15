"""wind_extremes on O320 files: input from x_interp, no O1280 matrix (o96->o320 big eval)."""
from __future__ import annotations

import json

import netCDF4
import numpy as np
import pytest

from eval.evaluators.wind_extremes import runner as W
from eval.tests.synthetic_predictions import octahedral_grid, write_predictions_file


@pytest.fixture(scope="module")
def o320_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("we") / "predictions"
    write_predictions_file(d / "predictions_20220819_step060.nc", states=("10u", "10v", "2t"),
                           members=(1, 2), seed=3)
    return d


def test_x_interp_input_on_the_real_o320_grid(o320_dir, tmp_path):
    cfg = {"input_source": "x_interp", "radii_km": [20, 50, 100], "adjacency_km": 40,
           "boxes": {"europe": [35.0, 62.0, -12.0, 25.0]}, "steps": [60]}
    W.run(o320_dir, {}, cfg, output_dir=tmp_path)
    payload = json.loads((tmp_path / "wind_extremes.json").read_text())
    assert payload["config"]["input_source"] == "x_interp"
    s = payload["samples"][0]
    # the input peak is the maximum of hypot(x_interp) over the core box
    with netCDF4.Dataset(o320_dir / "predictions_20220819_step060.nc") as d:
        lat, lon = d["lat_hres"][:], d["lon_hres"][:]
        u, v = (np.asarray(d["x_interp"][0, 0, :, j], dtype=np.float64) for j in (0, 1))
    core = (lat >= 35) & (lat <= 62) & (lon >= -12) & (lon <= 25)
    assert s["input"]["peak"] == pytest.approx(np.hypot(u, v)[core].max())
    # a 20 km disk holds the point alone on O320 (nearest neighbour 32 km): retention 1
    assert s["model"]["local_retention"]["20"] == pytest.approx(1.0)
    # a 40 km adjacency connects the O320 neighbours
    assert s["model"]["patch_points_90pct"] >= 1


def test_default_o1280_matrix_is_refused_on_o320(o320_dir, tmp_path):
    import scipy.sparse as sps

    m = tmp_path / "up_o320_o1280.npz"
    sps.save_npz(m, sps.csr_matrix((6599680 // 1000, 421120 // 1000)))   # wrong point count
    with pytest.raises(RuntimeError, match="grid mismatch"):
        W.run(o320_dir, {}, {"paths": {"up_matrix": str(m)}, "radii_km": [50],
                             "boxes": {"europe": [35.0, 62.0, -12.0, 25.0]}},
              output_dir=tmp_path / "out")
