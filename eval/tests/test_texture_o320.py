"""texture on O320 files (o96->o320 big eval): forcings npz, O96<->O320 pair, x_interp input.

Runs on the real O320 and O96 point sets (eval.tests.synthetic_predictions); the matrices are
built by eval.shared.inter_mat. Nothing from Atos is read.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sps

from eval.evaluators.texture import runner as T
from eval.shared.inter_mat import linear_matrix
from eval.tests.synthetic_predictions import octahedral_grid, write_predictions_file


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("tex")
    gh, gl = octahedral_grid(320), octahedral_grid(96)
    preds = root / "predictions"
    write_predictions_file(preds / "predictions_20220819_step060.nc", grids=(gh, gl),
                           states=("10u", "10v", "2t", "msl", "t_850"), members=(1, 2), seed=5)
    up = root / "up.npz"
    down = root / "down.npz"
    sps.save_npz(up, linear_matrix(gl[0], gl[1], gh[0], gh[1]))
    sps.save_npz(down, linear_matrix(gh[0], gh[1], gl[0], gl[1]))
    lat, lon = gh
    lsm = ((np.sin(np.deg2rad(3 * lon)) > 0.3) & (np.abs(lat) < 70)).astype(np.float32)
    z = (lsm * 9.80665 * (500.0 + 800.0 * np.cos(np.deg2rad(7 * lon)) ** 2)).astype(np.float32)
    np.savez(root / "o320-forcings.npz", lsm=lsm, z=z,
             cos_latitude=np.cos(np.deg2rad(lat)), sin_latitude=np.sin(np.deg2rad(lat)),
             cos_longitude=np.cos(np.deg2rad(lon)), sin_longitude=np.sin(np.deg2rad(lon)))
    np.save(root / "o320_dict.npy", {"stdev": {"10u": 2.0, "10v": 2.0, "2t": 1.5, "msl": 80.0,
                                               "t_850": 1.0}}, allow_pickle=True)
    paths = {"up_matrix": str(up), "down_matrix": str(down), "forcings_npz": str(root / "o320-forcings.npz"),
             "residual_stats": str(root / "o320_dict.npy"), "knn_cache": str(root / "o320_knn32.npz")}
    return root, preds, paths


def test_texture_runs_on_o320_with_npz_forcings_and_x_interp(setup):
    root, preds, paths = setup
    out = root / "out"
    T.run(preds, {}, {"paths": paths, "input_source": "x_interp", "steps": [60],
                      "weather_states": ["10u", "2t"]}, output_dir=out)
    p = json.loads((out / "texture.json").read_text())
    assert p["grid"]["n_points"] == 421120
    assert p["config"]["input_source"] == "x_interp"
    assert 25.0 < p["grid"]["nearest_km_median"] < 40.0          # O320 spacing, not O1280's 9 km
    assert p["strata"]["alps"]["n_points"] > 300
    assert p["n_samples_per_cell"] == 2
    noise = p["noise_reference"]["all"]["fine_lag1_zonal"]["mean"]
    assert noise < 0.0                                            # the high-pass signature
    assert (root / "o320_knn32.npz").exists()                     # cache built for this grid


def test_npz_without_coordinates_uses_the_file_points(setup, tmp_path):
    root, preds, paths = setup
    with np.load(paths["forcings_npz"]) as z:
        np.savez(tmp_path / "bare.npz", lsm=z["lsm"], z=z["z"])
    lat, lon, lsm, oro = T._read_forcings_npz(str(tmp_path / "bare.npz"),
                                              fallback_latlon=octahedral_grid(320))
    assert lat.size == lsm.size == 421120
    with pytest.raises(RuntimeError, match="no coordinates"):
        T._read_forcings_npz(str(tmp_path / "bare.npz"))


def test_o1280_knn_cache_is_refused_on_o320(setup, tmp_path):
    root, preds, paths = setup
    np.savez(tmp_path / "o1280_knn32.npz", idx=np.zeros((10, 32), np.int32),
             dist_km=np.zeros((10, 32), np.float32))
    with pytest.raises(RuntimeError, match="rows but the grid has 421120"):
        T.run(preds, {}, {"paths": {**paths, "knn_cache": str(tmp_path / "o1280_knn32.npz")},
                          "input_source": "x_interp", "weather_states": ["10u"]},
              output_dir=tmp_path / "out")
