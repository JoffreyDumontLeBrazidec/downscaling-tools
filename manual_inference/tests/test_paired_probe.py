import numpy as np

from manual_inference.prediction import paired_probe as pp


def _grid(n):
    lat = np.random.default_rng(0).uniform(40, 50, n)
    lon = np.random.default_rng(1).uniform(0, 10, n)
    return lat, lon


def test_operator_rows_sum_to_one_and_constant_preserved():
    lat_h, lon_h = _grid(4000)
    lat_l, lon_l = _grid(250)
    D = pp.gaussian_knn_operator(lat_h, lon_h, lat_l, lon_l, 18.0, 32)
    assert np.allclose(np.asarray(D.sum(1)).ravel(), 1.0)
    assert np.allclose(D @ np.full(4000, 3.0), 3.0)


def test_replace_inputs_keeps_forcings(tmp_path):
    lat_h, lon_h = _grid(4000)
    lat_l, lon_l = _grid(250)
    D = pp.gaussian_knn_operator(lat_h, lon_h, lat_l, lon_l, 18.0, 32)
    x = np.random.rand(1, 2, 250, 3).astype(np.float32)
    y = np.random.rand(1, 2, 4000, 2).astype(np.float32)
    nin = {"10u": 0, "cos_lat": 1, "msl": 2}
    nout = {"msl": 0, "10u": 1}
    xt, shared = pp.replace_inputs(x, y, D, nin, nout)
    assert set(shared) == {"msl", "10u"}
    assert np.array_equal(xt[..., 1], x[..., 1])
    assert np.allclose(xt[0, 1, :, 2], D @ y[0, 1, :, 0], atol=1e-6)
    pp.scale_gap(x, xt, lat_l, lon_l, shared, nin, tmp_path / "g.npz")
    g = np.load(tmp_path / "g.npz")
    assert g["var_real"].shape == (1, 2, 4, 2)
