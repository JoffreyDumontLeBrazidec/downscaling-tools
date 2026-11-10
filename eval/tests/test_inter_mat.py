"""eval.shared.inter_mat: spherical linear interpolation between octahedral grids."""
from __future__ import annotations

import logging

import numpy as np

from eval.shared.inter_mat import build_pair_from_predictions, linear_matrix, unit_vectors
from eval.tests.synthetic_predictions import octahedral_grid, write_predictions_file


def test_o96_to_o320_rows_sum_to_one_and_interpolate_smooth_fields(caplog):
    (lh, oh), (ll, ol) = octahedral_grid(320), octahedral_grid(96)
    with caplog.at_level(logging.WARNING):
        up = linear_matrix(ll, ol, lh, oh)
    assert "fell back" not in caplog.text
    assert up.shape == (421120, 40320)
    np.testing.assert_allclose(np.asarray(up.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    assert (up.data >= 0).all() and up.getnnz(axis=1).max() <= 3
    # a smooth field on the sphere (cos(lon) alone is singular at the poles)
    xs, xd = unit_vectors(ll, ol), unit_vectors(lh, oh)
    f = xs[:, 2] ** 2 + 0.3 * xs[:, 0]
    g = xd[:, 2] ** 2 + 0.3 * xd[:, 0]
    assert np.abs(up @ f - g).max() < 5e-3


def test_identity_on_coincident_points():
    lat, lon = octahedral_grid(16)
    m = linear_matrix(lat, lon, lat, lon)
    v = np.random.default_rng(0).standard_normal(lat.size)
    np.testing.assert_allclose(m @ v, v, atol=1e-9)


def test_pair_from_a_prediction_file(tmp_path):
    f = write_predictions_file(tmp_path / "p" / "predictions_20220819_step060.nc",
                               n_hres=32, n_lres=16, states=("10u",), members=(1,))
    paths = build_pair_from_predictions(f, tmp_path / "mats")
    import scipy.sparse as sps
    up, down = sps.load_npz(paths["up_matrix"]), sps.load_npz(paths["down_matrix"])
    assert up.shape == (4 * 32 * 41, 4 * 16 * 25) and down.shape == up.shape[::-1]
    assert unit_vectors([0.0], [0.0]).shape == (1, 3)
