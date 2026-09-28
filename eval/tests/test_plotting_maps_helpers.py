"""Tests for the pure helpers in eval.plotting.maps_helpers (no rendering)."""
from __future__ import annotations

import numpy as np
import pytest

from eval.plotting.maps_helpers import (
    is_difference_key,
    octahedral_grid_name,
    regrid_nearest,
    region_panel_title,
)


@pytest.mark.parametrize("n, name", [
    (10_944, "O48"), (40_320, "O96"), (421_120, "O320"), (6_599_680, "O1280"), (26_306_560, "O2560"),
])
def test_octahedral_grid_name_from_point_count(n, name):
    assert octahedral_grid_name(n) == name


def test_octahedral_grid_name_rejects_other_sizes():
    assert octahedral_grid_name(1000) is None
    assert octahedral_grid_name(0) is None


def test_regrid_nearest_keeps_values_and_leaves_gaps_empty():
    # Dense points in the western half only: the eastern half must stay empty (NaN),
    # the western half must carry only source values (nearest neighbour, no averaging).
    lon, lat = np.meshgrid(np.arange(0.0, 5.0, 0.25), np.arange(0.0, 10.0, 0.25))
    val = np.where(lon.ravel() < 2.5, 1.0, 2.0)
    gx, gy, grid = regrid_nearest(lon.ravel(), lat.ravel(), val, (0.0, 10.0, 0.0, 10.0))
    assert grid.shape == (gy.size, gx.size)
    west = grid[:, gx < 4.5]
    east = grid[:, gx > 6.0]
    assert np.isfinite(west).all()
    assert set(np.unique(west)) <= {1.0, 2.0}
    assert np.isnan(east).all()


def test_regrid_nearest_handles_longitudes_across_the_antimeridian():
    lon = np.array([179.8, -179.8, 179.9, -179.9])
    lat = np.array([0.0, 0.0, 0.1, 0.1])
    gx, gy, grid = regrid_nearest(lon, lat, np.ones(4), (179.5, 180.5, -0.5, 0.5), res=0.1)
    assert np.isfinite(grid).any()


def test_region_panel_titles_are_readable():
    kw = dict(input_grid="O320", target_grid="O1280")
    assert region_panel_title("x_0", **kw) == "Input (O320)"
    assert region_panel_title("x_interp_0", **kw) == "Input interpolated to O1280"
    assert region_panel_title("y_0", **kw) == "Truth (O1280)"
    assert region_panel_title("y_pred_0", **kw) == "Model (O1280)"
    assert region_panel_title("y", truth="ENFO O1280", **kw) == "Truth (ENFO O1280)"
    # The derived residual panels are interpolated input minus truth / minus model.
    assert region_panel_title("residuals_0", **kw) == "Interpolated input minus truth"
    assert region_panel_title("residuals_pred_0", **kw) == "Interpolated input minus model"
    assert region_panel_title("x_interp_minus_y_pred") == "Interpolated input minus model"
    assert region_panel_title("inter_step_12 (sigma=0.500)") == "Intermediate state, step 12 (σ = 0.5)"
    for key in ("x_0", "x_interp_0", "y_0", "y_pred_0", "residuals_0", "residuals_pred_0"):
        assert "_" not in region_panel_title(key, **kw)


def test_difference_keys():
    assert is_difference_key("residuals_0")
    assert is_difference_key("residuals_pred")
    assert is_difference_key("x_interp_minus_y")
    assert not is_difference_key("y_pred_0")
    assert not is_difference_key("x_interp_0")
