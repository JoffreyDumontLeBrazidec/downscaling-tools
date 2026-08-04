from __future__ import annotations

import json

import numpy as np

from eval.predict.local_scope import apply_local_output_scope, hres_mask_for_scope
from eval.predict.types import EnsemblePrediction


def _ensemble() -> EnsemblePrediction:
    return EnsemblePrediction(
        init_date=np.datetime64("2023-08-29T00:00:00", "ns"),
        lead_step_hours=24,
        member_ids=[1],
        source_bundle_paths=["bundle.nc"],
        members_missing_target=[],
        weather_states=["msl", "10u"],
        lon_lres=np.array([-90.0, -80.0]),
        lat_lres=np.array([20.0, 30.0]),
        lon_hres=np.array([-86.0, -84.0, -10.0, 170.0]),
        lat_hres=np.array([27.0, 29.0, 0.0, 10.0]),
        x_stack=np.ones((1, 1, 2, 2), dtype=np.float32),
        y_stack=np.arange(8, dtype=np.float32).reshape(1, 1, 4, 2),
        y_pred_stack=(np.arange(8, dtype=np.float32) + 10).reshape(1, 1, 4, 2),
        x_interp_stack=(np.arange(8, dtype=np.float32) + 20).reshape(1, 1, 4, 2),
    )


def test_bbox_scope_masks_hres_points():
    scope = {"mode": "bbox", "lat_min": 25.0, "lat_max": 30.0, "lon_min": -90.0, "lon_max": -80.0}
    mask = hres_mask_for_scope(np.array([-86.0, -84.0, -10.0]), np.array([27.0, 29.0, 0.0]), scope)
    assert mask.tolist() == [True, True, False]


def test_apply_local_output_scope_crops_hres_arrays_only():
    scope = json.dumps({"mode": "bbox", "lat_min": 25.0, "lat_max": 30.0, "lon_min": -90.0, "lon_max": -80.0})
    cropped = apply_local_output_scope(_ensemble(), scope)

    assert cropped.lon_hres.tolist() == [-86.0, -84.0]
    assert cropped.y_pred_stack.shape == (1, 1, 2, 2)
    assert cropped.y_stack.shape == (1, 1, 2, 2)
    assert cropped.x_interp_stack.shape == (1, 1, 2, 2)
    assert cropped.x_stack.shape == (1, 1, 2, 2)
