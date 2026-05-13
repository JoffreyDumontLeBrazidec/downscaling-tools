"""Tests pinning PrepML predictions_*.nc to the canonical manual schema.

Backend equality goal: `eval.cli predict --mode prepml` must write the same
NC schema as `--mode manual` so every downstream evaluator can stay
backend-agnostic. The canonical contract is `validate_predictions_dataset`
in `eval/predict/dataset_builder.py`; manual NCs pass it with zero errors,
PrepML NCs currently fail (missing sample dim, missing date/init_date/
lead_step_hours/valid_time vars, missing init_date/lead_step_hours/
checkpoint_id attrs).

These tests cover the in-memory canonicalization step. They build a
fake mars-retrieve result + a fake bundle-truth result with the
*pre-fix* PrepML shapes, run the new canonicalizer, and assert the
output passes the canonical validator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


_N_HRES = 64
_N_LRES = 16
_N_ENS = 3
_WEATHER_STATES = ["10u", "10v", "2t", "msl", "t_850", "z_500"]


def _fake_mars_y_pred() -> xr.Dataset:
    """Build a dataset shaped like `_reshape_to_prediction_format` output.

    y_pred dims: (weather_state, ensemble_member, forecast_reference_time, step, grid_point_hres)
    Includes lon_hres / lat_hres / step coords as the prepml path produces them.
    """
    rng = np.random.default_rng(0)
    n_ws = len(_WEATHER_STATES)
    y_pred = rng.standard_normal((n_ws, _N_ENS, 1, 1, _N_HRES)).astype("float32")
    ds = xr.Dataset(
        {
            "y_pred": xr.DataArray(
                y_pred,
                dims=("weather_state", "ensemble_member", "forecast_reference_time", "step", "grid_point_hres"),
                coords={
                    "weather_state": pd.Index(_WEATHER_STATES, name="weather_state"),
                    "ensemble_member": np.arange(_N_ENS),
                    "forecast_reference_time": np.array([np.datetime64("2025-09-26T00:00", "ns")]),
                    "step": np.array([24], dtype="int64"),
                    "grid_point_hres": np.arange(_N_HRES),
                },
                attrs={"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "lon_hres": xr.DataArray(
                rng.uniform(-180, 180, _N_HRES).astype("float32"),
                dims=("grid_point_hres",),
            ),
            "lat_hres": xr.DataArray(
                rng.uniform(-90, 90, _N_HRES).astype("float32"),
                dims=("grid_point_hres",),
            ),
        }
    )
    return ds


def _fake_bundle_dataset() -> xr.Dataset:
    """Build a dataset shaped like `_load_bundle_truth_and_input` output.

    y dims: (weather_state, grid_point_hres) — no sample/ens dims
    x dims: (weather_state, grid_point_lres)
    """
    rng = np.random.default_rng(1)
    n_ws = len(_WEATHER_STATES)
    y = rng.standard_normal((n_ws, _N_HRES)).astype("float32")
    x = rng.standard_normal((n_ws, _N_LRES)).astype("float32")
    ds = xr.Dataset(
        {
            "y": xr.DataArray(
                y,
                dims=("weather_state", "grid_point_hres"),
                coords={"weather_state": pd.Index(_WEATHER_STATES, name="weather_state")},
                attrs={"lon": "lon_hres", "lat": "lat_hres"},
            ),
            "x": xr.DataArray(
                x,
                dims=("weather_state", "grid_point_lres"),
                coords={"weather_state": pd.Index(_WEATHER_STATES, name="weather_state")},
                attrs={"lon": "lon_lres", "lat": "lat_lres"},
            ),
            "lat_hres": xr.DataArray(rng.uniform(-90, 90, _N_HRES).astype("float32"), dims=("grid_point_hres",)),
            "lon_hres": xr.DataArray(rng.uniform(-180, 180, _N_HRES).astype("float32"), dims=("grid_point_hres",)),
            "lat_lres": xr.DataArray(rng.uniform(-90, 90, _N_LRES).astype("float32"), dims=("grid_point_lres",)),
            "lon_lres": xr.DataArray(rng.uniform(-180, 180, _N_LRES).astype("float32"), dims=("grid_point_lres",)),
        }
    )
    return ds


def test_canonicalize_passes_manual_schema_validator():
    """The canonicalizer must produce a dataset that satisfies the manual schema."""
    from eval.predict.dataset_builder import validate_predictions_dataset
    from eval.predict.mars_retrieve import canonicalize_prepml_predictions

    ds_pred = _fake_mars_y_pred()
    ds_bundle = _fake_bundle_dataset()

    canonical = canonicalize_prepml_predictions(
        ds_pred=ds_pred,
        ds_bundle=ds_bundle,
        date="20250926",
        step=24,
        members=list(range(1, _N_ENS + 1)),
        weather_states=_WEATHER_STATES,
        checkpoint_id="fake_ckpt_for_test",
    )

    errors = validate_predictions_dataset(canonical)
    assert errors == [], f"canonical PrepML schema validation failed:\n  " + "\n  ".join(errors)


def test_canonicalize_y_pred_dim_order_matches_manual():
    from eval.predict.mars_retrieve import canonicalize_prepml_predictions

    ds_pred = _fake_mars_y_pred()
    ds_bundle = _fake_bundle_dataset()

    canonical = canonicalize_prepml_predictions(
        ds_pred=ds_pred,
        ds_bundle=ds_bundle,
        date="20250926",
        step=24,
        members=list(range(1, _N_ENS + 1)),
        weather_states=_WEATHER_STATES,
        checkpoint_id="fake_ckpt_for_test",
    )

    assert canonical["y_pred"].dims == ("sample", "ensemble_member", "grid_point_hres", "weather_state")
    assert canonical["y_pred"].shape == (1, _N_ENS, _N_HRES, len(_WEATHER_STATES))
    assert canonical["y"].dims == ("sample", "ensemble_member", "grid_point_hres", "weather_state")
    assert canonical["x"].dims == ("sample", "ensemble_member", "grid_point_lres", "weather_state")


def test_canonicalize_y_pred_values_preserved():
    """Reordering must permute values correctly — picked values must round-trip."""
    from eval.predict.mars_retrieve import canonicalize_prepml_predictions

    ds_pred = _fake_mars_y_pred()
    ds_bundle = _fake_bundle_dataset()

    canonical = canonicalize_prepml_predictions(
        ds_pred=ds_pred,
        ds_bundle=ds_bundle,
        date="20250926",
        step=24,
        members=list(range(1, _N_ENS + 1)),
        weather_states=_WEATHER_STATES,
        checkpoint_id="fake_ckpt_for_test",
    )

    for ws_idx, ws_name in enumerate(_WEATHER_STATES):
        for ens_idx in range(_N_ENS):
            raw = ds_pred["y_pred"].values[ws_idx, ens_idx, 0, 0, :]
            canon = canonical["y_pred"].values[0, ens_idx, :, ws_idx]
            np.testing.assert_array_equal(raw, canon, err_msg=f"value mismatch at ws={ws_name} ens={ens_idx}")


def test_canonicalize_truth_broadcast_across_members():
    """Truth y has no ensemble_member dim in prepml bundle; canonicalized y must broadcast it."""
    from eval.predict.mars_retrieve import canonicalize_prepml_predictions

    ds_pred = _fake_mars_y_pred()
    ds_bundle = _fake_bundle_dataset()

    canonical = canonicalize_prepml_predictions(
        ds_pred=ds_pred,
        ds_bundle=ds_bundle,
        date="20250926",
        step=24,
        members=list(range(1, _N_ENS + 1)),
        weather_states=_WEATHER_STATES,
        checkpoint_id="fake_ckpt_for_test",
    )

    y = canonical["y"].values
    # truth identical across members
    for ens_idx in range(1, _N_ENS):
        np.testing.assert_array_equal(y[0, 0], y[0, ens_idx])
    # broadcasted truth equals the source (ws, grid) array transposed to (grid, ws)
    expected = ds_bundle["y"].values.T  # (grid_hres, ws)
    np.testing.assert_array_equal(y[0, 0], expected)


def test_canonicalize_lead_time_and_valid_time():
    from eval.predict.mars_retrieve import canonicalize_prepml_predictions

    ds_pred = _fake_mars_y_pred()
    ds_bundle = _fake_bundle_dataset()

    canonical = canonicalize_prepml_predictions(
        ds_pred=ds_pred,
        ds_bundle=ds_bundle,
        date="20250926",
        step=24,
        members=list(range(1, _N_ENS + 1)),
        weather_states=_WEATHER_STATES,
        checkpoint_id="fake_ckpt_for_test",
    )

    init = np.asarray(canonical["init_date"].values).reshape(-1)[0]
    valid = np.asarray(canonical["valid_time"].values).reshape(-1)[0]
    lead = int(np.asarray(canonical["lead_step_hours"].values).reshape(-1)[0])
    assert init == np.datetime64("2025-09-26T00:00:00", "ns")
    assert lead == 24
    assert valid == init + np.timedelta64(24, "h")
    assert canonical.attrs["checkpoint_id"] == "fake_ckpt_for_test"


@pytest.mark.parametrize(
    "skip_var",
    ["lat_hres", "lon_hres", "lat_lres", "lon_lres"],
)
def test_canonicalize_propagates_coord_arrays(skip_var):
    """Coord arrays must be present even when bundle is sparse — fall back to ds_pred where possible."""
    from eval.predict.mars_retrieve import canonicalize_prepml_predictions

    ds_pred = _fake_mars_y_pred()
    ds_bundle = _fake_bundle_dataset().drop_vars(skip_var)

    canonical = canonicalize_prepml_predictions(
        ds_pred=ds_pred,
        ds_bundle=ds_bundle,
        date="20250926",
        step=24,
        members=list(range(1, _N_ENS + 1)),
        weather_states=_WEATHER_STATES,
        checkpoint_id="fake_ckpt_for_test",
    )

    if skip_var in ("lat_hres", "lon_hres"):
        # ds_pred carries lat_hres/lon_hres, so canonicalizer should fall back
        assert skip_var in canonical
    else:
        # lat_lres/lon_lres only live in the bundle; if bundle is missing,
        # canonicalizer cannot synthesize them — schema validation will flag.
        assert skip_var not in canonical
