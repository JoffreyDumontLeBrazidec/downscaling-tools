"""Tests for PrepML orchestration."""
from __future__ import annotations

import pytest


def _lane_config_with_prepml() -> dict:
    return {
        "predict": {
            "dates": ["20230826", "20230827"],
            "members": [1, 2],
            "steps": [24, 48],
            "sampler": {"num_steps": 30},
        },
        "prepml": {
            "debug_expvers": ["dbg_test_1", "dbg_test_2"],
            "runner": "anemoi-dev",
            "venv": "/path/to/venv",
            "input": {"class": "od", "stream": "eefo", "type": "pf", "grid": "O96"},
            "output": {"class": "rd", "stream": "enfo", "type": "pf"},
            "output_template": "/data/template.grib",
            "forcings_npz": "/data/forcings.npz",
            "constant_high_res_forcings": ["z"],
            "high_res_input": ["z"],
            "truth": {"root": "/perm/reference/o96_o320"},
            "time_step": "24h",
            "lead_time": "240h",
            "platform": {"gpu": {"time": "0-12:18"}},
        },
        "evaluator_groups": {"default": []},
    }


def test_resolve_expver_explicit():
    from eval.predict.prepml import resolve_expver
    result = resolve_expver("j2pw", _lane_config_with_prepml())
    assert result == "j2pw"


def test_resolve_expver_debug_default():
    from eval.predict.prepml import resolve_expver
    result = resolve_expver(None, _lane_config_with_prepml())
    assert result == "dbg_test_1"


def test_resolve_expver_no_debug_pool_raises():
    from eval.predict.prepml import resolve_expver
    config = _lane_config_with_prepml()
    config["prepml"]["debug_expvers"] = []
    with pytest.raises(ValueError, match="No debug expvers"):
        resolve_expver(None, config)


def test_resolve_expver_no_prepml_section_raises():
    from eval.predict.prepml import resolve_expver
    config = {"predict": {}, "evaluator_groups": {"default": []}}
    with pytest.raises(ValueError, match="No 'prepml' section"):
        resolve_expver(None, config)


# --- _resolve_prepml_weather_states priority chain ---


def _write_bundle(path, *, surface_params, pl_bases=(), pl_levels=()):
    import netCDF4 as nc

    ds = nc.Dataset(str(path), "w")
    try:
        ds.createDimension("point_hres", 4)
        for p in surface_params:
            v = ds.createVariable(f"target_hres_{p}", "f4", ("point_hres",))
            v[:] = 0
        if pl_bases and pl_levels:
            ds.createDimension("target_level", len(pl_levels))
            tl = ds.createVariable("target_level", "i4", ("target_level",))
            tl[:] = pl_levels
            for base in pl_bases:
                v = ds.createVariable(
                    f"target_hres_{base}", "f4", ("target_level", "point_hres"),
                )
                v[:] = 0
    finally:
        ds.close()


def _bundle_dir_with_one_bundle(tmp_path):
    bundle = tmp_path / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    _write_bundle(
        bundle,
        surface_params=("10u", "10v", "2t", "msl", "2d", "skt", "sp", "tcw"),
        pl_bases=("t", "z", "q"),
        pl_levels=(1000, 850, 500),
    )
    return tmp_path


def test_resolve_weather_states_explicit_predict_field_wins(tmp_path):
    """Explicit predict.weather_states overrides all other sources."""
    from eval.predict.prepml import _resolve_prepml_weather_states

    lane_config = _lane_config_with_prepml()
    bundle_dir = _bundle_dir_with_one_bundle(tmp_path)
    lane_config["predict"]["input_root"] = str(bundle_dir)
    lane_config["predict"]["weather_states"] = ["10u", "2t", "z_500"]

    states = _resolve_prepml_weather_states(checkpoint="nonexistent.ckpt", lane_config=lane_config)
    assert states == ["10u", "2t", "z_500"]


def test_resolve_weather_states_bundle_discovery_used_when_no_explicit(tmp_path):
    """Bundle discovery gives the canonical surface-plus-core-pl coverage."""
    from eval.predict.prepml import _resolve_prepml_weather_states

    lane_config = _lane_config_with_prepml()
    bundle_dir = _bundle_dir_with_one_bundle(tmp_path)
    lane_config["predict"]["input_root"] = str(bundle_dir)
    # No predict.weather_states; spectra.fields would only emit 6, ensuring bundle wins.
    lane_config["spectra"] = {"fields": ["10u", "10v", "2t", "msl", "t_850", "z_500"]}

    states = _resolve_prepml_weather_states(checkpoint="nonexistent.ckpt", lane_config=lane_config)
    assert set(states) == {
        "10u", "10v", "2t", "msl", "2d", "skt", "sp", "tcw",
        "t_850", "z_500",
    }


def test_resolve_weather_states_drops_invalid(tmp_path):
    """Invalid weather_states (non-MARS-mappable) are dropped with a warning."""
    from eval.predict.prepml import _resolve_prepml_weather_states

    lane_config = _lane_config_with_prepml()
    lane_config["predict"]["input_root"] = str(tmp_path)
    lane_config["predict"]["weather_states"] = ["10u", "fancy_param_no_level", "2t"]

    states = _resolve_prepml_weather_states(checkpoint="x", lane_config=lane_config)
    assert states == ["10u", "2t"]


def test_resolve_weather_states_falls_back_to_spectra_fields(tmp_path):
    """When bundle and checkpoint are unavailable, spectra.fields is the last resort."""
    from eval.predict.prepml import _resolve_prepml_weather_states

    lane_config = _lane_config_with_prepml()
    # input_root points to an empty dir → no bundles
    lane_config["predict"]["input_root"] = str(tmp_path)
    lane_config["spectra"] = {"fields": ["10u", "10v", "2t", "msl", "t_850", "z_500"]}

    states = _resolve_prepml_weather_states(checkpoint="nonexistent.ckpt", lane_config=lane_config)
    assert states == ["10u", "10v", "2t", "msl", "t_850", "z_500"]


def test_resolve_weather_states_no_sources_raises(tmp_path):
    """If every source is empty, raise rather than silently choose nothing."""
    from eval.predict.prepml import _resolve_prepml_weather_states

    lane_config = _lane_config_with_prepml()
    lane_config["predict"]["input_root"] = str(tmp_path)
    # No spectra section, no explicit override, no usable bundle, no checkpoint.
    with pytest.raises(ValueError, match="Cannot determine output weather states"):
        _resolve_prepml_weather_states(checkpoint="nonexistent.ckpt", lane_config=lane_config)


def test_resolve_weather_states_bundle_intersected_with_model_outputs(tmp_path, monkeypatch):
    """Bundle discovery is intersected with what the model actually emits.

    Reproduces the o48_o96 bug discovered 2026-05-14: the bundle has 10 surface
    target_hres_* vars (incl `sp` because truth/analysis has it), but the model
    checkpoint's data_indices.model.output.name_to_index does not contain `sp` — so
    asking MARS for sp via PrepML yields "Expected 80, got 70" because the prepml
    suite never archives a var the model didn't produce. The intersection drops sp.
    """
    from eval.predict import prepml as prepml_mod

    lane_config = _lane_config_with_prepml()
    bundle_dir = _bundle_dir_with_one_bundle(tmp_path)
    lane_config["predict"]["input_root"] = str(bundle_dir)

    # Model emits 9 of the 10 bundle surface vars (no `sp`), plus the two core PL.
    model_outputs = [
        "10u", "10v", "2d", "2t", "msl", "skt", "tcw",
        "t_850", "z_500",
        # plus other PL the model trains on but we don't ask for
        "q_850", "u_500", "v_500",
    ]
    monkeypatch.setattr(
        prepml_mod, "_model_output_states_from_checkpoint",
        lambda ckpt: model_outputs,
    )

    states = prepml_mod._resolve_prepml_weather_states(
        checkpoint="any.ckpt", lane_config=lane_config,
    )
    # Bundle had 10 surface; model has 9 of them (no sp); core PL stays.
    assert "sp" not in states, f"sp leaked through: {states}"
    assert set(states) == {
        "10u", "10v", "2d", "2t", "msl", "skt", "tcw",
        "t_850", "z_500",
    }


def test_resolve_weather_states_no_intersection_when_model_outputs_unknown(tmp_path, monkeypatch):
    """If the checkpoint can't be read, bundle-discovery is used as-is (no regression)."""
    from eval.predict import prepml as prepml_mod

    lane_config = _lane_config_with_prepml()
    bundle_dir = _bundle_dir_with_one_bundle(tmp_path)
    lane_config["predict"]["input_root"] = str(bundle_dir)

    # Model output reader returns empty list (e.g. inference-* checkpoint with no companion base)
    monkeypatch.setattr(
        prepml_mod, "_model_output_states_from_checkpoint",
        lambda ckpt: [],
    )

    states = prepml_mod._resolve_prepml_weather_states(
        checkpoint="any.ckpt", lane_config=lane_config,
    )
    # All 10 surface + 2 core PL — same behavior as before the intersection patch.
    assert set(states) == {
        "10u", "10v", "2t", "msl", "2d", "skt", "sp", "tcw",
        "t_850", "z_500",
    }
