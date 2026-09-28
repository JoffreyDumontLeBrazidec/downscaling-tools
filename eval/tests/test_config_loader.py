"""Tests for eval.config.loader."""

from __future__ import annotations

import pytest
import yaml

from eval.config.loader import (
    ConfigValidationError,
    load_event,
    load_host,
    load_lane,
)


def test_load_lane_valid():
    config = load_lane("o96_o320")
    assert "predict" in config
    assert "evaluator_groups" in config
    predict = config["predict"]
    assert isinstance(predict["members"], list)
    assert isinstance(predict["steps"], list)
    assert isinstance(predict["dates"], list)


def test_load_lane_with_overrides():
    config = load_lane("o96_o320", overrides={"predict": {"members": [1, 2]}})
    assert config["predict"]["members"] == [1, 2]
    # Original steps should still be present
    assert len(config["predict"]["steps"]) > 0


def test_load_lane_rejects_unknown_key(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "lanes").mkdir()
    bad = {
        "predict": {"members": [1], "steps": [24], "dates": ["20230826"]},
        "evaluator_groups": {"default": ["tc"]},
        "bogus_key": True,
    }
    (tmp_path / "lanes" / "bad.yaml").write_text(yaml.dump(bad))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    with pytest.raises(ConfigValidationError, match="unknown top-level key"):
        load_lane("bad")


def test_load_lane_rejects_missing_required(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "lanes").mkdir()
    bad = {"evaluator_groups": {"default": ["tc"]}}
    (tmp_path / "lanes" / "bad.yaml").write_text(yaml.dump(bad))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    with pytest.raises(ConfigValidationError, match="missing required top-level key 'predict'"):
        load_lane("bad")


def test_load_lane_rejects_bad_predict_members(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "lanes").mkdir()
    bad = {
        "predict": {"members": ["a", "b"], "steps": [24], "dates": ["20230826"]},
        "evaluator_groups": {"default": ["tc"]},
    }
    (tmp_path / "lanes" / "bad.yaml").write_text(yaml.dump(bad))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    with pytest.raises(ConfigValidationError, match="predict.members.*must be a list of int"):
        load_lane("bad")


def test_load_host_valid():
    config = load_host("atos_ac")
    assert "scheduler" in config
    assert "environment_setup" in config
    assert config["scheduler"]["qos"] == "nf"


def test_load_host_rejects_relative_code_root(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "hosts").mkdir()
    bad = {
        "code_root": "relative/path",
        "scratch_root": "/home/user/scratch",
        "scheduler": {"qos": "nf", "default_time": "04:00:00"},
        "environment_setup": {
            "module_loads": ["ecmwf-toolbox"],
            "venv_activate": "/path/to/venv/bin/activate",
        },
    }
    (tmp_path / "hosts" / "bad.yaml").write_text(yaml.dump(bad))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    with pytest.raises(ConfigValidationError, match="code_root.*must be an absolute path"):
        load_host("bad")


def test_load_event_valid():
    config = load_event("idalia")
    assert config["name"] == "idalia"
    assert "lat_min" in config
    assert "lat_max" in config
    assert "lon_min" in config
    assert "lon_max" in config
    assert isinstance(config["dates"], list)


def test_load_lane_allows_resource_profiles():
    config = load_lane("o96_o320")
    assert "resource_profiles" in config
    assert "predict" in config["resource_profiles"]


def test_lane_configs_do_not_advertise_quaver_group():
    for lane in ["o48_o96", "o96_o320", "o320_o1280", "o1280_o2560"]:
        config = load_lane(lane)
        assert "quaver" not in config

        evaluator_groups = config["evaluator_groups"]
        for evaluators in evaluator_groups.values():
            assert "quaver" not in evaluators


def test_tc_o320_o1280_fast_harness_contract():
    config = load_lane("tc_o320_o1280")

    predict = config["predict"]
    assert predict["sampler"]["sigma_max"] == 100_000.0
    assert predict["sampler"]["S_max"] == 100_000.0
    assert predict["num_gpus_per_model"] == 1
    assert predict["dates"] == [
        "20230826", "20230827", "20230828", "20230829", "20230830"
    ]
    assert predict["local_scope"] == {
        "mode": "bbox",
        "cut_graph": True,
        "hidden_halo_hops": 1,
        "label": "franklin_idalia_full250_box",
        "lat_min": 10.0,
        "lat_max": 40.0,
        "lon_min": -100.0,
        "lon_max": -58.0,
    }

    tc = config["tc"]
    assert tc["events"] == ["idalia", "franklin"]
    assert tc["support_mode"] == "native"
    assert tc["analysis_expid"] == "target O1280"
    assert tc["target_nc_label"] == "target O1280"
    assert tc.get("grib_dir") is None
    assert tc["reference_expids"] == []

    assert config["evaluator_groups"]["default"] == ["tc", "local_global"]
    assert "spectra" not in config["evaluator_groups"]["default"]


def test_o320_o1280_standard_sampler_uses_sigma100k():
    for lane in [
        "o320_o1280",
        "tc_o320_o1280",
        "tc_o320_o1280_regionalbundle",
        "o320_o1280_b785bf12_unified",
        "o320_o1280_b785bf12_unified_full",
    ]:
        sampler = load_lane(lane)["predict"]["sampler"]
        assert sampler["sigma_max"] == 100_000.0
        assert sampler["S_max"] == 100_000.0

    assert load_lane("o320_o1280_s1k")["predict"]["sampler"]["sigma_max"] == 1_000.0
    assert load_lane("o320_o1280_sigma10k")["predict"]["sampler"]["sigma_max"] == 10_000.0


def test_load_lane_allows_tctracker_config(tmp_path, monkeypatch):
    import eval.config.loader as loader

    (tmp_path / "lanes").mkdir()
    lane = {
        "predict": {"members": [1], "steps": [24], "dates": ["20230826"]},
        "evaluator_groups": {"default": ["tc"]},
        "tctracker": {"grid": 320, "vorticity": False},
    }
    (tmp_path / "lanes" / "with_tracker.yaml").write_text(yaml.dump(lane))
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)

    loaded = load_lane("with_tracker")
    assert loaded["tctracker"]["grid"] == 320


# ---------------------------------------------------------------------------
# predict.sampler_overrides
#
# Background for anyone reading these tests cold. `load_lane` merges a lane onto
# the lane named by its `base:` key using `_deep_merge`, which merges only two
# levels deep. The sampler lives at `predict.sampler`, three levels down, so a
# child that sets any key under `predict.sampler` replaces the base's whole
# sampler block rather than merging into it. `predict.sampler_overrides` is the
# merging alternative: its keys are applied one by one to the sampler resolved
# from the base chain, leaving every key the child does not name untouched.
#
# `predict.sampler` keeps its wholesale-replace behaviour unchanged, because all
# existing lane files depend on it.
# ---------------------------------------------------------------------------

_BASE_SAMPLER = {
    "schedule_type": "karras",
    "num_steps": 25,
    "sigma_max": 10000.0,
    "sigma_min": 0.03,
    "rho": 7.0,
    "S_churn": 2.5,
}


def _write_lane(config_dir, name, body):
    (config_dir / "lanes" / f"{name}.yaml").write_text(yaml.dump(body))


def _sampler_lane_dir(tmp_path, monkeypatch, children):
    """Build a temp config dir holding one base lane plus the given child lanes."""
    import eval.config.loader as loader

    (tmp_path / "lanes").mkdir()
    base = {
        "predict": {
            "members": [1, 2],
            "steps": [24, 48],
            "dates": ["20230826"],
            "sampler": dict(_BASE_SAMPLER),
        },
        "evaluator_groups": {"default": ["tc"]},
    }
    _write_lane(tmp_path, "sampler_base", base)
    for name, body in children.items():
        _write_lane(tmp_path, name, body)
    monkeypatch.setattr(loader, "_CONFIG_DIR", tmp_path)
    return tmp_path


def test_sampler_overrides_inherits_unnamed_keys(tmp_path, monkeypatch):
    """A child using sampler_overrides keeps every base key it does not name."""
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "child_overrides": {
                "base": "sampler_base",
                "predict": {"sampler_overrides": {"sigma_max": 100000.0}},
            }
        },
    )

    sampler = load_lane("child_overrides")["predict"]["sampler"]

    assert sampler["sigma_max"] == 100000.0
    for key, value in _BASE_SAMPLER.items():
        if key != "sigma_max":
            assert sampler[key] == value, f"{key} should have been inherited"


def test_sampler_overrides_key_is_consumed(tmp_path, monkeypatch):
    """The resolved config exposes only `sampler`; `sampler_overrides` is folded in."""
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "child_overrides": {
                "base": "sampler_base",
                "predict": {"sampler_overrides": {"num_steps": 40}},
            }
        },
    )

    predict = load_lane("child_overrides")["predict"]

    assert "sampler_overrides" not in predict
    assert predict["sampler"]["num_steps"] == 40


def test_sampler_still_replaces_wholesale(tmp_path, monkeypatch):
    """`predict.sampler` keeps its historical behaviour: it replaces, not merges."""
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "child_sampler": {
                "base": "sampler_base",
                "predict": {"sampler": {"sigma_max": 100000.0}},
            }
        },
    )

    sampler = load_lane("child_sampler")["predict"]["sampler"]

    assert sampler == {"sigma_max": 100000.0}
    assert "num_steps" not in sampler
    assert "schedule_type" not in sampler


def test_sampler_and_sampler_overrides_together(tmp_path, monkeypatch):
    """Documented order when one config sets both keys.

    `predict.sampler` is applied first and replaces the base block wholesale, then
    `predict.sampler_overrides` is merged on top of that result. So the resolved
    block contains exactly the keys named by `sampler`, with any key also named by
    `sampler_overrides` taking the override's value, and nothing is inherited from
    the base.
    """
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "child_both": {
                "base": "sampler_base",
                "predict": {
                    "sampler": {"schedule_type": "exponential", "num_steps": 30},
                    "sampler_overrides": {"num_steps": 40, "S_churn": 8.0},
                },
            }
        },
    )

    sampler = load_lane("child_both")["predict"]["sampler"]

    # sampler decided the block, so no base key survives
    assert "sigma_max" not in sampler
    assert "rho" not in sampler
    # sampler_overrides wins on the key both name, and adds the key only it names
    assert sampler == {"schedule_type": "exponential", "num_steps": 40, "S_churn": 8.0}


def test_lane_without_sampler_overrides_is_unchanged(tmp_path, monkeypatch):
    """A child mentioning neither key inherits the base sampler exactly as before."""
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "child_plain": {
                "base": "sampler_base",
                "predict": {"members": [3, 4]},
            }
        },
    )

    config = load_lane("child_plain")

    assert config["predict"]["sampler"] == _BASE_SAMPLER
    assert config["predict"]["members"] == [3, 4]


def test_sampler_overrides_through_multi_level_base_chain(tmp_path, monkeypatch):
    """Overrides compose down a chain, each level merging onto the resolved sampler.

    sampler_base -> mid (overrides num_steps) -> leaf (overrides S_churn). The leaf
    must see its own S_churn, the mid's num_steps, and the base's remaining keys.
    """
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "mid": {
                "base": "sampler_base",
                "predict": {"sampler_overrides": {"num_steps": 40}},
            },
            "leaf": {
                "base": "mid",
                "predict": {"sampler_overrides": {"S_churn": 8.0}},
            },
        },
    )

    sampler = load_lane("leaf")["predict"]["sampler"]

    assert sampler["S_churn"] == 8.0      # from the leaf
    assert sampler["num_steps"] == 40     # from the mid level
    assert sampler["schedule_type"] == "karras"   # from the base
    assert sampler["sigma_max"] == 10000.0        # from the base
    assert sampler["sigma_min"] == 0.03           # from the base
    assert sampler["rho"] == 7.0                  # from the base


def test_wholesale_sampler_replacement_warns_but_does_not_raise(tmp_path, monkeypatch, capsys):
    """The guard is advisory: it names the dropped keys and never stops the load."""
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "child_sampler": {
                "base": "sampler_base",
                "predict": {"sampler": {"sigma_max": 100000.0}},
            }
        },
    )

    config = load_lane("child_sampler")  # must not raise
    warning = capsys.readouterr().err

    assert config["predict"]["sampler"] == {"sigma_max": 100000.0}
    assert "replaces the base block wholesale" in warning
    assert "num_steps" in warning
    assert "sampler_overrides" in warning


def test_no_warning_when_sampler_overrides_restores_the_keys(tmp_path, monkeypatch, capsys):
    """Keys a child restores through sampler_overrides are not reported as dropped."""
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "child_plain": {
                "base": "sampler_base",
                "predict": {"sampler_overrides": {"sigma_max": 100000.0}},
            }
        },
    )

    load_lane("child_plain")

    assert "replaces the base block wholesale" not in capsys.readouterr().err


def test_bad_sampler_overrides_type_warns_and_keeps_inherited_sampler(
    tmp_path, monkeypatch, capsys
):
    """A malformed sampler_overrides is ignored with a warning, never an exception."""
    _sampler_lane_dir(
        tmp_path,
        monkeypatch,
        {
            "child_bad": {
                "base": "sampler_base",
                "predict": {"sampler_overrides": ["sigma_max", 100000.0]},
            }
        },
    )

    config = load_lane("child_bad")  # must not raise

    assert config["predict"]["sampler"] == _BASE_SAMPLER
    assert "sampler_overrides" not in config["predict"]
    assert "must be a mapping" in capsys.readouterr().err
