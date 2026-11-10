"""Regression test: lane sampler keys must reach the noise scheduler (bug of 2026-09-22).

The o48_o96_fastlane lane asks for a 5 + 16 step split of experimental_piecewise; eval.cli
passed the split keys as loose kwargs through AnemoiModelInterface.predict_step(**kwargs),
the model's predict_step ignored them, and the scheduler ran its default 11 + 10 split.
"""
import inspect
import logging

import pytest

from manual_inference.prediction.predict import _predict_with_compatible_kwargs

FASTLANE_SAMPLER = {
    "schedule_type": "experimental_piecewise",
    "num_steps": 21,
    "sigma_max": 1000.0,
    "sigma_transition": 10.0,
    "sigma_min": 0.03,
    "high_schedule_type": "exponential",
    "low_schedule_type": "karras",
    "num_steps_high": 5,
    "num_steps_low": 16,
    "rho": 7.0,
    "sampler": "heun",
    "S_churn": 2.5,
    "S_min": 0.75,
    "S_max": 1000.0,
    "S_noise": 1.05,
}
LOOSE = {"num_steps", "sigma_max", "sigma_min", "rho", "schedule_type"}
LOOSE_SAMPLER = {"sampler", "S_churn", "S_min", "S_max", "S_noise"}


class _Model:
    """Mirrors the pristine AnemoiDiffusionDownscaler.predict_step merge rule."""

    def predict_step(self, batch, model_comm_group=None, noise_scheduler_params=None, sampler_params=None, **kwargs):
        ns = dict(noise_scheduler_params or {})
        sp = dict(sampler_params or {})
        for k, v in kwargs.items():
            if k in LOOSE:
                ns[k] = v
            elif k in LOOSE_SAMPLER:
                sp[k] = v
        return {"noise_scheduler": ns, "sampler": sp}


class _DeterministicModel:
    """Like the hres-lead local downscaler: no dict parameters at all."""

    def predict_step(self, batch, model_comm_group=None, **kwargs):
        return {"kwargs": kwargs}


class _Interface:
    """Mirrors AnemoiModelInterface.predict_step: only **kwargs, forwarded to the model."""

    def __init__(self, model):
        self.model = model

    def predict_step(self, batch, model_comm_group=None, gather_out=True, **kwargs):
        return self.model.predict_step(batch, model_comm_group=model_comm_group, **kwargs)


class _LegacyInterface:
    """ds-lineage interface: a single extra_args dict."""

    def predict_step(self, batch, model_comm_group=None, extra_args=None, **kwargs):
        return {"extra_args": extra_args}


def test_piecewise_keys_reach_scheduler_through_interface():
    out = _predict_with_compatible_kwargs(
        inference_model=_Interface(_Model()), batch={}, model_comm_group=None, extra_args=dict(FASTLANE_SAMPLER)
    )
    ns = out["noise_scheduler"]
    assert ns["num_steps_high"] == 5 and ns["num_steps_low"] == 16
    assert ns["sigma_transition"] == 10.0
    assert ns["high_schedule_type"] == "exponential" and ns["low_schedule_type"] == "karras"
    assert {k: out["sampler"][k] for k in LOOSE_SAMPLER} == {k: FASTLANE_SAMPLER[k] for k in LOOSE_SAMPLER}


def test_lane_without_split_keys_merges_the_same_as_before():
    sampler = {k: v for k, v in FASTLANE_SAMPLER.items() if k in LOOSE | LOOSE_SAMPLER}
    new = _predict_with_compatible_kwargs(
        inference_model=_Interface(_Model()), batch={}, model_comm_group=None, extra_args=dict(sampler)
    )
    old = _Model().predict_step({}, **sampler)  # the pre-fix route: every key loose
    assert new == old


def test_model_without_dict_params_keeps_loose_kwargs_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        out = _predict_with_compatible_kwargs(
            inference_model=_Interface(_DeterministicModel()),
            batch={},
            model_comm_group=None,
            extra_args=dict(FASTLANE_SAMPLER),
        )
    assert out["kwargs"]["num_steps_high"] == 5
    assert "noise_scheduler_params" not in out["kwargs"]
    assert "num_steps_high" in caplog.text


def test_extra_args_interface_unchanged():
    out = _predict_with_compatible_kwargs(
        inference_model=_LegacyInterface(), batch={}, model_comm_group=None, extra_args=dict(FASTLANE_SAMPLER)
    )
    assert out["extra_args"] == FASTLANE_SAMPLER


def test_real_signatures_match_the_mirrors():
    """The fix rests on these two facts about the installed anemoi-models."""
    interface = pytest.importorskip("anemoi.models.interface")
    ddm = pytest.importorskip("anemoi.models.models.diffusiondownscaler_encoder_processor_decoder")
    iface_params = inspect.signature(interface.AnemoiModelInterface.predict_step).parameters
    assert "noise_scheduler_params" not in iface_params
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in iface_params.values())
    model_cls = next(
        c for c in vars(ddm).values() if inspect.isclass(c) and "predict_step" in vars(c)
    )
    assert "noise_scheduler_params" in inspect.signature(model_cls.predict_step).parameters


def test_real_scheduler_builds_the_declared_split():
    samplers = pytest.importorskip("anemoi.models.samplers.diffusion_samplers")
    cfg = {k: v for k, v in FASTLANE_SAMPLER.items() if k not in LOOSE_SAMPLER}
    cls = samplers.NOISE_SCHEDULERS[cfg.pop("schedule_type")]
    sigmas = [float(s) for s in cls(**cfg).get_schedule()]
    assert len(sigmas) == 22 and sigmas[-1] == 0.0
    assert abs(sigmas[5] - 10.0) < 1e-9  # five exponential steps from 1000 reach the transition
    assert abs(sigmas[1] - 398.10717055349727) < 1e-9


# --------------------------------------------------------------------------- flow_map (jumps)
# A flow-map run (MeanFlow / iMF) samples by a few jumps from sigma 80. Its predict_step must
# receive exactly these two dicts (sampler_spec.py), on the keyword path and on the loose path.

from manual_inference.prediction.sampler_spec import (  # noqa: E402
    SamplerConfigError,
    describe_sampler,
    normalize_flow_map_sampler,
    normalize_sampler_json,
)

FLOW_MAP_LANE = {"schedule_type": "flow_map", "num_steps": 2, "sigma_max": 80.0, "sigma_min": 0.0, "sampler": "jumps"}
FLOW_MAP_NS = {"schedule_type": "flow_map", "num_steps": 2, "sigma_max": 80.0, "sigma_min": 0.0}
FLOW_MAP_SP = {"sampler": "jumps"}


class _RecordingFlowMapModel:
    """Stub flow-map checkpoint: predict_step records exactly the kwargs it receives."""

    def __init__(self):
        self.calls = []

    def predict_step(self, batch, model_comm_group=None, noise_scheduler_params=None, sampler_params=None, **kwargs):
        self.calls.append({"noise_scheduler_params": noise_scheduler_params,
                           "sampler_params": sampler_params, "kwargs": kwargs})
        return "y"


class _RecordingLooseModel:
    """Stub checkpoint without the dict parameters: every key arrives loose."""

    def __init__(self):
        self.calls = []

    def predict_step(self, batch, model_comm_group=None, **kwargs):
        self.calls.append(kwargs)
        return "y"


def _route(model, extra_args):
    return _predict_with_compatible_kwargs(inference_model=model, batch={}, model_comm_group=None,
                                           extra_args=dict(extra_args))


def test_flow_map_keyword_path_direct_and_through_interface():
    for wrap in (lambda m: m, _Interface):
        model = _RecordingFlowMapModel()
        assert _route(wrap(model), FLOW_MAP_LANE) == "y"
        (call,) = model.calls
        assert call["noise_scheduler_params"] == FLOW_MAP_NS
        assert call["sampler_params"] == FLOW_MAP_SP
        assert call["kwargs"] == {}


def test_flow_map_loose_path_passes_the_same_keys():
    model = _RecordingLooseModel()
    _route(_Interface(model), FLOW_MAP_LANE)
    (call,) = model.calls
    assert call == {**FLOW_MAP_NS, **FLOW_MAP_SP}


def test_flow_map_defaults_are_filled_and_types_canonical():
    model = _RecordingFlowMapModel()
    _route(model, {"schedule_type": "flow_map", "num_steps": 4, "sigma_max": 80})
    (call,) = model.calls
    assert call["noise_scheduler_params"] == {"schedule_type": "flow_map", "num_steps": 4,
                                              "sigma_max": 80.0, "sigma_min": 0.0}
    assert call["sampler_params"] == {"sampler": "jumps"}


def test_flow_map_extra_args_interface_gets_canonical_block():
    out = _route(_LegacyInterface(), FLOW_MAP_LANE)
    assert out["extra_args"] == FLOW_MAP_LANE


@pytest.mark.parametrize("bad", [
    {"S_churn": 0.0}, {"S_min": 0.75}, {"S_max": 80.0}, {"S_noise": 1.0},
])
def test_flow_map_rejects_churn_keys(bad):
    with pytest.raises(SamplerConfigError, match="churn"):
        normalize_flow_map_sampler({**FLOW_MAP_LANE, **bad})
    with pytest.raises(SamplerConfigError, match="churn"):
        _route(_RecordingFlowMapModel(), {**FLOW_MAP_LANE, **bad})


@pytest.mark.parametrize("bad,match", [
    ({"rho": 7.0}, "takes only"),
    ({"autoguide_checkpoint": "/x.ckpt", "autoguide_weight": 1.5}, "takes only"),
    ({"switch_checkpoint": "/x.ckpt"}, "takes only"),
    ({"sampler": "heun"}, "jumps"),
    ({"num_steps": 0}, "num_steps"),
    ({"num_steps": 2.0}, "num_steps"),
    ({"sigma_min": 90.0}, "sigma_min"),
])
def test_flow_map_rejects_other_keys_and_bad_values(bad, match):
    with pytest.raises(SamplerConfigError, match=match):
        normalize_flow_map_sampler({**FLOW_MAP_LANE, **bad})


def test_edm_blocks_are_untouched_by_the_flow_map_rules():
    assert normalize_flow_map_sampler(FASTLANE_SAMPLER) == FASTLANE_SAMPLER
    raw = '{"num_steps": 30, "S_churn": 0}'
    assert normalize_sampler_json(raw) == raw


def test_sampler_description_for_the_read_header():
    assert describe_sampler(FLOW_MAP_LANE) == "jumps 2 from sigma 80"
    edm30 = {"schedule_type": "karras", "num_steps": 30, "sigma_max": 80.0, "sigma_min": 0.03,
             "rho": 7.0, "sampler": "heun", "S_churn": 0.0}
    assert describe_sampler(edm30) == "heun karras 30 steps from sigma 80 to 0.03, rho 7, churn 0"
