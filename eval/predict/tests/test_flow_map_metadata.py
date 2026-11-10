"""A flow-map run's files record its sampler: sampling_config_json (canonical block) and
sampler_summary ("jumps 2 from sigma 80"), which eval.cli judge prints in the read header."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import install_torch_stub, load_predict_submodule

FLOW_MAP_JSON = json.dumps({"schedule_type": "flow_map", "num_steps": 2, "sigma_max": 80.0,
                            "sigma_min": 0.0, "sampler": "jumps"}, sort_keys=True)


def _metadata(monkeypatch, extra_args_json):
    install_torch_stub(monkeypatch)
    writer = load_predict_submodule(monkeypatch, "output_writer")
    types_mod = sys.modules["eval.predict.types"]
    ensemble = types_mod.EnsemblePrediction(
        init_date=np.datetime64("2023-02-26T00:00:00", "ns"), lead_step_hours=24, member_ids=[1, 2],
        source_bundle_paths=["a.nc", "b.nc"], members_missing_target=[], weather_states=["2t"],
        draw_seeds=[11, 12], seed_base=1234,
    )
    config = types_mod.PredictionConfig(checkpoint_path=Path("/ck/run/last.ckpt"), input_dir=Path("/in"),
                                        output_dir=Path("/out"), extra_args_json=extra_args_json)
    return writer._metadata_from_ensemble(ensemble, config).to_attrs()


def test_flow_map_attrs(monkeypatch):
    attrs = _metadata(monkeypatch, FLOW_MAP_JSON)
    assert json.loads(attrs["sampling_config_json"]) == json.loads(FLOW_MAP_JSON)
    assert attrs["sampler_summary"] == "jumps 2 from sigma 80"


def test_edm_attrs_get_a_summary_too(monkeypatch):
    attrs = _metadata(monkeypatch, '{"schedule_type": "karras", "num_steps": 30, "sigma_max": 80.0, '
                                   '"sigma_min": 0.03, "rho": 7.0, "sampler": "heun", "S_churn": 0.0}')
    assert attrs["sampler_summary"] == "heun karras 30 steps from sigma 80 to 0.03, rho 7, churn 0"


def test_main_normalizes_the_flow_map_block():
    from manual_inference.prediction.sampler_spec import normalize_sampler_json

    raw = '{"schedule_type": "flow_map", "num_steps": 1, "sigma_max": 80}'
    assert json.loads(normalize_sampler_json(raw)) == {"schedule_type": "flow_map", "num_steps": 1,
                                                       "sigma_max": 80.0, "sigma_min": 0.0, "sampler": "jumps"}
