"""The denoiser trace must be a no-op unless it is explicitly switched on."""
from __future__ import annotations

import json

import pytest

from eval.predict.model_loader import _activate_denoiser_trace

torch = pytest.importorskip("torch")


class _Inner:
    def __init__(self):
        self.calls = 0
        self.data_indices = None

    def fwd_with_preconditioning(self, *args, **kwargs):
        self.calls += 1
        return {"out_hres": args[1]["out_hres"]}


class _Model:
    def __init__(self):
        self.model = _Inner()


def test_trace_is_off_without_the_environment_variable(monkeypatch):
    monkeypatch.delenv("DS_DENOISER_TRACE", raising=False)
    m = _Model()
    assert _activate_denoiser_trace(m, config=None) is None
    # Still the class's own method: nothing was wrapped.
    assert m.model.fwd_with_preconditioning.__func__ is _Inner.fwd_with_preconditioning


def test_trace_records_the_extreme_per_channel(monkeypatch, tmp_path):
    out = tmp_path / "trace.json"
    monkeypatch.setenv("DS_DENOISER_TRACE", str(out))
    m = _Model()
    assert _activate_denoiser_trace(m, config=None) == {"denoiser_trace": str(out)}

    y = {"out_hres": torch.tensor([[[[[1.0, 7.0], [3.0, 2.0]]]]])}
    sigma = {"out_hres": torch.tensor([[[[[0.5]]]]])}
    m.model.fwd_with_preconditioning({"in": 0}, y, sigma)
    assert m.model.calls == 1, "the traced call must pass through to the real denoiser"

    m._denoiser_trace["dump"]()
    rec = json.loads(out.read_text())["records"]
    assert len(rec) == 1
    assert rec[0]["sigma"] == pytest.approx(0.5)
    # Channel maxima are taken over every point, per channel: 3.0 and 7.0.
    assert sorted(rec[0]["channels"].values()) == pytest.approx([3.0, 7.0])
