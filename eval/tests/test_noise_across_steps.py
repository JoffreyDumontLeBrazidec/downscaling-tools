"""noise_across_steps on a stub torch: shared re-seeds to the same field at every step, ou
chains the draws with the right correlation and counts the calls, independent does nothing."""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_policy():
    """Load the module by path: importing ``eval.predict`` pulls torch, which the policy itself
    only needs inside its contexts."""
    path = Path(__file__).resolve().parents[1] / "predict" / "noise_across_steps.py"
    spec = importlib.util.spec_from_file_location("noise_across_steps_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NoiseAcrossSteps


@pytest.fixture
def stub_torch(monkeypatch):
    """A minimal ``torch`` with manual_seed/randn on numpy, so the policy can be tested here."""
    state = {"rng": np.random.default_rng(0)}

    class _T(np.ndarray):
        @property
        def device(self):
            return "cpu"

        def detach(self):
            return self

        def clone(self):
            return self.copy()

    def randn(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return state["rng"].standard_normal(shape).view(_T)

    def manual_seed(seed):
        state["rng"] = np.random.default_rng(int(seed))

    cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda s: None)
    mod = types.ModuleType("torch")
    mod.randn, mod.manual_seed, mod.cuda = randn, manual_seed, cuda
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


def test_shared_gives_the_same_field_at_every_step(stub_torch):
    NoiseAcrossSteps = _load_policy()

    pol = NoiseAcrossSteps("shared", None, rank_seed=756)
    draws = {}
    for step in (12, 24, 36):
        for member in (1, 2):
            with pol.member_context("20230226", step, member):
                draws[(step, member)] = np.asarray(stub_torch.randn((5, 3)))
    assert np.array_equal(draws[(12, 1)], draws[(24, 1)])
    assert np.array_equal(draws[(12, 1)], draws[(36, 1)])
    assert not np.array_equal(draws[(12, 1)], draws[(12, 2)])


def test_ou_chain_correlation_and_counts(stub_torch):
    NoiseAcrossSteps = _load_policy()

    rho = 0.6
    pol = NoiseAcrossSteps("ou", rho, rank_seed=756)
    shared = NoiseAcrossSteps("shared", None, rank_seed=756)
    seq = []
    for step in (12, 24, 36, 48):
        with pol.member_context("20230226", step, 1):
            seq.append(np.asarray(stub_torch.randn((200, 50))))
    with shared.member_context("20230226", 12, 1):
        first_shared = np.asarray(stub_torch.randn((200, 50)))
    assert np.array_equal(seq[0], first_shared), "the first step of an OU run is the shared field"
    for a, b in zip(seq, seq[1:]):
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        assert abs(c - rho) < 0.03
        assert abs(b.std() - 1.0) < 0.03
    assert all(n == 1 for n in pol.draw_counts.values())
    assert pol.requires_sorted_steps()
    # the patch is removed after the context
    assert stub_torch.randn.__name__ == "randn"


def test_ou_rejects_steps_out_of_order_and_bad_rho(stub_torch):
    NoiseAcrossSteps = _load_policy()

    with pytest.raises(ValueError):
        NoiseAcrossSteps("ou", None, rank_seed=1)
    with pytest.raises(ValueError):
        NoiseAcrossSteps("ou", 1.5, rank_seed=1)
    pol = NoiseAcrossSteps("ou", 0.5, rank_seed=1)
    with pol.member_context("20230226", 24, 1):
        stub_torch.randn((2,))
    with pytest.raises(RuntimeError):
        with pol.member_context("20230226", 12, 1):
            pass


def test_independent_touches_nothing(stub_torch):
    NoiseAcrossSteps = _load_policy()

    pol = NoiseAcrossSteps("independent", None, rank_seed=1)
    stub_torch.manual_seed(3)
    a = np.asarray(stub_torch.randn((4,)))
    stub_torch.manual_seed(3)
    with pol.member_context("20230226", 12, 1):
        b = np.asarray(stub_torch.randn((4,)))
    assert np.array_equal(a, b)
    assert math.isclose(sum(pol.draw_counts.values()), 0)
