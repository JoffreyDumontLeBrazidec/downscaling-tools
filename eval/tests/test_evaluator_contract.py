"""Every registered, non-retired evaluator satisfies the contract of eval/evaluators/base.py."""
from __future__ import annotations

import importlib
import types

import pytest

from eval.evaluators import base, registry


@pytest.mark.parametrize("name", registry.runnable_names())
def test_evaluator_package_satisfies_the_contract(name):
    mod = importlib.import_module(f"eval.evaluators.{name}")
    assert base.check_contract(mod, name) == []


@pytest.mark.parametrize("name", registry.runnable_names())
def test_run_score_plot_are_exported(name):
    mod = importlib.import_module(f"eval.evaluators.{name}")
    assert {"run", "score", "plot", "EVALUATOR_SPEC"} <= set(mod.__all__)


def test_no_score_and_no_plot_do_nothing():
    assert base.no_score("results", {}, {}, predictions_dir="p") == []
    assert base.no_plot("results", {}, {}, output_dir="o") is None


def test_only_scored_evaluators_may_return_rows_by_default(tmp_path):
    """An evaluator that is not on the scoreboard and has no scorer of its own returns no rows."""
    for name in registry.names(registry.STANDARD, registry.DIAGNOSTIC):
        mod = importlib.import_module(f"eval.evaluators.{name}")
        if mod.score is base.no_score:
            assert mod.score(tmp_path, {}, {}) == []


def _module(**attrs):
    mod = types.ModuleType("fake")
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _good_spec(name="fake"):
    return {"name": name, "requires": ["predictions"], "outputs": ["x.json: a result."]}


def test_check_contract_accepts_a_minimal_module():
    mod = _module(
        EVALUATOR_SPEC=_good_spec(),
        run=lambda predictions_dir, lane_config, eval_config, **kw: None,
        score=lambda results_dir, lane_config, eval_config, **kw: [],
        plot=lambda results_dir, lane_config, eval_config, **kw: None,
    )
    assert base.check_contract(mod, "fake") == []


def test_check_contract_reports_each_kind_of_problem():
    mod = _module(
        EVALUATOR_SPEC={"name": "other", "requires": ["gpu"]},
        run=lambda predictions_dir, lane_config, eval_config: None,   # rejects output_dir=...
        score=None,
    )
    problems = "\n".join(base.check_contract(mod, "fake"))
    assert "name" in problems and "requires" in problems and "outputs" in problems
    assert "run cannot be called" in problems
    assert "score is missing" in problems
    assert "plot is missing" in problems
