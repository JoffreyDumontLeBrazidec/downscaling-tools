"""The contract every evaluator package fulfils.

An evaluator is a package ``eval/evaluators/<name>/`` whose ``__init__.py`` exposes
exactly four things. ``eval.cli evaluate`` (``eval/cli/evaluate.py``) calls them in
this order, and the scoreboard aggregator (``eval/scoreboard/aggregator.py``) calls
``score`` on its own.

``EVALUATOR_SPEC``
    A dict describing the code (see ``EvaluatorSpec``): ``name``, ``requires``,
    and optionally ``outputs`` and ``deliverables``. What role the evaluator plays
    (scored, standard, diagnostic, retired) is not in the spec; it lives once, in
    ``eval/evaluators/registry.py``, together with the one-sentence question.

``run(predictions_dir, lane_config, eval_config, *, output_dir, overwrite, checkpoint, run_label, **kwargs)``
    Does the computation and writes its raw results below ``output_dir`` (the
    evaluator's own results directory, ``<run>/evaluators/<name>/``). Called
    first, and skipped by ``evaluate --plot-only``. Returns the results directory
    (the return value is not used).

``score(results_dir, lane_config, eval_config, *, predictions_dir=None, **kwargs)``
    Reads the results that ``run`` wrote and returns the scoreboard rows, a list of
    ``{"metric": str, "value": float, "unit": str}``. An evaluator that produces no
    scoreboard rows returns an empty list (``no_score`` below does exactly that).
    The rows only reach ``scoreboard/scores.csv`` when the registry says the
    evaluator feeds the scoreboard.

``plot(results_dir, lane_config, eval_config, *, output_dir=None, **kwargs)``
    Renders figures from the results into ``results_dir``. Runs last and is the
    only part ``evaluate --plot-only`` repeats. An evaluator without figures
    uses ``no_plot`` below.

Arguments, all passed by ``eval.cli``:

``predictions_dir``
    Directory of ``predictions_<date>_step<NNN>.nc`` files written by ``predict``.
    ``None`` is possible only for evaluators that require a checkpoint instead.
``lane_config``
    The fully resolved lane configuration (``eval.config.loader.load_lane``), a
    dict. Evaluators read shared sections from it, for example ``regions``.
``eval_config``
    ``lane_config[<name>]``, the evaluator's own block, with ``stages`` added when
    ``--stages`` was given. An absent block is an empty dict.
``output_dir`` / ``results_dir``
    The evaluator's own results directory. ``run`` creates it, ``score`` and
    ``plot`` read it.
``overwrite``
    True when ``--overwrite`` was given.
``checkpoint``
    The model checkpoint path or ``None``. Evaluators whose spec ``requires``
    ``"checkpoint"`` are skipped when it is missing.
``run_label``
    A short display label for legends.

``**kwargs`` on every function absorbs keywords that ``eval.cli`` may add in
future, so an evaluator never breaks because the caller learned a new keyword.
``check_contract`` verifies all of this for one package; the test
``eval/tests/test_evaluator_contract.py`` runs it on every registered evaluator.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, TypedDict

# What an evaluator needs before it can run.
#   "predictions"  the prediction NetCDF files that `predict` writes
#   "checkpoint"   the model checkpoint itself (the evaluator runs the model)
Requirement = Literal["predictions", "checkpoint"]
REQUIREMENTS: tuple[str, ...] = ("predictions", "checkpoint")


class ScoreRow(TypedDict):
    """One scoreboard row returned by ``score``."""

    metric: str
    value: float
    unit: str


class EvaluatorSpec(TypedDict, total=False):
    """The ``EVALUATOR_SPEC`` dict of an evaluator package."""

    name: str                      # the package name, identical to the registry name
    requires: list[Requirement]    # what must be available (see Requirement)
    outputs: list[str]             # main files ``run`` writes below the results directory
    deliverables: dict             # files promoted to the run root by eval.lean_layout


class RunFn(Protocol):
    def __call__(
        self,
        predictions_dir: str | Path | None,
        lane_config: dict,
        eval_config: dict,
        *,
        output_dir: str | Path | None = None,
        overwrite: bool = False,
        checkpoint: str | None = None,
        run_label: str = "",
        **kwargs: Any,
    ) -> Any: ...


class ScoreFn(Protocol):
    def __call__(
        self,
        results_dir: str | Path,
        lane_config: dict,
        eval_config: dict,
        *,
        predictions_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> list[ScoreRow]: ...


class PlotFn(Protocol):
    def __call__(
        self,
        results_dir: str | Path,
        lane_config: dict,
        eval_config: dict,
        *,
        output_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None: ...


class EvaluatorModule(Protocol):
    """What ``import eval.evaluators.<name>`` must give."""

    EVALUATOR_SPEC: EvaluatorSpec
    run: RunFn
    score: ScoreFn
    plot: PlotFn


# ---------------------------------------------------------------------------
# Adapters for evaluators that have nothing to score or nothing to plot
# ---------------------------------------------------------------------------

def no_score(results_dir, lane_config, eval_config, **kwargs) -> list[ScoreRow]:
    """``score`` for an evaluator that produces no scoreboard rows."""
    return []


def no_plot(results_dir, lane_config, eval_config, *, output_dir=None, **kwargs) -> None:
    """``plot`` for an evaluator whose ``run`` already writes its figures, or has none."""
    return None


# ---------------------------------------------------------------------------
# Conformance check
# ---------------------------------------------------------------------------

# The exact calls eval.cli makes (eval/cli/evaluate.py), as (positional, keywords).
_CALLS = {
    "run": (
        ("predictions_dir", "lane_config", "eval_config"),
        {"output_dir": None, "overwrite": False, "checkpoint": None, "run_label": ""},
    ),
    "score": (("results_dir", "lane_config", "eval_config"), {"predictions_dir": None}),
    "plot": (("results_dir", "lane_config", "eval_config"), {"output_dir": None}),
}


def check_contract(module: ModuleType, name: str) -> list[str]:
    """Return what is wrong with an evaluator package, or an empty list when it conforms.

    Checks the spec (name, requires), that ``run``, ``score`` and ``plot`` exist
    and are callable, and that each can be called the way ``eval.cli`` calls it.
    Nothing is executed: the signatures are only bound.
    """
    problems: list[str] = []
    spec = getattr(module, "EVALUATOR_SPEC", None)
    if not isinstance(spec, dict):
        problems.append("EVALUATOR_SPEC is missing or not a dict")
    else:
        if spec.get("name") != name:
            problems.append(f"EVALUATOR_SPEC['name'] is {spec.get('name')!r}, expected {name!r}")
        requires = spec.get("requires")
        if not isinstance(requires, list) or not requires or any(r not in REQUIREMENTS for r in requires):
            problems.append(f"EVALUATOR_SPEC['requires'] must be a non-empty list drawn from {list(REQUIREMENTS)}")
        outputs = spec.get("outputs")
        if not isinstance(outputs, list) or not outputs or not all(isinstance(o, str) for o in outputs):
            problems.append("EVALUATOR_SPEC['outputs'] must be a non-empty list of strings")
    for fn_name, (positional, keywords) in _CALLS.items():
        fn = getattr(module, fn_name, None)
        if not callable(fn):
            problems.append(f"{fn_name} is missing or not callable")
            continue
        try:
            inspect.signature(fn).bind(*positional, **keywords)
        except TypeError as exc:
            problems.append(f"{fn_name} cannot be called as eval.cli calls it: {exc}")
    return problems
