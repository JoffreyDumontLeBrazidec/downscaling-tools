"""Evaluator packages: one package per evaluator, all following the same contract.

* ``registry.py`` is the one list of evaluators (role, question, host constraint,
  retirement).
* ``base.py`` is the contract each package fulfils: ``run``, ``score``, ``plot`` and
  ``EVALUATOR_SPEC``, with the arguments ``eval.cli`` passes.
* ``describe.py`` gathers what ``python -m eval.cli list`` and ``describe <name>``
  print, from the registry, the package docstring and the spec.

The computation of most evaluators lives in ``eval/_backends/<name>/``; the package
here is the thin, uniform front of it.
"""
