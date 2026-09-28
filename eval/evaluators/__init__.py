"""Evaluator packages: one package per evaluator, all following the same contract.

* ``registry.py`` is the one list of evaluators (role, question, host constraint,
  retirement).
* ``base.py`` is the contract each package fulfils: ``run``, ``score``, ``plot`` and
  ``EVALUATOR_SPEC``, with the arguments ``eval.cli`` passes.
* ``describe.py`` gathers what ``python -m eval.cli list`` and ``describe <name>``
  print, from the registry, the package docstring and the spec.

The contract files of a package (``runner.py``, ``scorer.py``, ``plotter.py``) sit at its
top, and the computation they call sits in its ``core/`` subpackage, with the tests in
``tests/``. See ARCHITECTURE.md, section 2, for the rules.
"""
