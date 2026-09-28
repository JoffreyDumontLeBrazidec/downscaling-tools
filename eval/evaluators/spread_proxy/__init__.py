"""Spread-proxy evaluator: model ensemble spread against ENFO ensemble spread.

Compares the spread of the model members (y_pred) with the spread of the ENFO
truth members (y) stored in the same prediction files, as a cheap proxy for how
well the model ensemble is dispersed. Diagnostic only; no scoreboard row. The
canonical probabilistic verdict is the `quaver` scorecard.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spread_proxy",
    "requires": ["predictions"],
    "outputs": [
        "spread_proxy_summary.json: spread of the model and of ENFO per variable.",
        "summary_by_lead.csv: the same numbers by lead time.",
        "a spread comparison figure (written by plot).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
