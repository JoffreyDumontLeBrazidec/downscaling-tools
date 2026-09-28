"""Probabilistic spread and CRPS evaluator (local, from the prediction files).

Computes the ensemble spread, the CRPS and the ensemble-mean error of the model
from the members in the prediction files, per variable, region and lead time.
This is the local, cheap probabilistic check; the canonical probabilistic verdict
is the `quaver` scorecard. No scoreboard row.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "probabilistic",
    "requires": ["predictions"],
    "outputs": [
        "probabilistic_summary.json: CRPS, spread and ensemble-mean error per variable and region.",
        "summary_by_lead.csv: the same numbers by lead time.",
        "probabilistic_scores.pdf: summary figure (written by plot).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
