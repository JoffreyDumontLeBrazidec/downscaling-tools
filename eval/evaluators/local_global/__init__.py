"""Local/global parity evaluator.

Compares predictions made by running the model on a local cut-out of the globe
with predictions made by running it on the whole globe, and writes the difference
statistics. Diagnostic only; no scoreboard row.
"""
from eval.evaluators.base import no_plot
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "local_global",
    "requires": ["predictions"],
    "outputs": [
        "local_global_parity.json: parity statistics between the local and the global run.",
    ],
}

plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
