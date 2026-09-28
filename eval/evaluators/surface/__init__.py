"""Surface evaluator: normalised mean squared error of the surface variables.

Computes the model's error against the truth on the surface variables, normalised
per variable, and an area- and variable-weighted total (the weighting comes from
the lane's `surface.weighting`, default "truth-std"). The scoring functions are the
ones of eval._backends.scoreboard.surface, imported directly so the numbers are
identical to the scoreboard's. This is the surface column of the scoreboard.
"""
from eval.evaluators.base import no_plot
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "surface",
    "requires": ["predictions"],
    "outputs": [
        "surface_loss.json: weighted and per-variable normalised mean squared error.",
    ],
}

plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
