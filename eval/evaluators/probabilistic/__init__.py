"""Probabilistic spread/CRPS evaluator."""

from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "probabilistic",
    "requires": ["predictions"],
}
