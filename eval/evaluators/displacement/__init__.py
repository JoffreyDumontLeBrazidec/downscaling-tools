"""Displacement evaluator: does the model move features away from its driver?"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "displacement",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
