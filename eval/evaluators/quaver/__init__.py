"""Quaver (Probabilistic Metrics) evaluator."""
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "quaver",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
