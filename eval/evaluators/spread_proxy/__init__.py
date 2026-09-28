"""Spread-proxy evaluator: ML (y_pred) vs ENFO (y) ensemble spread."""

from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spread_proxy",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
