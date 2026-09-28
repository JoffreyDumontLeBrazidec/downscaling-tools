"""Wind-extreme evaluator: is the strongest 10 m wind a real feature or grain?"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "wind_extremes",
    "requires": ["predictions"],
}
