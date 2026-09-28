"""Region Plot (6-Panel Comparisons) evaluator."""
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "region_plot",
    "requires": ["predictions"],
}
