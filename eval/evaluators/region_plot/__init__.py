"""Region plot evaluator: model, truth and input fields side by side.

A subprocess wrapper around eval.evaluators.region_plot.core.plot_regions. It passes
every region box of the lane's `regions` block to the backend, which renders the
six-panel comparison of each box for the first prediction file. Diagnostic only;
no scoreboard row.
"""
from eval.evaluators.base import no_score
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "region_plot",
    "requires": ["predictions"],
    "outputs": [
        "all_regions_plots.pdf: the combined figure of every lane region, written by eval.evaluators.region_plot.core.plot_regions.",
    ],
}

score = no_score

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
