"""tc_structure evaluator: tropical-cyclone structure (centre, Pmin, tangential-wind
profile, radius of maximum wind, wind radii, circulation vorticity, asymmetry,
wind-pressure relation, centre displacement) for model, truth and input.

Design and validation: docs/epics/fine-scale-o320-o1280/in-progress/
20260923_physical_realism_scores.md. Diagnostic only: not in any default group and
not a scoreboard evaluator.
"""
from eval.evaluators.base import no_plot
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "tc_structure",
    "requires": ["predictions"],
    "outputs": [
        "summary.json: structure statistics per event for the model, the truth and the input.",
        "cases.csv: the per-case rows behind the summary.",
        "run_meta.json: which predictions and settings the run used.",
    ],
}

plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
