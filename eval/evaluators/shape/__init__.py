"""shape evaluator: port of the feature-shape instrument (elongated fraction, window
aspect ratio, flow-relative anisotropy index of the band-passed 10u/10v residuals),
with date-clustered uncertainty. Design note:
docs/epics/fine-scale-o320-o1280/in-progress/20260923_physical_realism_scores.md.
Diagnostic only: not in any default group and not a scoreboard evaluator.
"""
from eval.evaluators.base import no_plot
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "shape",
    "requires": ["predictions"],
    "outputs": [
        "summary.json: the shape statistics with date-clustered uncertainty.",
        "rows.csv: the per-sample rows behind the summary.",
        "run_meta.json: which predictions and settings the run used.",
    ],
}

plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
