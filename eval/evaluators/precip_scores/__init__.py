"""Precipitation skill scores evaluator (6-hour window precipitation, mm).

Scores the model's precipitation and the interpolation baseline against
6-hour-window truth on the same step and grid, per member and for the ensemble
mean: RMSE, bias, correlation, ratios of the 99.9th percentile, the maximum and
the wet fraction, and the model-to-baseline RMSE ratio. The interpolation-baseline
row is part of the contract, so "does the model beat interpolating its input" can
be answered from the scoreboard alone. See runner.run for the truth and baseline
source resolution rules and scorer.score for the scoreboard records.
"""
from eval.evaluators.base import no_plot
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "precip_scores",
    "requires": ["predictions"],
    "outputs": [
        "scores.json: the overall summary and the per-member and per-step aggregates.",
        "scores_rows.csv: the same numbers as rows.",
        "plots/precip_scores.pdf: a summary figure (written by run).",
    ],
}

plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
