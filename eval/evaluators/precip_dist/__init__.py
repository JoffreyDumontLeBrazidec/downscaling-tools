"""Precipitation value-distribution evaluator.

Wraps eval.evaluators.precip_dist.core.tp_histogram_comparison. For each lead time it
compares the histogram of the model's precipitation values with the truth's, and
writes one multi-page PDF. Reads lane_config["precip_dist"] for tunables and
lane_config["precip"] for the truth and baseline GRIB fallbacks. Diagnostic only;
no scoreboard row.
"""
from eval.evaluators.base import no_plot, no_score
from .runner import run

EVALUATOR_SPEC = {
    "name": "precip_dist",
    "requires": ["predictions"],
    "outputs": [
        "tp_histograms.pdf: one page per lead time, truth against prediction.",
    ],
}

score = no_score
plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
