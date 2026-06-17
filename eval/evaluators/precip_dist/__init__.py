"""Precipitation value-distribution (PDF/histogram) evaluator.

Wraps eval._backends.precip.tp_histogram_comparison. Produces one multi-page PDF
with tp value histograms per lead (truth vs prediction).
"""
from .runner import run

EVALUATOR_SPEC = {
    "name": "precip_dist",
    "default_enabled": True,
    "scoreboard": False,
    "requires": ["predictions"],
}
