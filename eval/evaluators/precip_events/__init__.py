"""Heavy-precip local-plot evaluator.

Selects the top-N (date, step) slices by max truth tp, then renders each as a
bbox-cropped 6-panel region_plot map, merged into one PDF.
"""
from .runner import run

EVALUATOR_SPEC = {
    "name": "precip_events",
    "default_enabled": True,
    "scoreboard": False,
    "requires": ["predictions"],
}
