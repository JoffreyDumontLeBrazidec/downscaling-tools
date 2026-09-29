"""Zoom maps evaluator (called membermaps until 2026-09-29): case-inspection maps, driven by a lane.

Given one predictions directory it renders, for every region in the lane's own
configuration and every variable asked for, the driving O320 input, the embedded
same-index ENFO member as truth, and this run's prediction. It does that twice:
once as the field itself and once as the high-pass view that shows only the detail
the O320 driver could not carry. Unlike the `eval.cli zoom_maps` command, which
must be told every detail, this evaluator takes them from the lane.

Diagnostic only. Nothing here scores anything, so nothing reaches a scoreboard.
"""
from eval.evaluators.base import no_plot, no_score
from .runner import run

EVALUATOR_SPEC = {
    "name": "zoom_maps",
    "requires": ["predictions"],
    "outputs": [
        "<region>/: one sub-directory per lane region holding the rendered maps (file names come from eval/evaluators/zoom_maps/core/plot_member_wind_maps.py).",
    ],
}

score = no_score
plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
