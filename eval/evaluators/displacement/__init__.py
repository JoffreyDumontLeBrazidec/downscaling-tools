"""Displacement evaluator: does the model move features away from its driver?

Inside each geographical box of the lane, the model output and the driver (the
interpolated input) are sampled onto a regular longitude-latitude mesh, smoothed
to the scales the driver can carry, and compared under every whole-cell shift in a
search window. The shift with the highest correlation is the displacement, reported
in kilometres, together with the distance between the two pressure minima. The
scatter over (file, member) samples is the null that a claimed shift must beat.
Diagnostic only; no scoreboard row.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "displacement",
    "requires": ["predictions"],
    "outputs": [
        "displacement.json: displacement per box, field and pair of sources, with sample scatter.",
        "displacement_summary.md: the same numbers as a readable table.",
        "displacement_<box>_<field>.png: offset scatter and correlation gain (written by plot).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
