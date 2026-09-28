"""Wind-extreme evaluator: is the strongest 10 m wind a real feature or grain?

Measured on the native O1280 grid with no regridding. The way a wind maximum
survives averaging over a disk of growing radius tells a coherent structure (most
of the amplitude remains) from grid-scale grain (it collapses towards the local
mean). The verdict comes from the model-minus-truth difference of the retention
ratio, with the scatter over (file, member) samples as the null. Also reports the
raw maximum, the size of the connected patch above 90 percent of it, and the
distance between the maxima of two sources. Diagnostic only; no scoreboard row.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "wind_extremes",
    "requires": ["predictions"],
    "outputs": [
        "wind_extremes.json: retention, peak, patch size and peak displacement per box and source.",
        "wind_extremes_summary.md: the same numbers as a readable table.",
        "wind_extremes_<box>.png: one figure per box (written by plot).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
