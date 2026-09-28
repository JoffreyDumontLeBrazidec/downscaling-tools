"""Texture evaluator: fine-scale texture statistics on the native output grid.

Power spectra cannot tell grain (about the right small-scale energy, placed point
by point) from truth-like texture, because they are blind to phase. This evaluator
measures texture directly on the native O1280 (or O2560) points with no
regridding, so the model output and the truth receive identical treatment: the
variance of the fine part, the lag-1 zonal and nearest-neighbour correlations, the
share of the five largest values, the kurtosis, and a grain index
(model - truth) / (noise - truth) where 0 means truth-like and 1 means white
noise. Diagnostic only; no scoreboard row.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "texture",
    "requires": ["predictions"],
    "outputs": [
        "texture.json: texture statistics per weather state, stratum and source.",
        "texture_summary.md: the same numbers as a readable table.",
        "texture_<state>.png: one figure per weather state (written by plot).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
