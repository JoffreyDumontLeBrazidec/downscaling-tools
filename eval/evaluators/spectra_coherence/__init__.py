"""Spectra-coherence evaluator: per-scale amplitude ratio against phase coherence.

A power spectrum is blind to where the energy sits. For every spherical-harmonic
degree this evaluator computes, against the truth, the amplitude ratio R(l) (1
means the right amount of energy) and the coherence C(l) (1 means perfectly in
phase), which decompose the normalised per-degree error exactly. The floor
1 - C(l)^2 is the smallest error any rescaling of the prediction could reach at
that scale. Band-level scores are emitted by score(); diagnostic only, no
scoreboard row.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spectra_coherence",
    "requires": ["predictions"],
    "outputs": [
        "coherence.json and coherence_by_surface.json: R(l), C(l) and the per-degree error per weather state.",
        "calibration.json: the calibration of the estimate.",
        "spectra_coherence.png/.pdf and coherence_by_surface.png/.pdf (written by plot).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
