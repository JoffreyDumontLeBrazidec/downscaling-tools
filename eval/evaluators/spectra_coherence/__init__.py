"""Spectra-coherence evaluator: per-scale amplitude ratio vs phase coherence."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spectra_coherence",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
