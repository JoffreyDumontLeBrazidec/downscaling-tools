"""Texture evaluator: fine-scale texture statistics measured on the native O1280 grid."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "texture",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
