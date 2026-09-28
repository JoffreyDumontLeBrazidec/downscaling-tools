"""Per-leadtime (per-step) surface scores and spectra evaluator."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "leadtime",
    "default_enabled": True,
    "scoreboard": False,  # diagnostic-only; not folded into the scoreboard composite
    "requires": ["predictions"],
}
