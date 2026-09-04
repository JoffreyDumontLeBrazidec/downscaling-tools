"""Fair CRPS against surface station observations (STVL), as a function of lead time.

A cheap, numeric stand-in for quaver's surface scorecard. Calibrated against
quaver on ja6y (2 m temperature, northern hemisphere, 2026-09-04): identical
station counts and agreement to 0.09 per cent on average. Self-skips when the
run did not publish an ensemble to FDB, so it is safe in a default group.
"""

from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "obs_crps",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
