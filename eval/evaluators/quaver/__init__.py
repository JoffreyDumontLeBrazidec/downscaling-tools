"""Quaver probabilistic scorecard evaluator (FDB-based, ECMWF `quaver` binary).

Wired into the eval.cli evaluator framework so that prepml runs (which publish
their ensemble to FDB under an expver) always get a quaver CRPS/spread scorecard.
Self-skips for manual runs (no expver / no FDB output), so it is safe to keep in
a lane's default evaluator group.
"""

from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "quaver",
    "default_enabled": True,
    "scoreboard": False,
    "requires": ["predictions"],
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
