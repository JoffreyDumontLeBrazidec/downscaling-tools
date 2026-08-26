"""Precipitation skill scores evaluator (6h-window tp, mm).

Scores model tp and the interpolation baseline against 6h-window truth on the
same step and grid, per member and per ensemble mean. See runner.run for the
truth/baseline source-resolution rules and scorer.score for the scoreboard
records (which always include the interp-baseline row).
"""
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "precip_scores",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
}
