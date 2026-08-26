"""Precipitation skill scores evaluator (6h-window tp, mm).

Scores model tp and the interpolation baseline against 6h-window truth on the
same step and grid, per member and per ensemble mean. See runner.run for the
truth/baseline source-resolution rules.
"""
from .runner import run

EVALUATOR_SPEC = {
    "name": "precip_scores",
    "default_enabled": True,
    "scoreboard": False,
    "requires": ["predictions"],
}
