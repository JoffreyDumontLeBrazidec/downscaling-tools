"""Diagnostic figure bundle for a downscaling lane.

Reads measurements that already exist on disk, makes the few that need a
compute node, and renders one labelled bundle of figures with captions that
state the support, the sample size and the arm behind every number.

Deliberately produces no scoreboard row: this evaluator explains a result that
has already been scored, it does not predict a new one.
"""
from .runner import plot, run, score

EVALUATOR_SPEC = {
    "name": "lane_diagnostics",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
