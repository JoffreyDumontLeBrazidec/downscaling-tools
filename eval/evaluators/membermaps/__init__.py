"""Member maps evaluator — four-panel case-inspection maps for one run.

Renders the O320 driving input, the embedded ENFO truth and this run's
prediction, as the field itself and as the fine-scale high-pass view. Purely
diagnostic: it produces figures and never a score.
"""
from .runner import run

EVALUATOR_SPEC = {
    "name": "membermaps",
    "requires": ["predictions"],
}
