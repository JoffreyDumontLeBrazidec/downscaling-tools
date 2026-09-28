"""Local/global parity evaluator."""
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "local_global",
    "requires": ["predictions"],
}
