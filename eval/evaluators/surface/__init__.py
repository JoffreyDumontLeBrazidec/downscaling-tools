"""Surface (nMSE) evaluator."""
from .runner import run
from .scorer import score

EVALUATOR_SPEC = {
    "name": "surface",
    "requires": ["predictions"],
}
