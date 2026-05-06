"""Surface (nMSE) evaluator."""
from .scorer import score

EVALUATOR_SPEC = {
    "name": "surface",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
}
