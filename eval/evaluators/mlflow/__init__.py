"""MLflow training-loss evaluator."""
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "mlflow",
    "requires": ["checkpoint"],
}
