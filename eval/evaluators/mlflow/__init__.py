"""MLflow training-loss evaluator.

Finds the MLflow run that trained the checkpoint (from the run identifier in the
checkpoint path, on Atos or through a copy synchronised from Jupiter), loads its
logged losses, and plots them. It soft-fails: when no matching MLflow run exists it
logs a warning and returns. Requires --checkpoint. Diagnostic only; no scoreboard
row.
"""
from eval.evaluators.base import no_score
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "mlflow",
    "requires": ["checkpoint"],
    "outputs": [
        "loss_curves.png, overview.png, key_vars.png, all_vars.png: training and validation loss figures (written by run).",
        "metrics.json: the loaded loss series.",
        "import_log.txt: what the importer found and did.",
    ],
}

score = no_score

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
