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
    "requires": ["predictions"],
    "outputs": [
        "quaver.pdf: the scorecard figure.",
        "params.json, input_params.json, reference_params.json: the parameters passed to quaver for the run, the input and the reference.",
        "effective_config.json: the configuration the evaluator used.",
        "skipped.json: written instead of the above when the evaluator skipped itself.",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
