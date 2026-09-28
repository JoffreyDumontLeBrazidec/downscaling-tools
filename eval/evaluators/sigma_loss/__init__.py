"""sigma_loss evaluator — per-sigma denoiser-loss profile.

Produces, for a lane + checkpoint, the per-sigma per-variable F-space
(network-output) loss profile via SINGLE forward passes (no diffusion sampling),
reusing the manual_inference model loader. Diagnostics-only; not run by default.

requires "checkpoint" (NOT "predictions"): it runs the model, like sigma /
mechanistic.
"""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "sigma_loss",
    "requires": ["checkpoint"],
    "outputs": [
        "data/sigma_loss/per_sigma.csv: the F-space loss per noise level and variable.",
        "data/sigma_loss/meta.json: the sigma grid and sigma_data used.",
        "data/sigma_loss/metrics.json: the scoreboard rows (written by score).",
        "plots/sigma_loss/view_a_per_sigma_loss.png: the loss-against-sigma figure (written by plot).",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
