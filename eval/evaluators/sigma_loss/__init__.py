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
    "scoreboard": True,
    "default_enabled": False,
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
