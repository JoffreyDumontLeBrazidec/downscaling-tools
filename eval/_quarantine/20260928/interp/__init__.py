"""Interp evaluator — wraps interp.viz to render per-checkpoint PDFs.

The actual interp computations (feature permutation, conditioning ablation,
activation profiling, CKA) run out-of-band on AG via sbatch and dump JSONs
to ~/perm/interp/<ckpt_id>/<tool>/. This evaluator's only job is to call
interp.viz against that data and surface the PDFs through the eval CLI's
plot-consolidation step.

Layout produced under <results_dir>/:
    plots/
        feature_permutation.pdf
        feature_permutation_full_sampling.pdf   (if available)
        conditioning_ablation.pdf
        activation_profiling.pdf
        cka.pdf

The PDFs are also kept canonically at ~/perm/interp/<ckpt_id>/plots/.
"""
from .runner import run
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "interp",
    "default_enabled": True,
    "scoreboard": False,
    "requires": ["checkpoint"],
}
