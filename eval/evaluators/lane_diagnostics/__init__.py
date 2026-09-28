"""Diagnostic figure bundle for a downscaling lane.

Reads measurements that already exist on disk, makes the few that need a
compute node, and renders one labelled bundle of figures with captions that
state the support, the sample size and the arm behind every number.

Deliberately produces no scoreboard row: this evaluator explains a result that
has already been scored, it does not predict a new one.
"""
from .runner import plot, run, score

EVALUATOR_SPEC = {
    "name": "lane_diagnostics",
    "requires": ["predictions"],
    "outputs": [
        "manifest.json and CAPTIONS.md: what each figure shows and the support behind it.",
        "<number>_<slug>.pdf: one figure per diagnostic, plus a combined o1280_o2560_diagnostic_bundle.pdf.",
        "loss_budget.json, sampler_peaks.json, pair_coherence.json, box_wind.json: the measurements behind the figures.",
    ],
}


__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
