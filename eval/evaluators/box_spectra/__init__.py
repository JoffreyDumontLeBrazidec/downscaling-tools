"""box_spectra evaluator: the per-bin box spectrum of standing rule 14.

Amplitude ratios per wavelength bin on a box, for the full field (prediction over truth), the
residual (prediction minus interpolated input, over truth minus interpolated input) and the
input (interpolated input over truth), with the coherence and the aligned part, pooled and per
date. Port of the docs tool ``plot_sampler_texture.py --spectra-method linear`` (gap G2 of the
run-judging protocol). Diagnostic only: no scoreboard row.
"""
from eval.evaluators.base import no_score
from .runner import run
from .plotter import plot

score = no_score

EVALUATOR_SPEC = {
    "name": "box_spectra",
    "requires": ["predictions"],
    "outputs": [
        "bins.csv: per variable and wavelength bin, pooled (scope all) and per date: amplitudes "
        "and the residual, full-field and input ratios, coherence, aligned part.",
        "bins.md: the pooled per-bin table.",
        "box_spectra_ladder.png: the ratios against the wavelength ladder (written by plot).",
        "run_meta.json: box, grid, files and members used.",
    ],
}

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
