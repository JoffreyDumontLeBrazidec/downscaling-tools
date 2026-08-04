"""Spectra (Power Spectra) evaluator."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "spectra",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
    # Promote the consolidated multi-page PDF with the canonical lean name.
    # Per-variable PDFs ride the default plots/ subdir. See eval.lean_layout.
    "deliverables": {
        "top_level": [
            {"src": "all_spectra_proxy.pdf", "as": "spectra_proxy.pdf"},
        ],
    },
}
