"""TC (Tropical Cyclone Extremes) evaluator."""
from .runner import run
from .scorer import score
from .plotter import plot

EVALUATOR_SPEC = {
    "name": "tc",
    "default_enabled": True,
    "scoreboard": True,
    "requires": ["predictions"],
    # Consolidated overview lives under plots/; promote it with the canonical
    # name. Per-member maps are an extra plot subdir. See eval.lean_layout.
    "deliverables": {
        "top_level": [
            {"src": "plots/all_tc_distributions.pdf", "as": "tc_pdf_distributions.pdf"},
        ],
        "plots": ["plots", "member_maps"],
    },
}
