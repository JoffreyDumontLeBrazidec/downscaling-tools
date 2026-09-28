"""Storm maps and full spectra evaluator (regional tropical cyclone lanes).

Renders a storm-map figure (10 m wind and sea-level pressure, truth against model
against input, zoomed on the deepest-eye storm) and the full radial power spectra
at all wavenumbers for 10u, 10v and msl. Reads the event and storm box from the
lane's `tc` block, and falls back to the tc_atlantic_mdr_west box. Diagnostic
only; no scoreboard row. Backend: eval._backends.storm_maps.render.
"""
from eval.evaluators.base import no_plot, no_score
from .runner import run

EVALUATOR_SPEC = {
    "name": "storm_maps",
    "requires": ["predictions"],
    "outputs": [
        "storm_maps.png: 10 m wind and sea-level pressure, truth against model against input, zoomed on the deepest storm.",
        "full_spectra.png: full radial power spectra at all wavenumbers for 10u, 10v and msl.",
    ],
}

score = no_score
plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
