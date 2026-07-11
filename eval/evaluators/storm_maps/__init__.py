"""Storm maps + full spectra evaluator (regional TC lanes).

Renders, on top of the usual regional plots, a storm-map panel (10 m wind + MSL, TRUTH vs
MODEL vs INPUT, zoomed on the deepest-eye storm) and the full radial power spectra at all
wavenumbers. Diagnostic only (no scoreboard). Backend: eval._backends.storm_maps.render.
"""
from .runner import run

EVALUATOR_SPEC = {
    "name": "storm_maps",
    "default_enabled": False,
    "scoreboard": False,
    "requires": ["predictions"],
}
