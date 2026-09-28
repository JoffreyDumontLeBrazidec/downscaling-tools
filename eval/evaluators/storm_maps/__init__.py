"""Storm maps and full spectra evaluator (regional tropical cyclone lanes).

Renders a storm-map figure (10 m wind and sea-level pressure, truth against model
against input, zoomed on the deepest-eye storm) and the full radial power spectra
at all wavenumbers for 10u, 10v and msl. Reads the event and storm box from the
lane's `tc` block, and falls back to the tc_atlantic_mdr_west box. Diagnostic
only; no scoreboard row. Backend: eval._backends.storm_maps.render.

Lane configuration, all keys optional (the `storm_maps:` block of the lane file):

  box        The region the spectra and the map are computed in. A mapping with
             lat_min, lat_max, lon_min and lon_max, or a list [lat0, lat1, lon0, lon1].
             Default: lat 5 to 35, lon -100 to -40 (the tc_atlantic_mdr_west box).
  storm_box  The region searched for the deepest sea-level pressure minimum, same
             format as `box`. Default: the `tc` block's `storm_box` (or `box`) when
             the lane sets one, otherwise lat 10 to 35, lon -100 to -80.
  steps      Lead times in hours to render, for example [24, 72, 120]. Default: [72].
             One lead time writes its figures directly in the evaluator folder, as
             before; several lead times write one folder per lead time,
             `step024/`, `step072/` and so on, each holding the same files.

Example:

  storm_maps:
    box: {lat_min: 5, lat_max: 30, lon_min: -80, lon_max: -50}
    steps: [48, 72]
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
