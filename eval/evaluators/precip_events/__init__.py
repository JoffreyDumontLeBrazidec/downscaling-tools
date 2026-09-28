"""Heavy-precipitation event maps.

Selects the top-N (date, step) slices by maximum truth precipitation, renders each
as a bounding-box-cropped six-panel map, and merges the pages into one PDF. Reads
lane_config["precip_events"] (n_events, dlat, dlon, rank_by) and
lane_config["precip"] for the truth and baseline GRIB fallbacks. Diagnostic only;
no scoreboard row.
"""
from eval.evaluators.base import no_plot, no_score
from .runner import run

EVALUATOR_SPEC = {
    "name": "precip_events",
    "requires": ["predictions"],
    "outputs": [
        "events.json: the selected (date, step) events and their ranking values.",
        "precip_events_local.pdf: one event-centred page per selected event.",
    ],
}

score = no_score
plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
