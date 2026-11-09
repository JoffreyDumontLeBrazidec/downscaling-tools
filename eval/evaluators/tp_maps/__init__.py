"""tp map evaluator: the map-video look (9 Nov 2026) inside ``eval.cli evaluate``.

For the run's prediction files it renders, with ``eval.tools.map_video``: one still per file
(``stills/tp_<init>_+<lead>h.png``) and, for every base date with two or more leads, one MP4 over
the leads (``tp_<init>.mp4``). Panels are chosen per run (``auto``): the model, plus the truth and
the input when the files hold a non-empty tp for them (the 9 km input carries no tp). The colours
are the precipitation classes of the owner's reference frame (0.1-50 mm, white below); the grid is
0.1 degrees on the globe and 0.05 degrees on a crop, one cell per pixel.

Lane block ``tp_maps`` (all optional): ``variable`` (tp), ``member`` (0), ``dates``, ``steps``,
``max_dates`` (3: the first dates when none are given), ``res``, ``extent`` [W, E, S, N],
``palette`` (classes | rain), ``accum_hours``, ``tween`` (2), ``fps`` (4), ``video`` (true),
``geography`` (an .npz cache; default the cartopy Natural Earth files), ``ffmpeg``.

Diagnostic only; nothing reaches a scoreboard. Without ffmpeg the stills are still written.
"""
from eval.evaluators.base import no_plot, no_score
from .runner import run

EVALUATOR_SPEC = {
    "name": "tp_maps",
    "requires": ["predictions"],
    "outputs": [
        "stills/<variable>_<init>_+<lead>h.png: one frame per prediction file.",
        "<variable>_<init>.mp4: one video per base date with two or more leads (needs ffmpeg).",
    ],
}

score = no_score
plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
