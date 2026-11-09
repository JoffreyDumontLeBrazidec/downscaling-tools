"""tp map evaluator: the map-video look (9 Nov 2026) inside ``eval.cli evaluate``. Standard on tp lanes.

It runs by default on the lanes that predict tp (their ``default`` group) and joins any selection
that holds a tp evaluator (precip_read, precip_dist, precip_events, precip_scores), ``--only`` too;
``--no-tp-maps`` leaves it out. Files without tp in ``y_pred`` give a SKIPPED.txt, not a failure.

For each base date it renders, with ``eval.tools.map_video``:

* the whole map: ``stills/tp_<init>_+<lead>h.png`` (one per file) and ``tp_<date>.mp4`` (two or more leads);
* one zoom per region, at 0.05 degrees: ``crops/<name>/stills/...`` and ``crops/<name>/tp_<date>.mp4``.
  On a whole global grid the regions default to west_pacific, atlantic and europe; a crop run (the
  E2 box) is already a zoom and has none unless the lane names some;
* ``previews/tp_<date>[_<name>].png|mp4``: the middle lead at 1600 px and the video at 1280 px,
  each under 1 MB, ready for the docs repository or a chat.

Panels are chosen per run (``auto``): the model, plus the truth and the input when the files hold a
non-empty tp for them (the 9 km input carries no tp). The colours are the precipitation classes of
the owner's reference frame (0.1-50 mm, white below); the grid is 0.1 degrees on the globe, one cell
per pixel.

Lane block ``tp_maps`` (all optional): ``variable`` (tp), ``member`` (0), ``dates``, ``steps``,
``max_dates`` (3: the first dates when none are given), ``res``, ``extent`` [W, E, S, N] (the main
map), ``crops`` ({name: [W, E, S, N]}; false for none), ``crop_res`` (0.05), ``previews`` (true),
``palette`` (classes | rain), ``accum_hours``, ``tween`` (2), ``fps`` (4), ``video`` (true),
``geography`` (an .npz cache; default the cartopy Natural Earth files), ``ffmpeg``.

Nothing reaches a scoreboard. Without ffmpeg the stills and PNG previews are still written.
"""
from eval.evaluators.base import no_plot, no_score
from .runner import run

EVALUATOR_SPEC = {
    "name": "tp_maps",
    "requires": ["predictions"],
    "outputs": [
        "stills/<variable>_<init>_+<lead>h.png: one frame per prediction file.",
        "<variable>_<init>.mp4: one video per base date with two or more leads (needs ffmpeg).",
        "crops/<name>/{stills/*.png,<variable>_<init>.mp4}: the same per zoom region.",
        "previews/<variable>_<init>[_<name>].{png,mp4}: copies under 1 MB for notes and chat.",
    ],
}

score = no_score
plot = no_plot

__all__ = ["run", "score", "plot", "EVALUATOR_SPEC"]
