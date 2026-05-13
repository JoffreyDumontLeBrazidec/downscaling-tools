"""eval._backends.videogen — modular MP4 video generator for downscaling predictions.

Quick start::

    from eval._backends.videogen import SCENES, make_video
    make_video(SCENES["franklin_dual"])

Or from the CLI::

    python -m eval.cli videogen --scene franklin_dual --mode all
    python -m eval._backends.videogen --scene franklin_dual --mode preview

Module layout
-------------
config.py     SceneConfig dataclass — single source of truth per video.
scenes.py     Pre-built SceneConfig instances (SCENES registry).
data.py       Frame stitching, var slicing, bbox masks, regridding, TC tracking.
panels.py     Cartopy + cmcrameri panel rendering (pcolormesh + contours).
layouts.py    LAYOUT_RENDERERS: render_dual_row, render_single_inset, …
pipeline.py   compute_norms, render_all_frames, encode_mp4, make_video.
__main__.py   CLI dispatcher.

Adding a new scene
------------------
Either edit ``scenes.py`` (add to the ``SCENES`` dict) or build a fresh
``SceneConfig`` at call site.

Adding a new layout
-------------------
Write a ``render_<name>(frame, scene, norms, out_png)`` in ``layouts.py`` and
register it in ``LAYOUT_RENDERERS``. Set ``layout="<name>"`` on the scene.
"""
from .config import SceneConfig
from .pipeline import (
    compute_norms,
    encode_mp4,
    make_video,
    render_all_frames,
    render_one_frame,
    render_preview,
)
from .scenes import SCENES

__all__ = [
    "SCENES",
    "SceneConfig",
    "compute_norms",
    "encode_mp4",
    "make_video",
    "render_all_frames",
    "render_one_frame",
    "render_preview",
]
