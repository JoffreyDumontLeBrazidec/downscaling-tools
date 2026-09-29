"""SceneConfig — every parameter for one video.

A scene is intentionally a flat frozen dataclass so that ``dataclasses.replace``
makes it trivial to swap one field (e.g. predictions_dir) at call-site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BBox = tuple[float, float, float, float]  # (lon_min, lon_max, lat_min, lat_max)
InsetKind = Literal["fixed", "track_msl_min"]
Layout = Literal["single_inset", "dual_row"]


@dataclass(frozen=True)
class SceneConfig:
    # --- identity --------------------------------------------------------
    name: str
    title: str = ""
    ckpt_label: str = ""

    # --- IO --------------------------------------------------------------
    predictions_dir: Path = Path(".")
    output_dir: Path = Path(".")

    # --- frame selection -------------------------------------------------
    inits: tuple[str, ...] = ()                  # YYYYMMDD strings
    steps: tuple[int, ...] = (24, 48, 72, 96, 120)
    ensemble_member: int = 0

    # --- geometry --------------------------------------------------------
    bg_bbox: BBox = (-180.0, 180.0, -90.0, 90.0)
    inset_kind: InsetKind = "fixed"
    inset_bbox: BBox | None = None               # used when inset_kind == "fixed"
    inset_half_deg: float = 4.0                  # half-side when tracked
    inset_search_bbox: BBox | None = None        # used when inset_kind == "track_msl_min"

    # --- variables + layout ---------------------------------------------
    vars: tuple[str, ...] = ("msl",)
    layout: Layout = "single_inset"

    # --- rendering -------------------------------------------------------
    bg_resolution_deg: float = 0.35
    hres_resolution_deg: float = 0.05
    fps: int = 3
    dpi: int = 220

    # --- ffmpeg ---------------------------------------------------------
    ffmpeg_module: str | None = "ffmpeg/7.1.1"   # set None if ffmpeg is on PATH
    crf: int = 18

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "frames" / self.name

    @property
    def preview_path(self) -> Path:
        return self.output_dir / f"preview_{self.name}.png"

    @property
    def video_path(self) -> Path:
        return self.output_dir / f"video_{self.name}.mp4"

    @property
    def norm_cache_path(self) -> Path:
        return self.output_dir / "norm_cache.json"
