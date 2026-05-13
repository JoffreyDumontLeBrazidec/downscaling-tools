"""End-to-end orchestration: norms, frames, encode.

Public functions:
  * ``compute_norms``        — (vmin, vmax) per variable, with JSON cache.
  * ``render_one_frame``     — dispatch to a layout renderer.
  * ``render_preview``       — single-frame preview with single-frame norm.
  * ``render_all_frames``    — all frames in a scene, using cached global norm.
  * ``encode_mp4``           — ffmpeg wrapper.
  * ``make_video``           — render_all_frames + encode_mp4.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .config import SceneConfig
from .data import Frame, bbox_mask, build_frame_list, load_var_slice, resolve_inset_bbox
from .layouts import LAYOUT_RENDERERS

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Norms (with cache)
# ---------------------------------------------------------------------------

def _vars_needed(scene: SceneConfig) -> set[str]:
    """Vars to compute norms for. dual_row's bg always shows msl."""
    required = set(scene.vars)
    if scene.layout == "dual_row":
        required.add("msl")
    return required


def compute_norms(
    frames: list[Frame], scene: SceneConfig,
) -> dict[str, tuple[float, float]]:
    """Global (vmin, vmax) per variable across all frames, cached to JSON."""
    cache_path = scene.norm_cache_path
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    out: dict[str, tuple[float, float]] = {}
    to_compute: list[str] = []
    for v in _vars_needed(scene):
        key = f"{scene.name}::{v}"
        if key in cache:
            out[v] = (cache[key]["vmin"], cache[key]["vmax"])
        else:
            to_compute.append(v)

    if not to_compute:
        return out

    running = {v: (np.inf, -np.inf) for v in to_compute}
    for f in frames:
        LOG.info("norm scan: %s", f.nc_path.name)
        with xr.open_dataset(f.nc_path) as ds:
            inset_bbox = resolve_inset_bbox(ds, scene)
            for v in to_compute:
                lon, lat, vals = load_var_slice(ds, "lres", v, ensemble_member=scene.ensemble_member)
                m = bbox_mask(lon, lat, scene.bg_bbox)
                cur_min, cur_max = running[v]
                if m.any():
                    cur_min = min(cur_min, float(vals[m].min()))
                    cur_max = max(cur_max, float(vals[m].max()))
                lon_h, lat_h, vals_h = load_var_slice(ds, "hres", v, ensemble_member=scene.ensemble_member)
                m_h = bbox_mask(lon_h, lat_h, inset_bbox)
                if m_h.any():
                    cur_min = min(cur_min, float(vals_h[m_h].min()))
                    cur_max = max(cur_max, float(vals_h[m_h].max()))
                running[v] = (cur_min, cur_max)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for v, (vmin, vmax) in running.items():
        cache[f"{scene.name}::{v}"] = {"vmin": vmin, "vmax": vmax}
        out[v] = (vmin, vmax)
    cache_path.write_text(json.dumps(cache, indent=2))
    return out


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_one_frame(
    frame: Frame, scene: SceneConfig,
    norms: dict[str, tuple[float, float]],
    out_png: Path,
) -> None:
    renderer = LAYOUT_RENDERERS.get(scene.layout)
    if renderer is None:
        raise ValueError(f"unknown layout: {scene.layout!r}; known: {list(LAYOUT_RENDERERS)}")
    renderer(frame, scene, norms, out_png)


def render_preview(
    scene: SceneConfig, valid_time: datetime | None = None,
) -> Path:
    """Render a single preview frame using a single-frame norm.

    If ``valid_time`` is None, pick the middle frame. The preview goes to
    ``scene.preview_path`` (does not enter ``scene.frames_dir``).
    """
    frames = build_frame_list(scene)
    if not frames:
        raise FileNotFoundError(f"no prediction files under {scene.predictions_dir}")
    if valid_time is not None:
        chosen = next((f for f in frames if f.valid_time == valid_time), None)
        if chosen is None:
            raise ValueError(f"no frame at valid_time {valid_time}")
    else:
        chosen = frames[len(frames) // 2]

    norms: dict[str, tuple[float, float]] = {}
    with xr.open_dataset(chosen.nc_path) as ds:
        inset_bbox = resolve_inset_bbox(ds, scene)
        for v in _vars_needed(scene):
            lon, lat, vals = load_var_slice(ds, "lres", v, ensemble_member=scene.ensemble_member)
            m = bbox_mask(lon, lat, scene.bg_bbox)
            v_l = vals[m]
            lon_h, lat_h, vals_h = load_var_slice(ds, "hres", v, ensemble_member=scene.ensemble_member)
            m_h = bbox_mask(lon_h, lat_h, inset_bbox)
            norms[v] = (
                float(min(v_l.min(), vals_h[m_h].min())),
                float(max(v_l.max(), vals_h[m_h].max())),
            )

    out_png = scene.preview_path
    out_png.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("preview %s -> %s", chosen.label(), out_png)
    render_one_frame(chosen, scene, norms, out_png)
    return out_png


def render_all_frames(scene: SceneConfig) -> Path:
    """Render every frame to ``scene.frames_dir``; returns that dir."""
    frames = build_frame_list(scene)
    if not frames:
        raise FileNotFoundError(f"no prediction files under {scene.predictions_dir}")
    LOG.info("scene=%s  frames=%d", scene.name, len(frames))
    norms = compute_norms(frames, scene)
    for v, (vmin, vmax) in norms.items():
        LOG.info("  norm %s: vmin=%g vmax=%g", v, vmin, vmax)
    frames_dir = scene.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        out_png = frames_dir / f"frame_{i:03d}.png"
        LOG.info("[%d/%d] %s -> %s", i + 1, len(frames), f.label(), out_png.name)
        render_one_frame(f, scene, norms, out_png)
    return frames_dir


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

def encode_mp4(
    frames_dir: Path, out_mp4: Path, *,
    fps: int = 3, crf: int = 18,
    ffmpeg_module: str | None = "ffmpeg/7.1.1",
) -> Path:
    """Run ffmpeg to make ``out_mp4`` from ``frames_dir/frame_%03d.png``.

    On hpc-login ffmpeg lives behind ``module load``; if ``ffmpeg_module`` is
    set we wrap the command in ``bash -lc 'module load ... && ffmpeg ...'``.
    The ``scale=trunc(iw/2)*2:trunc(ih/2)*2`` filter rounds dimensions to even
    so libx264 + yuv420p don't reject odd-pixel PNGs.
    """
    pattern = str(frames_dir / "frame_%03d.png")
    cmd = (
        f"ffmpeg -y -framerate {fps} -i '{pattern}' "
        f"-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' "
        f"-c:v libx264 -pix_fmt yuv420p -crf {crf} '{out_mp4}'"
    )
    if ffmpeg_module:
        cmd = f"module load {ffmpeg_module} && {cmd}"
    LOG.info("ffmpeg: %s", cmd)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bash", "-lc", cmd], check=True)
    return out_mp4


def make_video(scene: SceneConfig) -> Path:
    """End-to-end: render every frame, encode to MP4, return the .mp4 path."""
    frames_dir = render_all_frames(scene)
    return encode_mp4(
        frames_dir, scene.video_path,
        fps=scene.fps, crf=scene.crf,
        ffmpeg_module=scene.ffmpeg_module,
    )
