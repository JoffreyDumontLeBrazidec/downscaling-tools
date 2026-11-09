"""Compose pixel-exact map frames and encode them to MP4.

One regular-grid cell is one pixel of the video, so nothing is resampled after the
interpolation from the native grid: at 0.1 degrees a global panel is 3600 x 1801 pixels and a
one-panel frame is 3840 x 2160 (4K UHD). The parts of a frame that do not change (white page,
panel labels, colour bar, coastlines and borders) are drawn once with Matplotlib; each frame is
then the coloured field (a lookup table), the coastline layer alpha-blended on top, and a new
title line.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DPI = 100
PT_PER_PX = 72.0 / DPI


@dataclass
class Layout:
    """Pixel layout of a frame: ``panels`` map rectangles (x0, y0, w, h) from the top-left corner."""

    width: int
    height: int
    panels: list[tuple[int, int, int, int]]
    header: tuple[int, int, int, int]
    colorbar: tuple[int, int, int, int]
    label_h: int = 0
    font_px: int = 26

    @classmethod
    def build(cls, map_w: int, map_h: int, n_panels: int, *, min_width: int = 3840,
              labels: bool = False) -> "Layout":
        font = int(np.clip(round(map_w / 115), 16, 34))
        margin = int(round(1.6 * font))
        gap = int(round(1.2 * font)) if n_panels > 1 else 0
        header_h = int(round(2.6 * font))
        label_h = int(round(1.7 * font)) if labels else 0
        cbar_block = int(round(5.2 * font))
        content_w = n_panels * map_w + (n_panels - 1) * gap
        width = max(content_w + 2 * margin, min_width if n_panels == 1 and map_w > 1800 else 0)
        width += width % 2
        x_start = (width - content_w) // 2
        y_map = header_h + label_h
        height = y_map + map_h + cbar_block
        height += height % 2
        if n_panels == 1 and min_width == 3840 and width == 3840 and height < 2160:
            # pad to 4K UHD: the extra rows go above and below the map evenly
            extra = 2160 - height
            header_h += extra // 2
            y_map += extra // 2
            height = 2160
        panels = [(x_start + i * (map_w + gap), y_map, map_w, map_h) for i in range(n_panels)]
        cb_w = int(min(0.5 * content_w, 1800))
        cb_h = int(round(0.9 * font))
        cb_x = (width - cb_w) // 2
        cb_y = y_map + map_h + int(round(1.2 * font))
        return cls(width=width, height=height, panels=panels,
                   header=(0, 0, width, header_h), colorbar=(cb_x, cb_y, cb_w, cb_h),
                   label_h=label_h, font_px=font)


def _axes_rect(fig_w: int, fig_h: int, rect) -> list[float]:
    x0, y0, w, h = rect
    return [x0 / fig_w, 1.0 - (y0 + h) / fig_h, w / fig_w, h / fig_h]


def _canvas_rgba(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def wrap_lines(lines: list[np.ndarray], west: float) -> np.ndarray:
    """Concatenate (n, 2) lon/lat polylines with NaN breaks, longitudes moved into [west, west + 360)."""
    parts = []
    for xy in lines:
        lon = np.mod(xy[:, 0] - west, 360.0) + west
        lat = xy[:, 1]
        cut = np.flatnonzero(np.abs(np.diff(lon)) > 180.0) + 1
        for seg_lon, seg_lat in zip(np.split(lon, cut), np.split(lat, cut)):
            if seg_lon.size >= 2:
                parts.append(np.column_stack([seg_lon, seg_lat]))
                parts.append(np.array([[np.nan, np.nan]]))
    return np.concatenate(parts) if parts else np.zeros((0, 2))


def natural_earth_lines(kind: str, scale: str = "50m") -> list[np.ndarray]:
    """Natural Earth coastlines (``coast``) or land borders (``borders``) as lon/lat polylines."""
    import cartopy.feature as cfeature

    feat = (cfeature.NaturalEarthFeature("physical", "coastline", scale) if kind == "coast" else
            cfeature.NaturalEarthFeature("cultural", "admin_0_boundary_lines_land", scale))
    out = []
    for geom in feat.geometries():
        for line in getattr(geom, "geoms", [geom]):
            out.append(np.asarray(line.coords)[:, :2])
    return out


def load_geography(cache: Path | None, scale: str = "50m") -> dict[str, list[np.ndarray]]:
    """Coastline and border polylines: from an ``.npz`` cache if given and present, else cartopy.

    With a cache path that does not exist yet, the cartopy lines are written to it, so a machine
    with internet can prepare the file for a compute node without it.
    """
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        return {k: [z[n] for n in sorted(z.files) if n.startswith(k + "_")] for k in ("coast", "borders")}
    geo = {"coast": natural_earth_lines("coast", scale), "borders": natural_earth_lines("borders", scale)}
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{f"{k}_{i:05d}": a.astype(np.float32)
                                      for k, v in geo.items() for i, a in enumerate(v)})
    return geo


@dataclass
class FrameComposer:
    layout: Layout
    lat: np.ndarray
    lon: np.ndarray
    palette: object
    panel_titles: list[str] = field(default_factory=list)
    geography: dict | None = None
    coast_px: float = 0.8
    border_px: float = 0.5
    coast_rgb: str = "#4d4d4d"
    border_rgb: str = "#a0a0a0"

    def __post_init__(self):
        import matplotlib

        matplotlib.use("Agg")
        self._base = self._draw_base()
        self._overlay = self._draw_overlay()

    # -- static layers -------------------------------------------------------------------
    def _figure(self, transparent: bool = False):
        import matplotlib.pyplot as plt

        L = self.layout
        fig = plt.figure(figsize=(L.width / DPI, L.height / DPI), dpi=DPI)
        fig.patch.set_alpha(0.0 if transparent else 1.0)
        fig.patch.set_facecolor("white")
        return fig

    def _draw_base(self) -> np.ndarray:
        import matplotlib.pyplot as plt
        from matplotlib.colorbar import ColorbarBase

        L = self.layout
        fig = self._figure()
        font_pt = L.font_px * PT_PER_PX
        for rect, title in zip(L.panels, self.panel_titles or [None] * len(L.panels)):
            if title:
                x0, y0, w, _ = rect
                fig.text((x0 + w / 2) / L.width, 1.0 - (y0 - 0.35 * L.label_h) / L.height, title,
                         ha="center", va="bottom", fontsize=font_pt * 1.05, fontweight="bold")
        cax = fig.add_axes(_axes_rect(L.width, L.height, L.colorbar))
        cmap, norm, bounds = self.palette.colorbar_cmap_norm()
        kw = dict(cmap=cmap, norm=norm, orientation="horizontal", extend=self.palette.extend)
        if bounds is not None:
            kw.update(boundaries=bounds, ticks=bounds, spacing="uniform")
            if self.palette.extend == "max":
                kw["extend"] = "both"  # white "under" triangle, like the reference frame
        cb = ColorbarBase(cax, **kw)
        if bounds is not None:
            cb.set_ticklabels([f"{b:g}" for b in bounds])
        cb.ax.tick_params(labelsize=font_pt * 0.85, length=0.3 * L.font_px * PT_PER_PX * 2)
        cb.outline.set_linewidth(0.6)
        cb.set_label(self.palette.label, fontsize=font_pt * 0.95)
        img = _canvas_rgba(fig)[..., :3]
        plt.close(fig)
        return img

    def _draw_overlay(self) -> np.ndarray | None:
        if not self.geography:
            return None
        import matplotlib.pyplot as plt

        L = self.layout
        fig = self._figure(transparent=True)
        res_lon = float(np.abs(np.diff(self.lon[:2]))[0]) if self.lon.size > 1 else 1.0
        res_lat = float(np.abs(np.diff(self.lat[:2]))[0]) if self.lat.size > 1 else 1.0
        west = float(self.lon[0]) - res_lon / 2
        east = float(self.lon[-1]) + res_lon / 2
        south = float(self.lat[-1]) - res_lat / 2
        north = float(self.lat[0]) + res_lat / 2
        borders = wrap_lines(self.geography.get("borders", []), west)
        coast = wrap_lines(self.geography.get("coast", []), west)
        for rect in L.panels:
            ax = fig.add_axes(_axes_rect(L.width, L.height, rect))
            ax.set_axis_off()
            ax.patch.set_alpha(0.0)
            if borders.size and self.border_px > 0:
                ax.plot(borders[:, 0], borders[:, 1], color=self.border_rgb,
                        lw=self.border_px * PT_PER_PX, solid_capstyle="round", antialiased=True)
            if coast.size and self.coast_px > 0:
                ax.plot(coast[:, 0], coast[:, 1], color=self.coast_rgb,
                        lw=self.coast_px * PT_PER_PX, solid_capstyle="round", antialiased=True)
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
        rgba = _canvas_rgba(fig)
        plt.close(fig)
        return rgba

    # -- per frame -----------------------------------------------------------------------
    def _title_strip(self, title: str) -> np.ndarray:
        import matplotlib.pyplot as plt

        L = self.layout
        _, _, w, h = L.header
        fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
        fig.patch.set_facecolor("white")
        x_left = L.panels[0][0] / w
        fig.text(x_left, 0.5, title, ha="left", va="center", fontsize=L.font_px * PT_PER_PX)
        img = _canvas_rgba(fig)[..., :3]
        plt.close(fig)
        return img

    def compose(self, fields: list[np.ndarray], title: str) -> np.ndarray:
        """RGB uint8 frame (height, width, 3) from one regular-grid field per panel."""
        L = self.layout
        frame = self._base.copy()
        for rect, values in zip(L.panels, fields):
            x0, y0, w, h = rect
            frame[y0:y0 + h, x0:x0 + w] = self.palette.rgb(values)
        if self._overlay is not None:
            a = self._overlay[..., 3:4].astype(np.float32) / 255.0
            frame = (frame * (1.0 - a) + self._overlay[..., :3] * a + 0.5).astype(np.uint8)
        _, _, w, h = L.header
        frame[0:h, 0:w] = self._title_strip(title)
        return frame


def find_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on the host
        raise RuntimeError("no ffmpeg: load a module, pip install imageio-ffmpeg, or pass --ffmpeg") from exc


class Mp4Writer:
    """Pipe RGB frames to ffmpeg (H.264, constant quality)."""

    def __init__(self, path: Path, width: int, height: int, fps: float, *, crf: int = 16,
                 preset: str = "slow", chroma: str = "yuv420p", ffmpeg: str | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [find_ffmpeg(ffmpeg), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps}",
               "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-pix_fmt", chroma, "-movflags", "+faststart", str(path)]
        self.size = (height, width)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        if frame.shape[:2] != self.size:
            raise ValueError(f"frame {frame.shape[:2]} != video {self.size}")
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with status {self.proc.returncode}")
