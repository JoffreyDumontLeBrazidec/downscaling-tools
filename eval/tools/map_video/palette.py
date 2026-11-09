"""Colour tables for map videos: discrete precipitation classes and continuous house maps.

Values are coloured with a lookup table, so a frame is coloured exactly (no resampling of an
image) and fast. Precipitation is drawn in discrete classes with white below the first level:

    0.1  0.25  0.5  1  2  3  5  7.5  10  15  20  30  50 mm, and an arrow above 50 mm.

Two palettes exist for these classes:

* ``rain`` (default): the house colour map for precipitation, cmocean ``rain`` (rule 25,
  ``eval.plotting.variables.precip_cmap``), sampled once per class.
* ``reference``: the blue-green-yellow-red-purple classes of the corrector video frame that
  was shown on 9 Nov 2026 (colours measured on that frame's colour bar).

Other variables use their house colour map from ``eval.plotting.variables`` on a fixed range.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TP_LEVELS_MM = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0)

# Measured on the colour bar of the reference frame (9 Nov 2026), one colour per class, then "over".
REFERENCE_TP_RGB = (
    (174, 208, 245), (124, 170, 233), (79, 121, 206), (98, 150, 104), (134, 184, 109),
    (203, 216, 120), (246, 225, 130), (235, 188, 102), (223, 141, 77), (204, 90, 61),
    (149, 65, 58), (186, 88, 160),
)
REFERENCE_TP_OVER = (118, 56, 127)

NAN_RGB = (225, 225, 225)  # outside the data (crops), light grey


@dataclass
class Palette:
    """A value -> RGB mapping. ``levels`` set: discrete classes; otherwise continuous on [vmin, vmax]."""

    lut: np.ndarray                 # (n, 3) uint8
    levels: np.ndarray | None = None  # class edges (discrete): below levels[0] -> lut[0]
    vmin: float = 0.0
    vmax: float = 1.0
    extend: str = "max"
    label: str = ""

    @property
    def discrete(self) -> bool:
        return self.levels is not None

    def rgb(self, values: np.ndarray) -> np.ndarray:
        """uint8 RGB image of ``values``; NaN is light grey."""
        v = np.asarray(values)
        if self.discrete:
            k = np.searchsorted(self.levels, v, side="right")   # 0 below the first level
        else:
            n = self.lut.shape[0]
            k = np.clip(((v - self.vmin) / (self.vmax - self.vmin) * n).astype(np.int64), 0, n - 1)
        out = self.lut[np.where(np.isfinite(v), k, 0)]
        nan = ~np.isfinite(v)
        if nan.any():
            out[nan] = NAN_RGB
        return out

    def colorbar_cmap_norm(self):
        """(cmap, norm, boundaries or None) to draw this palette's colour bar with Matplotlib."""
        from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

        if self.discrete:
            inner = self.lut[1:-1] / 255.0 if self.extend == "max" else self.lut[1:] / 255.0
            cmap = ListedColormap(inner)
            cmap.set_under(self.lut[0] / 255.0)
            if self.extend == "max":
                cmap.set_over(self.lut[-1] / 255.0)
            return cmap, BoundaryNorm(self.levels, cmap.N), self.levels
        cmap = ListedColormap(self.lut / 255.0)
        return cmap, Normalize(self.vmin, self.vmax), None


def tp_palette(style: str = "rain", levels=TP_LEVELS_MM, label: str = "precipitation (mm)") -> Palette:
    """Discrete precipitation palette: white below ``levels[0]``, one colour per class, an over colour."""
    levels = np.asarray(levels, dtype=np.float64)
    n_classes = levels.size - 1
    if style == "reference":
        if n_classes != len(REFERENCE_TP_RGB):
            raise ValueError("the reference palette has 12 classes; use the default levels")
        rows = [(255, 255, 255), *REFERENCE_TP_RGB, REFERENCE_TP_OVER]
    elif style == "rain":
        from eval.plotting.variables import precip_cmap

        cmap = precip_cmap()
        # skip the near-white start of cmocean rain so the first class stands out from "no rain"
        pos = np.linspace(0.12, 0.92, n_classes)
        rows = [(255, 255, 255)]
        rows += [tuple(int(round(255 * c)) for c in cmap(p)[:3]) for p in pos]
        rows.append(tuple(int(round(255 * c)) for c in cmap(1.0)[:3]))
    else:
        raise ValueError(f"unknown precipitation palette {style!r} (rain or reference)")
    return Palette(lut=np.asarray(rows, dtype=np.uint8), levels=levels, extend="max", label=label)


def continuous_palette(variable: str, vmin: float | None = None, vmax: float | None = None,
                       n: int = 256) -> Palette:
    """House colour map of ``variable`` (``eval.plotting.variables``) on a fixed range."""
    from eval.plotting.variables import axis_label, variable_spec

    spec = variable_spec(variable)
    lo, hi = spec.default_range or (vmin, vmax)
    lo = lo if vmin is None else vmin
    hi = hi if vmax is None else vmax
    if lo is None or hi is None:
        raise ValueError(f"{variable}: give --vmin and --vmax (no default range in the variable table)")
    lo, hi = spec.centred_range(lo, hi)
    cmap = spec.field_cmap()
    if isinstance(cmap, str):
        import matplotlib

        cmap = matplotlib.colormaps[cmap]
    lut = (np.asarray(cmap(np.linspace(0, 1, n)))[:, :3] * 255).round().astype(np.uint8)
    return Palette(lut=lut, vmin=float(lo), vmax=float(hi), extend=spec.extend, label=axis_label(variable))
