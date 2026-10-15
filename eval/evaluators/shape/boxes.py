"""Where the shape instrument runs: the named boxes, the point selection and the per-box caches.

The instrument of 2026-09-16/17 (instrument.py, fullrung.py, verbatim) was built for one box,
10-40 N, 100-58 W, cut out of a global file by matching a probe file that holds exactly that
box. Its graph, kernels and open-ocean masks are cached on disk for that box only. This module
lets the same statistics run on any box:

- the box is cut out of each prediction file by its own ``lat_hres``/``lon_hres`` (any
  longitude convention), so a Europe-box (E2) file and a global file give the same points;
- the geometry and mask caches get one file per box and grid
  (``shape_<tag>_n<points>_geometry.npz`` and ``..._masks.npz`` under ``cache_dir``), so a
  cache built for one box is never loaded for another;
- the masks of a new box are built once by ``instrument.build_box_masks`` from the global
  O1280 land-sea mask and orography (Atos paths, overridable), then reused.

The default box ``tc_atlantic`` keeps the probe-file path untouched, so the numbers of
September stay reproducible. Numbers from two boxes are never compared: a read runs the
model and its reference on the same box.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc
import numpy as np

from eval.paths import resolve_eval_root
from eval.shared.regions import E2, NORTH_ATLANTIC, TC_ATLANTIC


@dataclass(frozen=True)
class Box:
    tag: str
    lat_min: float
    lat_max: float
    lon_min: float   # west edge, degrees east in [-180, 180)
    lon_max: float   # east edge

    def as_list(self) -> list[float]:
        return [self.lat_min, self.lat_max, self.lon_min, self.lon_max]


# The legacy box of the epic (probe file) and the open-ocean boxes that the Europe-box (E2)
# domain 10-70 N, 80 W-40 E holds. ``north_atlantic`` is inside both E2 and global files,
# so it is the box that makes a global run and an E2 run comparable.
BOXES = {
    "tc_atlantic": Box("tc_atlantic", *TC_ATLANTIC),
    "north_atlantic": Box("north_atlantic", *NORTH_ATLANTIC),
    "e2": Box("e2", *E2),
}
LEGACY_BOX = "tc_atlantic"


def resolve_box(spec) -> Box:
    """A box from a name in BOXES, or from [lat_min, lat_max, lon_min, lon_max]."""
    if spec is None or spec == "":
        return BOXES[LEGACY_BOX]
    if isinstance(spec, str):
        if spec in BOXES:
            return BOXES[spec]
        parts = [p for p in spec.replace(",", " ").split() if p]
        if len(parts) != 4:
            raise ValueError(f"shape box {spec!r}: a name in {sorted(BOXES)} or "
                             "'lat_min,lat_max,lon_min,lon_max'")
        spec = parts
    if isinstance(spec, dict):
        spec = [spec["lat_min"], spec["lat_max"], spec["lon_min"], spec["lon_max"]]
    la0, la1, lo0, lo1 = (float(v) for v in spec)
    if not la0 < la1:
        raise ValueError(f"shape box: lat_min {la0} must be below lat_max {la1}")
    norm = lambda v: (v + 180.0) % 360.0 - 180.0  # noqa: E731
    lo0, lo1 = norm(lo0), norm(lo1)
    for b in BOXES.values():
        if np.allclose(b.as_list(), [la0, la1, lo0, lo1]):
            return b
    tag = f"{la0:g}_{la1:g}N_{lo0:g}_{lo1:g}E".replace("-", "m")
    return Box(tag, la0, la1, lo0, lo1)


def is_legacy(box: Box) -> bool:
    return box.tag == LEGACY_BOX


def contiguous_lon(lon, west: float) -> np.ndarray:
    """Longitudes mapped to [west, west + 360), so a box never wraps inside its own array
    (the interior margin and the regrid of the instrument use lon.min() and lon.max())."""
    return west + np.mod(np.asarray(lon, dtype=np.float64) - west, 360.0)


def select_box(path, box: Box):
    """Indices of ``box`` in the points of one prediction file, with their lat/lon.

    Returns (sel, lat, lon, note); lon is contiguous from the box's west edge.
    """
    with nc.Dataset(str(path)) as d:
        glat = np.asarray(d["lat_hres"][:], dtype=np.float64)
        glon = np.asarray(d["lon_hres"][:], dtype=np.float64)
    # the bbox rule of eval.predict.local_scope (not imported: that package needs torch)
    east = contiguous_lon(glon, box.lon_min) - box.lon_min
    sel = np.flatnonzero((glat >= box.lat_min) & (glat <= box.lat_max) & (east <= _width(box)))
    if sel.size == 0:
        raise ValueError(f"shape box {box.tag} {box.as_list()} holds no point of {path} "
                         f"(file spans lat {glat.min():.1f}..{glat.max():.1f}, "
                         f"lon {glon.min():.1f}..{glon.max():.1f})")
    lat = glat[sel]
    lon = contiguous_lon(glon[sel], box.lon_min)
    # a box that the file only partly covers would give a truncated box with a different
    # interior, silently; refuse it. The tolerance is 0.2 degrees or one grid row, whichever
    # is larger: on O320 the rows are 0.28 degrees apart, so the first row inside a box edge
    # can sit 0.27 degrees in (35.27 N for the north_atlantic box) on a box the file holds.
    tol = max(0.2, _row_spacing_deg(glat))
    if (lat.min() > box.lat_min + tol or lat.max() < box.lat_max - tol
            or lon.min() > box.lon_min + tol or lon.max() < box.lon_min + _width(box) - tol):
        raise ValueError(f"shape box {box.tag} {box.as_list()} is not fully inside {path}: "
                         f"the file covers lat {lat.min():.2f}..{lat.max():.2f}, "
                         f"lon {lon.min():.2f}..{lon.max():.2f} of it")
    return sel, lat, lon, f"box {box.tag} {box.as_list()}: {sel.size} of {glat.size} points"


def _row_spacing_deg(glat) -> float:
    """Median spacing of the distinct latitudes of a grid (0.07 on O1280, 0.28 on O320)."""
    rows = np.unique(np.round(np.asarray(glat, dtype=np.float64), 6))
    return float(np.median(np.diff(rows))) if rows.size > 1 else 0.0


def _width(box: Box) -> float:
    w = (box.lon_max - box.lon_min) % 360.0
    return 360.0 if w == 0 else w


def default_cache_dir() -> Path:
    return resolve_eval_root() / "_cache" / "shape"


def cache_paths(box: Box, cache_dir=None, n_points: int | None = None) -> dict[str, str]:
    """Cache files of one box (and one grid: the point count is in the name, so an O1280 and
    an O2560 read of the same box never share a cache)."""
    root = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    stem = f"shape_{box.tag}" + (f"_n{int(n_points)}" if n_points else "")
    return {"geom_cache": str(root / f"{stem}_geometry.npz"),
            "mask_cache": str(root / f"{stem}_masks.npz")}


def check_masks(ap, box: Box) -> None:
    """A mask cache must hold one value per box point (a cache of another box would not)."""
    n = {k: int(np.asarray(v).size) for k, v in ap.masks.items()}
    bad = {k: v for k, v in n.items() if v != ap.n}
    if bad:
        raise ValueError(f"shape box {box.tag}: mask cache {ap.maskcache} has {bad} points, the box "
                         f"has {ap.n}; it was built for another box. Move it aside and rerun.")
    n_open = int(ap.masks["open_ocean_interior"].sum())
    if n_open == 0:
        raise ValueError(f"shape box {box.tag}: no open-ocean interior point; the statistics "
                         "need open ocean (pick an ocean box)")
