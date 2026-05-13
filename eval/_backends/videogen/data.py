"""Frame stitching, variable slicing, bbox masking, regridding, TC tracking.

Stable layer — most iteration happens above (layouts, scenes).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from .config import BBox, SceneConfig


# ---------------------------------------------------------------------------
# Frame stitching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    valid_time: datetime
    init_date: str   # YYYYMMDD
    step: int        # hours
    nc_path: Path

    def label(self) -> str:
        return (
            f"Valid {self.valid_time:%Y-%m-%d %H}Z   "
            f"(init {self.init_date[:4]}-{self.init_date[4:6]}-{self.init_date[6:]} "
            f"+ {self.step:03d}h)"
        )


def build_frame_list(scene: SceneConfig) -> list[Frame]:
    """Pick one (init, step) per valid time using the freshest-init-≤-vt rule.

    Files that don't exist on disk are silently skipped (we try the next-older
    init for the same valid time).
    """
    by_vt: dict[datetime, list[tuple[str, int]]] = {}
    for init in scene.inits:
        init_dt = datetime.strptime(init, "%Y%m%d")
        for step in scene.steps:
            vt = init_dt + timedelta(hours=step)
            by_vt.setdefault(vt, []).append((init, step))

    frames: list[Frame] = []
    for vt in sorted(by_vt):
        for init, step in sorted(by_vt[vt], key=lambda kv: kv[0], reverse=True):
            nc_path = scene.predictions_dir / f"predictions_{init}_step{step:03d}.nc"
            if nc_path.exists():
                frames.append(Frame(valid_time=vt, init_date=init, step=step, nc_path=nc_path))
                break
    return frames


# ---------------------------------------------------------------------------
# Variable slicing
# ---------------------------------------------------------------------------

def load_var_slice(
    ds: xr.Dataset,
    grid: str,           # "lres" or "hres"
    var: str,            # weather_state coord value, or synthetic "wind"
    ensemble_member: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lon, lat, values)`` for one variable on one grid.

    Handles the synthetic ``wind`` variable: ``sqrt(10u² + 10v²)``.
    """
    data_var = "x" if grid == "lres" else "y_pred"
    da = ds[data_var].isel(sample=0, ensemble_member=ensemble_member)
    if var == "wind":
        u = da.sel(weather_state="10u").values.astype(np.float32)
        v = da.sel(weather_state="10v").values.astype(np.float32)
        values = np.hypot(u, v)
    else:
        values = da.sel(weather_state=var).values.astype(np.float32)
    lon = ds[f"lon_{grid}"].values.astype(np.float32)
    lat = ds[f"lat_{grid}"].values.astype(np.float32)
    return lon, lat, values


def bbox_mask(lon: np.ndarray, lat: np.ndarray, bbox: BBox) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = bbox
    return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)


# ---------------------------------------------------------------------------
# TC tracking
# ---------------------------------------------------------------------------

def find_msl_min(
    ds: xr.Dataset, search_bbox: BBox, ensemble_member: int = 0,
) -> tuple[float, float]:
    """``(lon, lat)`` of the lres MSL minimum inside ``search_bbox``.

    Falls back to the search-box centre if no lres points lie inside it.
    """
    lon, lat, msl = load_var_slice(ds, "lres", "msl", ensemble_member=ensemble_member)
    m = bbox_mask(lon, lat, search_bbox)
    sub_lon, sub_lat, sub_msl = lon[m], lat[m], msl[m]
    if sub_msl.size == 0:
        return ((search_bbox[0] + search_bbox[1]) / 2,
                (search_bbox[2] + search_bbox[3]) / 2)
    i = int(np.argmin(sub_msl))
    return float(sub_lon[i]), float(sub_lat[i])


def resolve_inset_bbox(ds: xr.Dataset, scene: SceneConfig) -> BBox:
    """Inset bbox for a single frame (fixed or tracked)."""
    if scene.inset_kind == "fixed":
        if scene.inset_bbox is None:
            raise ValueError(f"scene {scene.name!r}: inset_kind='fixed' needs inset_bbox")
        return scene.inset_bbox
    if scene.inset_kind == "track_msl_min":
        if scene.inset_search_bbox is None:
            raise ValueError(
                f"scene {scene.name!r}: inset_kind='track_msl_min' needs inset_search_bbox"
            )
        cx, cy = find_msl_min(ds, scene.inset_search_bbox,
                              ensemble_member=scene.ensemble_member)
        half = scene.inset_half_deg
        return (cx - half, cx + half, cy - half, cy + half)
    raise ValueError(f"unknown inset_kind: {scene.inset_kind!r}")


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

def nearest_regrid(
    lon: np.ndarray, lat: np.ndarray, vals: np.ndarray,
    bbox: BBox, resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """cKDTree nearest-neighbour regrid onto a regular lon/lat raster.

    Returns ``(lon_edges, lat_edges, lon_centers, lat_centers, values_2d)``.
    Pass edges to ``pcolormesh(shading="flat")`` and centres to ``contour``.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    nx = max(int(round((lon_max - lon_min) / resolution_deg)), 2)
    ny = max(int(round((lat_max - lat_min) / resolution_deg)), 2)
    lon_centers = np.linspace(lon_min, lon_max, nx)
    lat_centers = np.linspace(lat_min, lat_max, ny)
    grid_lon, grid_lat = np.meshgrid(lon_centers, lat_centers)
    tree = cKDTree(np.column_stack([lon, lat]))
    _, idx = tree.query(np.column_stack([grid_lon.ravel(), grid_lat.ravel()]), k=1)
    values_2d = vals[idx].reshape(grid_lon.shape)
    dx = (lon_max - lon_min) / (nx - 1)
    dy = (lat_max - lat_min) / (ny - 1)
    lon_edges = np.linspace(lon_min - dx / 2, lon_max + dx / 2, nx + 1)
    lat_edges = np.linspace(lat_min - dy / 2, lat_max + dy / 2, ny + 1)
    return lon_edges, lat_edges, lon_centers, lat_centers, values_2d
