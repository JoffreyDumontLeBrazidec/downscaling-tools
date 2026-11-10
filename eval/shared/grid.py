"""Shared spatial utilities used by multiple evaluators (TC, region_plot)."""
from __future__ import annotations

import numpy as np


def normalize_lon(lon: np.ndarray) -> np.ndarray:
    """Normalize longitude values to the range [-180, 180)."""
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def point_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    south: float,
    north: float,
    west: float,
    east: float,
) -> np.ndarray:
    """Boolean mask selecting points inside a lat/lon bounding box.

    Handles dateline-crossing boxes (where east < west after normalization).
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = normalize_lon(np.asarray(lon, dtype=np.float64))
    west_n = normalize_lon(np.asarray([west]))[0]
    east_n = normalize_lon(np.asarray([east]))[0]

    lat_mask = (lat_arr >= south) & (lat_arr <= north)
    if east_n >= west_n:
        lon_mask = (lon_arr >= west_n) & (lon_arr <= east_n)
    else:
        lon_mask = (lon_arr >= west_n) | (lon_arr <= east_n)
    return lat_mask & lon_mask


def unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Unit vectors on the sphere, shape (n, 3)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    c = np.cos(lat)
    return np.column_stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)])


def global_point_index(
    lat_sub: np.ndarray,
    lon_sub: np.ndarray,
    lat_global: np.ndarray,
    lon_global: np.ndarray,
    *,
    tol_km: float = 0.5,
) -> np.ndarray:
    """Index into the global grid of every point of a subset grid.

    A regional (cut-graph) prediction file holds a subset of the global output grid
    in an order of its own; this maps each of its points to the global point at the
    same location. Every subset point must coincide with a global point to within
    ``tol_km`` and no two subset points may map to the same global point.
    """
    from scipy.spatial import cKDTree

    earth_km = 6371.0088
    tree = cKDTree(unit_vectors(lat_global, lon_global))
    chord, idx = tree.query(unit_vectors(lat_sub, lon_sub), k=1, workers=-1)
    dist_km = earth_km * 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))
    bad = int((dist_km > tol_km).sum())
    if bad:
        raise ValueError(
            f"{bad} of {lat_sub.size} subset points are farther than {tol_km} km from any "
            f"global point (max {float(dist_km.max()):.2f} km): not a subset of this grid"
        )
    if np.unique(idx).size != idx.size:
        raise ValueError("two subset points map to the same global point")
    return np.asarray(idx, dtype=np.int64)
