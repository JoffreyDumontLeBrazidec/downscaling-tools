"""Numpy geometry shared by the interp tools (single source of truth).

Previously haversine / lon-normalization / storm-center detection were
re-implemented in activation_patching, integrated_gradients and regions.
The torch variants used inside differentiable functionals stay in
:mod:`interp.core.regions`; everything mask/probe related (numpy, no grad)
lives here.
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

# NW-Atlantic search window for TC-center auto-detection (lat0,lat1,lon0,lon1;
# lon 0..360). Covers Franklin 2023 (~-70 W = 290 E) and excludes Idalia's Gulf
# landfall (~-83 W = 277 E).
DEFAULT_AUTO_WINDOW = (10.0, 45.0, 280.0, 320.0)

# Wider N. Atlantic TC box (for "deepest Atlantic low" detection).
ATLANTIC_WINDOW = (8.0, 45.0, 270.0, 350.0)


def norm_lon(lon):
    """Normalize longitudes to [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def haversine_km(lat0, lon0, lat, lon):
    """Great-circle distance (km) from a single point to arrays lat/lon (degrees)."""
    dlon_deg = (lon - lon0 + 180.0) % 360.0 - 180.0
    lat0r, latr = np.radians(lat0), np.radians(lat)
    dlatr, dlonr = np.radians(lat - lat0), np.radians(dlon_deg)
    a = np.sin(dlatr / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlonr / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def box_mask_km(lat, lon, lat0, lon0, radius_km):
    """Boolean mask over the grid for cells within radius_km of (lat0, lon0)."""
    d = haversine_km(lat0, norm_lon(lon0), norm_lon(lat), norm_lon(lon))
    return d <= float(radius_km)


def great_circle_mask_deg(lat, lon, clat, clon, radius_deg):
    """Boolean mask of nodes within `radius_deg` great-circle degrees of center."""
    la1, lo1 = np.radians(lat), np.radians(lon)
    la2, lo2 = np.radians(clat), np.radians(clon)
    cosd = np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2)
    ang = np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))
    return ang <= radius_deg


def detect_min_center(field, lat, lon, window=DEFAULT_AUTO_WINDOW):
    """Locate a storm center as the argmin of `field` (typically msl) within a
    lat/lon window (lat0, lat1, lon0, lon1; lon in 0..360).

    Returns (lat, lon_0360).
    """
    la0, la1, lo0, lo1 = window
    lonn = np.asarray(lon) % 360.0
    lat = np.asarray(lat)
    sel = (lat >= la0) & (lat <= la1) & (lonn >= lo0) & (lonn <= lo1)
    if not sel.any():
        raise ValueError(f"auto-detect window {window} contains no grid cells")
    idx = int(np.argmin(np.where(sel, field, np.inf)))
    return float(lat[idx]), float(lonn[idx])


def parse_boxes(box_specs, lat, lon, msl_field=None, window=None):
    """Parse box specs into {name: dict(lat, lon, radius_km, mask, n_cells)}.

    Spec form: "name:lat,lon,radius_km" (explicit) or "name:auto[,radius_km]"
    (storm-centered: center = argmin(msl) inside `window`, needs `msl_field`).
    """
    boxes = {}
    for spec in box_specs:
        name, _, rest = spec.partition(":")
        parts = [p.strip() for p in rest.split(",")]
        if parts and parts[0].lower() == "auto":
            if msl_field is None:
                raise ValueError(f"box {name!r} uses 'auto' but no msl field available "
                                 "(auto-detect needs bundle target data)")
            radius_km = float(parts[1]) if len(parts) > 1 else 500.0
            lat0, lon0 = detect_min_center(msl_field, lat, lon,
                                           window or DEFAULT_AUTO_WINDOW)
            LOGGER.info("box %s: auto-detected storm center (%.2f,%.2f)", name, lat0, lon0)
        elif len(parts) == 3:
            lat0, lon0, radius_km = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"box spec {spec!r} must be 'name:lat,lon,radiuskm' or 'name:auto'")
        mask = box_mask_km(lat, lon, lat0, lon0, radius_km)
        n = int(mask.sum())
        if n == 0:
            raise ValueError(f"box {name} ({lat0},{lon0},{radius_km}km) has 0 grid cells")
        boxes[name] = {"lat": lat0, "lon": lon0 % 360.0, "radius_km": radius_km,
                       "mask": mask, "n_cells": n}
        LOGGER.info("box %s: center=(%.2f,%.2f) R=%.0fkm -> %d cells",
                    name, lat0, lon0 % 360.0, radius_km, n)
    return boxes


def node_latlon_deg(graph_data, name):
    """Return (lat_deg, lon_deg in [-180,180]) for a graph node set.

    Anemoi stores node coords in graph[name].x as radians [lat, lon].
    """
    xy = graph_data[name].x.detach().cpu().numpy()
    lat = np.degrees(xy[:, 0])
    lon = norm_lon(np.degrees(xy[:, 1]))
    return lat, lon
