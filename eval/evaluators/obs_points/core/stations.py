"""Station handling shared by every evaluator that scores against station observations
(obs_points, and the station part of precip_read): the cache files, the nearest-grid-point
match, regions, height classes and the optional station-density weights. Free of STVL, so it
runs on any cluster.

The cache is the directory ``eval.cli obs fetch`` writes (see ``core/fetch.py``): one
``obs_<var>.csv`` per variable (valid YYYYMMDDHH, lat, lon, stnid, elevation, obs, h_o1280,
h_o320), ``hres_<var>.csv`` where STVL holds the operational HRES at the stations, and
``cache_meta.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R_EARTH_KM = 6371.0

# quaver's toss.yaml hard limits; tp in mm per accumulation window.
LIMITS = {"2t": (173.0, 333.0), "10ff": (0.0, 100.0), "msl": (80000.0, 110000.0), "tp": (0.0, 300.0),
          "tp24": (0.0, 300.0)}


def xyz(lat, lon):
    """Unit-sphere coordinates, so that nearest-neighbour search ignores longitude wrapping."""
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def wrap_lon(lon):
    lon = np.asarray(lon, np.float64)
    return np.where(lon > 180.0, lon - 360.0, lon)


class StationMatcher:
    """Nearest grid point of each station on one fixed grid (quaver's ``nearestpoint``)."""

    def __init__(self, lat, lon):
        from scipy.spatial import cKDTree

        self.lat, self.lon = np.asarray(lat), wrap_lon(lon)
        self.tree = cKDTree(xyz(self.lat, self.lon))

    def same_grid(self, lat, lon) -> bool:
        return self.lat.size == np.size(lat) and np.allclose(self.lat, lat) and np.allclose(self.lon, wrap_lon(lon))

    def match(self, slat, slon, max_km: float):
        """Index of the nearest grid point and a mask of the stations within ``max_km`` of it."""
        dist, idx = self.tree.query(xyz(np.asarray(slat), np.asarray(slon)))
        return idx, dist * R_EARTH_KM <= max_km

    def discs(self, idx, radius_km: float):
        """For each matched grid point, the grid points within ``radius_km`` of it (disc means)."""
        return self.tree.query_ball_point(xyz(self.lat[idx], self.lon[idx]), radius_km / R_EARTH_KM)


def load_obs(path, obs_max: float = 300.0):
    """One observation CSV with values outside [0, obs_max] dropped; returns (frame, n dropped).
    Used by precip_read for its single tp file."""
    import pandas as pd

    obs = pd.read_csv(path, dtype={"valid": str})
    n_bad = int(((obs.obs < 0) | (obs.obs > obs_max)).sum())
    return obs[(obs.obs >= 0) & (obs.obs <= obs_max)], n_bad


def resolve_cache(cache) -> Path:
    """The cache directory: a path, or a list or mapping of candidate paths (one per cluster),
    of which the first that exists on this machine is used."""
    if cache is None:
        raise ValueError("obs_points: the lane block needs 'cache' (the directory written by `eval.cli obs fetch`)")
    cands = list(cache.values()) if isinstance(cache, dict) else ([cache] if isinstance(cache, (str, Path)) else list(cache))
    for c in cands:
        p = Path(str(c)).expanduser()
        if p.is_dir():
            return p
    raise FileNotFoundError(f"obs_points: no observation cache found among {cands}")


def load_cache(cache_dir, variables) -> tuple[dict, dict, dict]:
    """(obs per var, hres per var, cache_meta) from a cache directory; quaver hard limits applied again."""
    import pandas as pd

    cache_dir = Path(cache_dir)
    obs, hres = {}, {}
    for v in variables:
        p = cache_dir / f"obs_{v}.csv"
        if not p.exists():
            continue
        o = pd.read_csv(p, dtype={"valid": str, "stnid": str})
        lo, hi = LIMITS.get(v, (-np.inf, np.inf))
        obs[v] = o[(o.obs >= lo) & (o.obs <= hi)]
        h = cache_dir / f"hres_{v}.csv"
        if h.exists():
            hres[v] = pd.read_csv(h, dtype={"date": str, "stnid": str})
    meta_path = cache_dir / "cache_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return obs, hres, meta


def load_seeps_clim(cache_dir):
    """The SEEPS station climatology of the cache (``obs fetch --seeps``), or None."""
    import pandas as pd

    p = Path(cache_dir) / "seeps_clim_tp24.csv"
    return pd.read_csv(p, dtype={"valid": str, "stnid": str}) if p.exists() else None


def region_masks(lat, lon, regions: dict) -> dict:
    """name -> boolean mask; a region is None (everything) or [lat0, lat1, lon0, lon1], half-open."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    return {k: np.ones(lat.size, bool) if b is None else (lat >= b[0]) & (lat < b[1]) & (lon >= b[2]) & (lon < b[3])
            for k, b in regions.items()}


def height_masks(dz_abs, classes: dict) -> dict:
    """name -> mask on |model orography - station height|, classes given as [lo, hi) in metres."""
    dz_abs = np.asarray(dz_abs)
    return {k: (dz_abs >= float(lo)) & (dz_abs < float(hi)) for k, (lo, hi) in classes.items()}


def density_weights(lat, lon) -> np.ndarray:
    """quaver's station-density weights (Gaussian kernel, scale 0.75 degree), summing to one.
    The implementation is the one validated against vtb in eval/tools/obs_crps."""
    from eval.tools.obs_crps.obs_crps_compute import station_density_weights

    return station_density_weights(np.asarray(lat), np.asarray(lon))
