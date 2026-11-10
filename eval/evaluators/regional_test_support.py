"""Synthetic fixtures shared by the regional-mode tests of texture and wind_extremes.

A 1-degree global grid (64,800 points), a 4-degree coarse grid, inverse-distance
``up`` (4 coarse neighbours) and nearest-point ``down`` matrices, a forcings zarr
with lsm/z, and prediction files in the layout predict.py writes: a global file
and its cut to a lat/lon box, the cut carrying ``x_interp = (up @ x)[box]``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BOX = (10.0, 40.0, -100.0, -58.0)   # lat_min, lat_max, lon_min, lon_max (Franklin-Idalia)
STATES = ["10u", "10v", "2t"]


def _grid(step: float):
    lat1 = np.arange(90.0 - step / 2, -90.0, -step)
    lon1 = np.arange(step / 2, 360.0, step)
    lon, lat = np.meshgrid(lon1, lat1)
    return lat.ravel(), lon.ravel()


def _unit(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def build_matrices(lat_h, lon_h, lat_c, lon_c):
    import scipy.sparse as sps
    from scipy.spatial import cKDTree

    tree_c = cKDTree(_unit(lat_c, lon_c))
    d, i = tree_c.query(_unit(lat_h, lon_h), k=4)
    w = 1.0 / np.maximum(d, 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    rows = np.repeat(np.arange(lat_h.size), 4)
    up = sps.csr_matrix((w.ravel(), (rows, i.ravel())), shape=(lat_h.size, lat_c.size))
    d1, i1 = tree_c.query(_unit(lat_h, lon_h), k=1)
    counts = np.bincount(i1, minlength=lat_c.size).astype(np.float64)
    down = sps.csr_matrix((1.0 / counts[i1], (i1, np.arange(lat_h.size))),
                          shape=(lat_c.size, lat_h.size))
    return up, down


def make_world(root: Path, seed: int = 7):
    """Write static inputs; return a dict of paths and arrays."""
    import scipy.sparse as sps
    import zarr

    root = Path(root)
    lat_h, lon_h = _grid(1.0)
    lat_c, lon_c = _grid(4.0)
    up, down = build_matrices(lat_h, lon_h, lat_c, lon_c)
    inter = root / "inter"
    inter.mkdir(parents=True, exist_ok=True)
    sps.save_npz(inter / "up.npz", up)
    sps.save_npz(inter / "down.npz", down)

    rng = np.random.default_rng(seed)
    lon180 = np.where(lon_h > 180.0, lon_h - 360.0, lon_h)
    land = ((lon180 > -95.0) & (lon180 < -80.0) & (lat_h > 25.0) & (lat_h < 50.0)) | (lat_h < -70.0)
    lsm = land.astype(np.float32)
    z = np.where(land, 500.0 + 300.0 * rng.standard_normal(lat_h.size), 0.0).astype(np.float32)
    zpath = root / "forcings.zarr"
    g = zarr.open_group(str(zpath), mode="w", zarr_format=2)
    g.create_array("latitudes", data=lat_h)
    g.create_array("longitudes", data=lon_h)
    data = np.zeros((1, 2, 1, lat_h.size), dtype=np.float32)
    data[0, 0, 0] = lsm
    data[0, 1, 0] = z * 9.80665
    g.create_array("data", data=data)
    (zpath / ".zattrs").write_text(json.dumps({"variables": ["lsm", "z"]}))

    resid = root / "resid.npy"
    np.save(resid, {"stdev": {s: 1.0 for s in STATES}}, allow_pickle=True)
    return {
        "lat_h": lat_h, "lon_h": lon_h, "lat_c": lat_c, "lon_c": lon_c, "up": up, "down": down,
        "paths": {
            "up_matrix": str(inter / "up.npz"), "down_matrix": str(inter / "down.npz"),
            "residual_stats": str(resid), "forcings_zarr": str(zpath),
            "knn_cache": str(root / "knn.npz"),
        },
    }


def smooth_field(rng, lat, lon, n_modes: int = 12, scale: float = 1.0):
    """A sum of a few random spherical plane waves; smooth at grid scale."""
    xyz = _unit(lat, lon)
    f = np.zeros(lat.size)
    for _ in range(n_modes):
        k = rng.standard_normal(3) * 6.0
        f += rng.standard_normal() * np.cos(xyz @ k + rng.uniform(0, 2 * np.pi))
    return scale * f


def write_prediction(path: Path, world: dict, n_members: int, seed: int, cut: bool):
    """A prediction file; when ``cut`` is True only the BOX points, with x_interp."""
    import netCDF4

    rng = np.random.default_rng(seed)
    lat_h, lon_h, lat_c, lon_c, up = (world[k] for k in ("lat_h", "lon_h", "lat_c", "lon_c", "up"))
    lon180 = np.where(lon_h > 180.0, lon_h - 360.0, lon_h)
    n_h, n_c, n_s = lat_h.size, lat_c.size, len(STATES)
    x = np.stack([smooth_field(rng, lat_c, lon_c, scale=3.0) for _ in STATES], axis=1)
    x_interp_full = up @ x
    y = x_interp_full + np.stack([smooth_field(rng, lat_h, lon_h, n_modes=20, scale=0.5)
                                  for _ in STATES], axis=1) + 0.2 * rng.standard_normal((n_h, n_s))
    y_pred = np.stack([
        x_interp_full + np.stack([smooth_field(rng, lat_h, lon_h, n_modes=20, scale=0.5)
                                  for _ in STATES], axis=1)
        + 0.3 * rng.standard_normal((n_h, n_s))
        for _ in range(n_members)
    ], axis=0)                                         # (member, point, state)

    if cut:
        sel = np.flatnonzero((lat_h >= BOX[0]) & (lat_h <= BOX[1])
                             & (lon180 >= BOX[2]) & (lon180 <= BOX[3]))
        # A cut-graph file stores the points in its own order: shuffle them.
        sel = rng.permutation(sel)
    else:
        sel = np.arange(n_h)

    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("sample", 1)
        ds.createDimension("ensemble_member", n_members)
        ds.createDimension("lres_point", n_c)
        ds.createDimension("hres_point", sel.size)
        ds.createDimension("weather_state", n_s)
        ds.createDimension("nchar", 8)
        v = ds.createVariable("weather_state", str, ("weather_state",))
        for k, s in enumerate(STATES):
            v[k] = s
        ds.createVariable("ensemble_member", "i4", ("ensemble_member",))[:] = np.arange(1, n_members + 1)
        ds.createVariable("lat_lres", "f8", ("lres_point",))[:] = lat_c
        ds.createVariable("lon_lres", "f8", ("lres_point",))[:] = lon_c
        ds.createVariable("lat_hres", "f8", ("hres_point",))[:] = lat_h[sel]
        ds.createVariable("lon_hres", "f8", ("hres_point",))[:] = lon_h[sel]
        xv = ds.createVariable("x", "f4", ("sample", "ensemble_member", "lres_point", "weather_state"))
        xv[:] = np.broadcast_to(x[None, None], (1, n_members, n_c, n_s))
        yv = ds.createVariable("y", "f4", ("sample", "ensemble_member", "hres_point", "weather_state"))
        yv[:] = np.broadcast_to(y[sel][None, None], (1, n_members, sel.size, n_s))
        pv = ds.createVariable("y_pred", "f4", ("sample", "ensemble_member", "hres_point", "weather_state"))
        pv[:] = y_pred[:, sel][None]
        if cut:
            iv = ds.createVariable("x_interp", "f4",
                                   ("sample", "ensemble_member", "hres_point", "weather_state"))
            iv[:] = np.broadcast_to(x_interp_full[sel][None, None], (1, n_members, sel.size, n_s))
        ds.setncattr("lead_step_hours", 24)
    return sel
