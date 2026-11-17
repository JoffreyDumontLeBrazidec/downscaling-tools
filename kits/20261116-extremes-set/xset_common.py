"""Shared pieces of the extremes set XSET-1 kit (16 Nov 2026): constants, geometry, the global tail histograms and the
prediction-file writer. numpy only at import; netCDF4 inside write_row_file."""
from __future__ import annotations

import numpy as np

LEADS = (12, 24, 36, 48, 60, 72)
SAVE_VARS = ("10u", "10v", "2t", "msl", "tp")
R_EARTH_KM = 6371.0
# latitude bands of the global histograms (south to north)
BAND_EDGES = (-60.0, -30.0, -10.0, 10.0, 30.0, 60.0)
BAND_NAMES = ("90S-60S", "60S-30S", "30S-10S", "10S-10N", "10N-30N", "30N-60N", "60N-90N")
TP_EDGES_MM = np.array([0, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500, 1e9])
WS_EDGES_MS = np.concatenate([np.arange(0.0, 82.5, 2.5), [1e9]])


def unit_xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def chord(km):
    """Chord length on the unit sphere of a great-circle distance in km (for cKDTree ball queries)."""
    return 2.0 * np.sin(np.asarray(km, np.float64) / R_EARTH_KM / 2.0)


def wrap180(lon):
    return ((np.asarray(lon, np.float64) + 180.0) % 360.0) - 180.0


def band_index(lat):
    return np.digitize(np.asarray(lat, np.float64), BAND_EDGES).astype(np.int64)


def _hist(values, band, edges):
    """values (members, points) -> counts (bands, members, bins) int64 and maxima (bands, members) float32."""
    nb, nbin = len(BAND_NAMES), len(edges) - 1
    m = values.shape[0]
    counts = np.zeros((nb, m, nbin), np.int64)
    maxima = np.full((nb, m), -np.inf, np.float32)
    for k in range(m):
        v = np.asarray(values[k], np.float64)
        b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, nbin - 1)
        counts[:, k, :] = np.bincount(band * nbin + b, minlength=nb * nbin).reshape(nb, nbin)
        np.maximum.at(maxima[:, k], band, v.astype(np.float32))
    return counts, maxima


def tail_histograms(lat, tp_m, u, v):
    """Global histograms of one (date, lead) of one row. tp_m (members, points) in metres per 12 h, u and v in m/s.
    Counts are grid points (not area weighted): rows on the same grid compare exactly, other grids by frequency."""
    band = band_index(lat)
    tp_c, tp_x = _hist(np.asarray(tp_m) * 1000.0, band, TP_EDGES_MM)
    ws_c, ws_x = _hist(np.hypot(np.asarray(u, np.float64), np.asarray(v, np.float64)), band, WS_EDGES_MS)
    return dict(tp_counts=tp_c, tp_max_mm=tp_x, ws_counts=ws_c, ws_max_ms=ws_x, tp_edges_mm=TP_EDGES_MM,
                ws_edges_ms=WS_EDGES_MS, bands=np.array(BAND_NAMES), points=np.bincount(band, minlength=len(BAND_NAMES)))


def write_row_file(path, *, lat, lon, point_index, states, y, y_pred, x=None, lat_lres=None, lon_lres=None,
                   lres_index=None, extra=None, attrs=None):
    """Prediction file in the eval.cli layout on the evaluation points of one (date, lead).
    y (points, states) the single 9 km field; y_pred (members, points, states); x (members, native points, states) the
    native input members (rule 41); extra {name: (points,) array} e.g. imerg_tp12, why."""
    import netCDF4
    n_m, n_p, n_s = y_pred.shape
    with netCDF4.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", n_m)
        d.createDimension("grid_point", n_p)
        d.createDimension("weather_state", n_s)
        d.createVariable("lat_hres", "f8", ("grid_point",))[:] = lat
        d.createVariable("lon_hres", "f8", ("grid_point",))[:] = lon
        d.createVariable("point_index", "i4", ("grid_point",))[:] = point_index
        v = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            v[i] = s
        d.createVariable("ensemble_member", "i4", ("ensemble_member",))[:] = np.arange(1, n_m + 1)
        d.createVariable("y", "f4", ("sample", "grid_point", "weather_state"))[:] = y[None]
        d.createVariable("y_pred", "f4", ("sample", "ensemble_member", "grid_point", "weather_state"))[:] = y_pred[None]
        if x is not None:
            d.createDimension("grid_point_lres", x.shape[1])
            d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = lat_lres
            d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = lon_lres
            d.createVariable("lres_index", "i4", ("grid_point_lres",))[:] = lres_index
            d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))[:] = x[None]
        for name, arr in (extra or {}).items():
            arr = np.asarray(arr)
            d.createVariable(name, arr.dtype.str[1:], ("grid_point",))[:] = arr
        for k, val in (attrs or {}).items():
            d.setncattr(k, val if isinstance(val, (int, float)) else str(val))
