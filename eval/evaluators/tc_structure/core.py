"""Tropical-cyclone structure measured on the native grid points (pure functions).

Definitions (design note 20260923_physical_realism_scores, "Evaluator tc_structure"):

1. Centre. The minimum of msl inside the event box (optionally restricted to a
   disc of ``search_km`` around a first guess), refined as the pressure-weighted
   centroid of the grid points within 1 hPa of that minimum. The weight of a point
   is ``Pmin + 1 hPa - p``, so the deepest point weighs most and a point exactly
   1 hPa above the minimum weighs nothing. Only points within ``refine_km`` of the
   grid minimum enter the centroid, so a second, disconnected low that happens to
   lie within 1 hPa cannot pull the centre.
2. Central pressure Pmin: the grid-point minimum of msl found in step 1 (hPa). It
   is deliberately the value at the grid minimum, not a value interpolated at the
   refined centre, so that it equals the minimum the ``tc`` evaluator reads.
3. Tangential wind profile. The 10 m wind at every point within ``rmax_km`` of the
   centre is split into radial and tangential components using the great-circle
   bearing from the centre to the point, evaluated at the point. The tangential
   component (positive = cyclonic; the sign follows the hemisphere of the centre)
   is averaged in radial bins of ``dr_km``. A bin is kept only if it holds at
   least ``min_coverage`` of the points an annulus of that area should hold at the
   nominal O1280 point density, and at least 2 points.
4. Radius of maximum wind (RMW): radius of the maximum of the binned profile,
   refined by a parabola through that bin and its two neighbours; the parabola's
   peak value is the azimuthal-mean maximum tangential wind ``vmax_tan``. Also the
   plain maximum 10 m wind speed within 300 km.
5. Wind radii R34 / R50: outermost radius where the binned profile is at least
   17.5 / 25.7 m/s, linearly interpolated to the crossing with the next bin. NaN
   (missing, not zero) when the threshold is never reached.
6. Rotation: mean relative vorticity inside radius r from the circulation,
   2 Vt(r) / r (s^-1), with Vt(r) linearly interpolated between bin centres.
7. Asymmetry: amplitude of the wavenumber-1 Fourier component of the 10 m wind
   speed on the ring |r - RMW| <= dr/2, divided by the ring mean (least-squares
   fit of a0 + a1 cos(az) + b1 sin(az) over the ring points).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

R_EARTH_KM = 6371.0
# Nominal area per O1280 grid point: surface of the sphere / 6,599,680 points.
O1280_NPOINTS = 6_599_680
O1280_CELL_AREA_KM2 = 4.0 * math.pi * R_EARTH_KM ** 2 / O1280_NPOINTS
KT34_MS = 17.5
KT50_MS = 25.7


@dataclass(frozen=True)
class StructureParams:
    search_km: float = 250.0        # centre search disc around the first guess
    search_edge_km: float = 15.0    # a minimum this close to the disc edge = storm not in the disc
    box_edge_km: float = 15.0       # a minimum this close to the event-box edge = not a closed centre
    refine_km: float = 100.0        # centroid only from points this close to the grid minimum
    refine_hpa: float = 1.0         # centroid from points within this of the minimum
    rmax_km: float = 500.0
    dr_km: float = 10.0
    min_coverage: float = 0.5
    min_bin_points: int = 2
    cell_area_km2: float = O1280_CELL_AREA_KM2
    maxwind_radius_km: float = 300.0
    vort_radii_km: tuple = (50.0, 100.0, 200.0)
    ring_min_points: int = 8
    ring_min_octants: int = 6
    pressure_ref_hpa: float = 1010.0


# --------------------------------------------------------------------------------------
# spherical geometry
# --------------------------------------------------------------------------------------


def unit_xyz(lat_deg, lon_deg) -> np.ndarray:
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lo = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    cl = np.cos(la)
    return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=-1)


def gc_distance_km(lat0, lon0, lat, lon) -> np.ndarray:
    """Great-circle distance (haversine), km."""
    p0 = math.radians(float(lat0))
    p = np.deg2rad(np.asarray(lat, dtype=np.float64))
    dl = np.deg2rad(np.asarray(lon, dtype=np.float64) - float(lon0))
    a = np.sin(0.5 * (p - p0)) ** 2 + math.cos(p0) * np.cos(p) * np.sin(0.5 * dl) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing(lat0, lon0, lat, lon) -> np.ndarray:
    """Bearing (radians, clockwise from north) of the great circle from the centre
    (lat0, lon0) towards each point, measured at the centre: the azimuth around it."""
    p0 = math.radians(float(lat0))
    p = np.deg2rad(np.asarray(lat, dtype=np.float64))
    dl = np.deg2rad(np.asarray(lon, dtype=np.float64) - float(lon0))
    y = np.sin(dl) * np.cos(p)
    x = math.cos(p0) * np.sin(p) - math.sin(p0) * np.cos(p) * np.cos(dl)
    return np.arctan2(y, x)


def outward_bearing_at_point(lat0, lon0, lat, lon) -> np.ndarray:
    """Direction (radians, clockwise from north) pointing away from the centre along
    the great circle, evaluated at each point (the final bearing centre -> point)."""
    p0 = math.radians(float(lat0))
    p = np.deg2rad(np.asarray(lat, dtype=np.float64))
    dl = np.deg2rad(float(lon0) - np.asarray(lon, dtype=np.float64))
    # initial bearing from the point back to the centre, then reversed
    y = np.sin(dl) * math.cos(p0)
    x = np.cos(p) * math.sin(p0) - np.sin(p) * math.cos(p0) * np.cos(dl)
    return np.arctan2(y, x) + math.pi


def box_edge_distance_km(lat, lon, bbox) -> np.ndarray:
    """Distance (km, approximate) from each point to the nearest edge of the box
    (south, north, west, east in degrees; lon in the same convention as the points)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    south, north, west, east = bbox
    kdeg = math.pi / 180.0 * R_EARTH_KM
    d_lat = np.minimum(lat - south, north - lat) * kdeg
    d_lon = np.minimum(lon - west, east - lon) * kdeg * np.cos(np.deg2rad(lat))
    return np.minimum(d_lat, d_lon)


# --------------------------------------------------------------------------------------
# centre
# --------------------------------------------------------------------------------------


def find_centre(lat, lon, msl_hpa, candidate, *, bbox=None, first_guess=None,
                params: StructureParams = StructureParams()) -> dict:
    """Centre of the low in ``msl_hpa`` among the ``candidate`` points (boolean mask).

    Returns a dict with ``found`` (bool), ``reason`` (str, empty when found),
    ``pmin_hpa``, ``lat``, ``lon`` (refined centre), ``imin`` (grid minimum index).
    """
    cand = np.asarray(candidate, dtype=bool).copy()
    if first_guess is not None:
        dfg = gc_distance_km(first_guess[0], first_guess[1], lat, lon)
        cand &= dfg <= params.search_km
    idx = np.flatnonzero(cand)
    out = {"found": False, "reason": "", "pmin_hpa": np.nan, "lat": np.nan, "lon": np.nan,
           "imin": -1, "grid_lat": np.nan, "grid_lon": np.nan}
    if idx.size == 0:
        out["reason"] = "no candidate points"
        return out
    vals = np.asarray(msl_hpa, dtype=np.float64)[idx]
    j = int(np.argmin(vals))
    imin = int(idx[j])
    pmin = float(vals[j])
    out.update(pmin_hpa=pmin, imin=imin, grid_lat=float(lat[imin]), grid_lon=float(lon[imin]))

    near = idx[gc_distance_km(lat[imin], lon[imin], lat[idx], lon[idx]) <= params.refine_km]
    p = np.asarray(msl_hpa, dtype=np.float64)[near]
    sel = p <= pmin + params.refine_hpa
    w = (pmin + params.refine_hpa) - p[sel]
    if w.sum() <= 0:
        w = np.ones(int(sel.sum()))
    xyz = unit_xyz(lat[near][sel], lon[near][sel])
    c = (xyz * w[:, None]).sum(axis=0)
    c /= np.linalg.norm(c)
    clat = math.degrees(math.asin(max(-1.0, min(1.0, c[2]))))
    clon = math.degrees(math.atan2(c[1], c[0]))
    # keep the longitude convention of the input points
    if np.nanmax(lon) > 180.0 and clon < 0:
        clon += 360.0
    out.update(lat=clat, lon=clon)

    if first_guess is not None:
        d = float(gc_distance_km(first_guess[0], first_guess[1], [lat[imin]], [lon[imin]])[0])
        if d > params.search_km - params.search_edge_km:
            out["reason"] = "minimum on the edge of the search disc"
            return out
    if bbox is not None:
        de = float(box_edge_distance_km([lat[imin]], [lon[imin]], bbox)[0])
        if de < params.box_edge_km:
            out["reason"] = "minimum on the edge of the event box"
            return out
    out["found"] = True
    return out


# --------------------------------------------------------------------------------------
# profile and derived scores
# --------------------------------------------------------------------------------------


def tangential_profile(lat, lon, u, v, clat, clon, params: StructureParams = StructureParams()):
    """Binned azimuthal-mean tangential (and radial) wind about the centre.

    Returns dict with ``r_centres`` (km), ``vt`` (m/s, NaN in rejected bins),
    ``vr``, ``count``, ``expected`` and the per-point arrays needed downstream
    (``r``, ``az``, ``speed``) for the points within rmax.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    # cheap prefilter before the exact distance
    dlat = params.rmax_km / 111.0 + 0.2
    pre = np.abs(lat - clat) <= dlat
    r_all = np.full(lat.shape, np.inf)
    r_all[pre] = gc_distance_km(clat, clon, lat[pre], lon[pre])
    inside = r_all <= params.rmax_km
    ii = np.flatnonzero(inside)
    r = r_all[ii]
    beta = outward_bearing_at_point(clat, clon, lat[ii], lon[ii])
    uu = np.asarray(u, dtype=np.float64)[ii]
    vv = np.asarray(v, dtype=np.float64)[ii]
    sign = 1.0 if clat >= 0 else -1.0
    # radial unit vector (east, north) = (sin b, cos b); cyclonic (NH: counter-clockwise)
    # tangential unit vector = (-cos b, sin b)
    vt = sign * (-uu * np.cos(beta) + vv * np.sin(beta))
    vr = uu * np.sin(beta) + vv * np.cos(beta)
    nb = int(round(params.rmax_km / params.dr_km))
    b = np.minimum((r / params.dr_km).astype(np.int64), nb - 1)
    count = np.bincount(b, minlength=nb).astype(np.float64)
    s_vt = np.bincount(b, weights=vt, minlength=nb)
    s_vr = np.bincount(b, weights=vr, minlength=nb)
    edges = np.arange(nb + 1) * params.dr_km
    expected = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / params.cell_area_km2
    valid = (count >= params.min_bin_points) & (count >= params.min_coverage * expected)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof_vt = np.where(valid, s_vt / np.maximum(count, 1), np.nan)
        prof_vr = np.where(valid, s_vr / np.maximum(count, 1), np.nan)
    return {
        "r_centres": 0.5 * (edges[1:] + edges[:-1]),
        "vt": prof_vt, "vr": prof_vr, "count": count, "expected": expected,
        "r": r, "az": initial_bearing(clat, clon, lat[ii], lon[ii]),
        "speed": np.hypot(uu, vv),
    }


def rmw_and_vmax(r_centres, vt, dr_km):
    """(rmw_km, vmax_parabola, vmax_bin, imax). NaN if the profile is empty."""
    if not np.isfinite(vt).any():
        return np.nan, np.nan, np.nan, -1
    i = int(np.nanargmax(vt))
    y0 = float(vt[i])
    if 0 < i < len(vt) - 1 and np.isfinite(vt[i - 1]) and np.isfinite(vt[i + 1]):
        ym, yp = float(vt[i - 1]), float(vt[i + 1])
        den = ym - 2.0 * y0 + yp
        if den < 0:
            off = 0.5 * (ym - yp) / den
            off = max(-1.0, min(1.0, off))
            return (float(r_centres[i] + off * dr_km), float(y0 - 0.25 * (ym - yp) * off), y0, i)
    return float(r_centres[i]), y0, y0, i


def vt_at(r_centres, vt, r_km):
    """Linear interpolation of the binned profile at radius r; NaN unless both
    bracketing bins are valid."""
    rc = np.asarray(r_centres)
    if r_km <= rc[0] or r_km >= rc[-1]:
        return np.nan
    j = int(np.searchsorted(rc, r_km)) - 1
    a, b = vt[j], vt[j + 1]
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    t = (r_km - rc[j]) / (rc[j + 1] - rc[j])
    return float(a + t * (b - a))


def wind_radius(r_centres, vt, threshold):
    """(radius_km, censored). Outermost radius where vt >= threshold; NaN if never."""
    ok = np.isfinite(vt) & (vt >= threshold)
    if not ok.any():
        return np.nan, False
    i = int(np.flatnonzero(ok)[-1])
    if i + 1 >= len(vt):
        return float(r_centres[i]), True
    if not np.isfinite(vt[i + 1]):
        # the profile stops (rejected bin) while still above the threshold
        return float(r_centres[i]), True
    a, b = float(vt[i]), float(vt[i + 1])
    t = (a - threshold) / (a - b) if a != b else 0.0
    return float(r_centres[i] + t * (r_centres[i + 1] - r_centres[i])), False


def ring_asymmetry(r, az, speed, rmw_km, dr_km, params: StructureParams = StructureParams()):
    """(wavenumber-1 amplitude / ring mean, n ring points)."""
    if not np.isfinite(rmw_km):
        return np.nan, 0
    ring = np.abs(r - rmw_km) <= 0.5 * dr_km
    n = int(ring.sum())
    if n < params.ring_min_points:
        return np.nan, n
    a = az[ring]
    octs = np.unique(np.floor(((a % (2 * math.pi)) / (2 * math.pi)) * 8).astype(int))
    if octs.size < params.ring_min_octants:
        return np.nan, n
    A = np.stack([np.ones(n), np.cos(a), np.sin(a)], axis=1)
    coef, *_ = np.linalg.lstsq(A, speed[ring], rcond=None)
    if coef[0] <= 0:
        return np.nan, n
    return float(math.hypot(coef[1], coef[2]) / coef[0]), n


def measure_structure(lat, lon, u, v, clat, clon, params: StructureParams = StructureParams()):
    """All profile-based scores about a given centre. Returns (scores dict, vt profile)."""
    prof = tangential_profile(lat, lon, u, v, clat, clon, params)
    rc, vt = prof["r_centres"], prof["vt"]
    rmw, vmax, vmax_bin, _ = rmw_and_vmax(rc, vt, params.dr_km)
    r34, c34 = wind_radius(rc, vt, KT34_MS)
    r50, c50 = wind_radius(rc, vt, KT50_MS)
    within = prof["r"] <= params.maxwind_radius_km
    maxwind = float(prof["speed"][within].max()) if within.any() else np.nan
    asym, nring = ring_asymmetry(prof["r"], prof["az"], prof["speed"], rmw, params.dr_km, params)
    out = {
        "rmw_km": rmw, "vmax_tan_ms": vmax, "vmax_tan_bin_ms": vmax_bin,
        "maxwind300_ms": maxwind, "r34_km": r34, "r34_censored": c34,
        "r50_km": r50, "r50_censored": c50, "asym_rmw": asym, "ring_points": nring,
        "n_valid_bins": int(np.isfinite(vt).sum()),
    }
    for rk in params.vort_radii_km:
        vtr = vt_at(rc, vt, rk)
        out[f"zeta{int(rk)}_s"] = 2.0 * vtr / (rk * 1000.0) if np.isfinite(vtr) else np.nan
    return out, vt


# --------------------------------------------------------------------------------------
# helpers for tests: the O1280 grid points of a box
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _gaussian_latitudes(n_lat):
    x, _ = np.polynomial.legendre.leggauss(2 * n_lat)
    return np.degrees(np.arcsin(x))[::-1]            # north to south


def o1280_points_in_box(lat_min, lat_max, lon_min, lon_max, n_lat=1280):
    """Octahedral reduced Gaussian grid O<n_lat> points inside a lat/lon box.

    Rows are the 2*n_lat Gaussian latitudes; the row i-th from the pole (i = 1..n_lat)
    carries 20 + 4 (i - 1) equally spaced longitudes starting at 0. Longitudes are
    returned in -180..180. For n_lat = 1280 the full grid has 6,599,680 points.
    """
    lats = _gaussian_latitudes(n_lat)
    rows_lat, rows_lon = [], []
    for k, la in enumerate(lats):
        if la < lat_min or la > lat_max:
            continue
        i = k + 1 if k < n_lat else 2 * n_lat - k
        n = 20 + 4 * (i - 1)
        lo = np.arange(n) * (360.0 / n)
        lo = ((lo + 180.0) % 360.0) - 180.0
        keep = (lo >= lon_min) & (lo <= lon_max)
        rows_lat.append(np.full(int(keep.sum()), la))
        rows_lon.append(lo[keep])
    return np.concatenate(rows_lat), np.concatenate(rows_lon)
