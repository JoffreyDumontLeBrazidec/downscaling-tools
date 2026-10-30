"""The large-scale low-pass ``P_L`` behind the scale split of the probabilistic scores.

Why one operator
----------------
The read of a downscaling run wants to know how much of the fair CRPS, the spread and the
ensemble-mean error sit above the input's own scales (the synoptic drift between the o320
input and the o1280 target, ablation note section 15) and how much below (the detail the
model is asked to add). Every band score is therefore computed from ONE low-pass
``P_L``: the large part of a field is ``L = P_L(field)``, the fine part ``F = field - L``,
applied identically to every member and to the truth, so the two parts add up to the field
and the same operator is used for every score, variable, member and arm.

The operator
------------
``P_L`` is an isotropic Gaussian taper in kilometres (the TD-248 kit: sigma 53 km has its
half-power wavelength at 400 km, 33 km at 250 km, 80 km at 600 km; ``sigma = cut_km *
sqrt(ln 2) / (2 pi)``). A Gaussian keeps the power ``exp(-4 pi^2 sigma^2 / lambda^2)`` at
wavelength ``lambda``: 0.5 at the cut, 0.17 at 250 km for the 400 km cut, below 1e-4 at
100 km, 0.90 at 1000 km, 0.97 at 2000 km. It is smooth (no ringing, no negative lobes)
and the same on every grid; it is idempotent where its transfer is 0 or 1 and not at the
cut itself (applying it twice keeps 0.25 of the power there), which is the price of a
taper over a sharp projection and is accepted (review of 30 Oct 2026, section 15.7).

A direct Gaussian on the o1280 points is out of reach (1,000-1,800 neighbours per point at
3-4 sigma for 6.6 M points), so ``P_L`` is factored in three steps, each cheap:

1. **Conservative area average onto a regular lat-lon grid** of ``grid_km`` (default 25 km,
   0.225 degrees, 800 x 1600 cells for the globe). Each point adds its area weight times its
   value to its cell; the cell value is the weighted mean. The grid is always global, so a
   regional box file simply leaves most cells empty. A 25 km box average keeps 0.987 of the
   power at 400 km (``sinc^2(pi 25 / 400)``), a 50 km one only 0.95, which is why the grid
   is finer than the 50 km first proposed.
2. **Gaussian on the grid, isotropic in km**, as a normalised convolution: the masked field
   and the mask are both filtered and divided, so empty cells (outside a regional box, or a
   gap in a sparse test grid) carry no weight and the result near a box edge is the
   Gaussian average of the data that exist. The zonal pass uses sigma / (grid_km cos lat)
   cells per latitude row, wrapping at the dateline, and a row whose sigma exceeds a
   quarter of the circle (within about 1.5 degrees of a pole for the 400 km cut) becomes
   its zonal mean. The meridional pass sees the field continued across each pole (the
   rows mirrored and shifted by 180 degrees), so the poles are not edges.
3. **Cubic spline interpolation back to the points** (``scipy.ndimage.map_coordinates``,
   order 3, on the grid padded zonally and across the poles). Bilinear interpolation
   would imprint the 25 km lattice as a piecewise-linear error of several percent of the
   400 km amplitude; the cubic spline keeps it below half a percent. Steps 1 and 3 are
   linear in the number of points; step 2 is linear in the number of grid cells.

What this means for the scores
------------------------------
The transfer measured in ``tests/test_scale_split.py`` on a synthetic 9 km point set is the
product of the three steps; it is reported there and stays within 0.02 of the Gaussian's
own curve from 250 km up and leaks less than 0.002 of the power at 100 km. The L band of
a field is then "what the o320 input could already resolve"; the F band is everything
below the cut, including the model's fine detail and the storm wings around 200-300 km
that the taper damps by 20-40 % (so a feature at 250 km is split between the bands, not
assigned to one).

Edges of a regional box: the Gaussian average is one-sided within a few sigma of the box
edge, so points closer than ``edge_margin_km`` (default: the cut, 400 km, as the review
asked) to the bounding box of a non-global point set are flagged and left out of the band
scores (``inner_mask``). A global point set (longitude span above 350 degrees, latitude
span above 170) has no edges. Precipitation: a low-pass of rain is a legitimate band split
of the scores (the L part is the rain mass at synoptic scale, the F part its placement
and intensity), and nothing is written back as a field, so the negative values a Gaussian
of rain creates in the F part are not a problem; see ``scoring.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KM_PER_DEG = 111.2
DEFAULT_CUT_KM = 400.0
DEFAULT_GRID_KM = 25.0

# sigma such that the Gaussian keeps half the POWER at the cut wavelength
SIGMA_PER_CUT = math.sqrt(math.log(2.0)) / (2.0 * math.pi)   # 0.1325; 400 km -> 53.0 km

# a row whose zonal sigma exceeds this fraction of the circle becomes its zonal mean
_POLAR_ROW_FRACTION = 0.25
# grid cells of padding for the spline interpolation (order 3 needs 2; 4 is safe)
_PAD = 4


def sigma_km_for_cut(cut_km: float) -> float:
    """Gaussian sigma (km) whose half-power wavelength is ``cut_km`` (400 -> 53.0)."""
    return float(cut_km) * SIGMA_PER_CUT


def gaussian_power_transfer(wavelength_km: np.ndarray | float, sigma_km: float) -> np.ndarray:
    """Power kept by a Gaussian of ``sigma_km`` at ``wavelength_km`` (0.5 at the cut)."""
    lam = np.asarray(wavelength_km, dtype=np.float64)
    return np.exp(-4.0 * math.pi ** 2 * sigma_km ** 2 / lam ** 2)


def is_global(lat: np.ndarray, lon: np.ndarray) -> bool:
    """True when the point set covers the sphere (no box edges to flag)."""
    lon = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    return bool(np.ptp(lat) > 170.0 and np.ptp(lon) > 350.0)


def inner_mask(lat: np.ndarray, lon: np.ndarray, margin_km: float) -> np.ndarray:
    """Points farther than ``margin_km`` from the bounding box of a regional point set.

    Every point of a global set is inner. The longitude span is measured on the
    -180..180 wrap unless the box crosses the dateline, when the 0..360 wrap is tighter.
    """
    lat = np.asarray(lat, dtype=np.float64)
    if is_global(lat, lon) or margin_km <= 0.0:
        return np.ones(lat.shape, dtype=bool)
    lon180 = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lon360 = lon180 % 360.0
    lon = lon180 if np.ptp(lon180) <= np.ptp(lon360) else lon360
    margin_deg_lat = margin_km / KM_PER_DEG
    margin_deg_lon = margin_km / (KM_PER_DEG * np.maximum(np.cos(np.deg2rad(lat)), 1e-3))
    return ((lat - lat.min() >= margin_deg_lat) & (lat.max() - lat >= margin_deg_lat)
            & (lon - lon.min() >= margin_deg_lon) & (lon.max() - lon >= margin_deg_lon))


def box_edge_mask(lat: np.ndarray, lon: np.ndarray, edge_deg: float) -> np.ndarray:
    """Points at least ``edge_deg`` degrees (latitude and longitude) inside a regional box.

    The wide box kits drop a 1-degree strip at the E2 box edge before scoring; this is the
    same strip, applied to every band. Every point of a global set is kept.
    """
    lat = np.asarray(lat, dtype=np.float64)
    if is_global(lat, lon) or edge_deg <= 0.0:
        return np.ones(lat.shape, dtype=bool)
    lon180 = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lon360 = lon180 % 360.0
    lon = lon180 if np.ptp(lon180) <= np.ptp(lon360) else lon360
    return ((lat - lat.min() >= edge_deg) & (lat.max() - lat >= edge_deg)
            & (lon - lon.min() >= edge_deg) & (lon.max() - lon >= edge_deg))


def _pole_pad(grid: np.ndarray, rows: int) -> np.ndarray:
    """Continue a (lat, lon) grid across both poles: mirrored rows shifted by 180 degrees."""
    n_lon = grid.shape[1]
    shift = n_lon // 2
    south = np.roll(grid[:rows][::-1], shift, axis=1)
    north = np.roll(grid[-rows:][::-1], shift, axis=1)
    return np.concatenate([south, grid, north], axis=0)


@dataclass
class LowPassOperator:
    """``P_L`` bound to one point set: build once per grid, apply to every field.

    ``lat``/``lon`` are the points in degrees, ``weights`` their area weights (any scale).
    ``apply(field)`` returns ``P_L(field)`` at the same points, NaN where the field is NaN.
    """

    lat: np.ndarray
    lon: np.ndarray
    weights: np.ndarray | None = None
    cut_km: float = DEFAULT_CUT_KM
    grid_km: float = DEFAULT_GRID_KM

    def __post_init__(self) -> None:
        from scipy.ndimage import gaussian_filter1d  # noqa: F401 - fail early if scipy is absent

        lat = np.asarray(self.lat, dtype=np.float64).reshape(-1)
        lon = ((np.asarray(self.lon, dtype=np.float64).reshape(-1) + 180.0) % 360.0) - 180.0
        if lat.size != lon.size:
            raise ValueError(f"lat/lon size mismatch: {lat.size} vs {lon.size}")
        self.sigma_km = sigma_km_for_cut(self.cut_km)
        self.cell_deg = float(self.grid_km) / KM_PER_DEG
        self.n_lat = max(int(round(180.0 / self.cell_deg)), 4)
        self.n_lon = max(int(round(360.0 / self.cell_deg)), 8)
        self.cell_deg_lat = 180.0 / self.n_lat
        self.cell_deg_lon = 360.0 / self.n_lon
        # fractional grid coordinates of every point (cell centres at integer indices)
        self._fi = (lat + 90.0) / self.cell_deg_lat - 0.5
        self._fj = (lon + 180.0) / self.cell_deg_lon - 0.5
        ii = np.clip(np.rint(self._fi).astype(np.int64), 0, self.n_lat - 1)
        jj = np.rint(self._fj).astype(np.int64) % self.n_lon
        self._cell = ii * self.n_lon + jj
        w = np.ones(lat.size) if self.weights is None else np.asarray(self.weights, dtype=np.float64).reshape(-1)
        self._w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
        # zonal sigma per row, in cells; polar rows become their zonal mean
        row_lat = -90.0 + self.cell_deg_lat * (np.arange(self.n_lat) + 0.5)
        dx_km = self.grid_km * np.maximum(np.cos(np.deg2rad(row_lat)), 1e-6) * (self.cell_deg_lon / self.cell_deg)
        self._sigma_x = self.sigma_km / dx_km
        self._polar_row = self._sigma_x > _POLAR_ROW_FRACTION * self.n_lon
        self._sigma_y = self.sigma_km / (self.grid_km * self.cell_deg_lat / self.cell_deg)
        self._pole_rows = int(min(self.n_lat // 2, math.ceil(4.0 * self._sigma_y) + _PAD))
        self.global_coverage = is_global(lat, lon)

    # ------------------------------------------------------------------ pieces
    def _bin(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n_cells = self.n_lat * self.n_lon
        return np.bincount(self._cell, weights=values * weights, minlength=n_cells).reshape(self.n_lat, self.n_lon)

    def _smooth(self, grid: np.ndarray) -> np.ndarray:
        """Separable Gaussian on the (lat, lon) grid: zonal per row with wrap, then across the poles."""
        from scipy.ndimage import gaussian_filter1d

        out = np.empty_like(grid)
        sig = np.round(self._sigma_x, 3)
        for s in np.unique(sig[~self._polar_row]):
            rows = (sig == s) & ~self._polar_row
            out[rows] = gaussian_filter1d(grid[rows], float(s), axis=1, mode="wrap", truncate=4.0)
        if np.any(self._polar_row):
            out[self._polar_row] = grid[self._polar_row].mean(axis=1, keepdims=True)
        padded = _pole_pad(out, self._pole_rows)
        padded = gaussian_filter1d(padded, self._sigma_y, axis=0, mode="nearest", truncate=4.0)
        return padded[self._pole_rows:self._pole_rows + self.n_lat]

    def _interpolate(self, grid: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        padded = _pole_pad(grid, _PAD)
        padded = np.concatenate([padded[:, -_PAD:], padded, padded[:, :_PAD]], axis=1)
        coords = np.vstack([self._fi + _PAD, self._fj + _PAD])
        return map_coordinates(padded, coords, order=3, mode="nearest", prefilter=True)

    # ------------------------------------------------------------------- apply
    def apply(self, field: np.ndarray) -> np.ndarray:
        """``P_L(field)`` at the points; NaN points carry no weight and come back NaN."""
        f = np.asarray(field, dtype=np.float64).reshape(-1)
        if f.size != self._cell.size:
            raise ValueError(f"field has {f.size} points, operator was built for {self._cell.size}")
        ok = np.isfinite(f)
        w = np.where(ok, self._w, 0.0)
        num = self._smooth(self._bin(np.where(ok, f, 0.0), w))
        den = self._smooth(self._bin(np.ones_like(f), w))
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = np.where(den > 1e-12 * den.max(), num / np.maximum(den, 1e-300), 0.0)
        out = self._interpolate(grid)
        out[~ok] = np.nan
        return out

    def split(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(L, F)``: the large part ``P_L(field)`` and the fine part ``field - L``."""
        large = self.apply(field)
        return large, np.asarray(field, dtype=np.float64).reshape(-1) - large

    def describe(self) -> dict:
        """The settings, for the summary JSON."""
        return {
            "cut_km": float(self.cut_km),
            "sigma_km": float(self.sigma_km),
            "grid_km": float(self.grid_km),
            "grid_cells": [int(self.n_lat), int(self.n_lon)],
            "global_coverage": bool(self.global_coverage),
            "method": "area-weighted bin to lat-lon grid, isotropic Gaussian in km (normalised "
                      "convolution, dateline wrap, pole continuation), cubic spline back to the points",
        }
