"""Octahedral reduced Gaussian grids and ring-wise bilinear interpolation to a regular grid.

An octahedral grid O<N> has 2N latitude rings at the Gaussian latitudes (the roots of the
Legendre polynomial P_2N). Ring i (i = 1..N from the north pole) holds 4 i + 16 points,
equally spaced in longitude from 0 degrees east; the southern rings mirror the northern ones.
O320 has 421,120 points, O1280 6,599,680, O2560 26,306,560.

``RingBilinear`` interpolates a field given on such a grid (all of it, or a crop of it such as
a box store) to any set of target latitudes and longitudes. On each of the two rings that
bracket a target latitude it interpolates linearly in longitude between the two nearest
points of that ring, then linearly in latitude between the two rings. This is the bilinear
method of ECMWF's MIR for reduced Gaussian grids: it uses the grid's real structure, needs no
triangulation and no nearest-neighbour search, and it is exact for a field that is linear in
longitude on each ring. North of the first ring and south of the last ring the value of that
ring is used (in longitude it is still interpolated). Target points whose four source points
are not all present (outside a crop) are NaN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gaussian_latitudes(n_half: int, *, iterations: int = 100, tol: float = 1e-15) -> np.ndarray:
    """The 2 ``n_half`` Gaussian latitudes in degrees, north to south.

    Roots of the Legendre polynomial of degree 2 ``n_half``, found with Newton's method from
    Tricomi's first guess, all roots at once. Agrees with ``numpy.polynomial.legendre.leggauss``
    to about 1e-12 degrees and costs a fraction of a second for O2560.
    """
    n = 2 * int(n_half)
    k = np.arange(1, n_half + 1, dtype=np.float64)
    theta = np.pi * (k - 0.25) / (n + 0.5)
    x = np.cos(theta + 1.0 / (8.0 * n * n) / np.tan(theta))  # Tricomi, northern half
    for _ in range(iterations):
        p_prev = np.ones_like(x)
        p = x.copy()
        for m in range(2, n + 1):
            p_prev, p = p, ((2 * m - 1) * x * p - (m - 1) * p_prev) / m
        dp = n * (x * p - p_prev) / (x * x - 1.0)
        dx = p / dp
        x = x - dx
        if np.max(np.abs(dx)) < tol:
            break
    north = np.degrees(np.arcsin(x))
    return np.concatenate([north, -north[::-1]])


def octahedral_n_from_size(n_points: int) -> int | None:
    """N of the octahedral grid with ``n_points`` points (4 N (N + 9)), or None."""
    n = int(round((-36 + np.sqrt(36 * 36 + 16 * n_points)) / 8))
    return n if n > 0 and 4 * n * (n + 9) == n_points else None


@dataclass(frozen=True)
class OctahedralGrid:
    """Ring structure of O<N>: latitudes, points per ring and the offset of each ring."""

    n: int
    lats: np.ndarray      # (2N,) degrees, north to south
    nlon: np.ndarray      # (2N,) points per ring
    offsets: np.ndarray   # (2N + 1,) index of the first point of each ring, then the total

    @classmethod
    def build(cls, n: int) -> "OctahedralGrid":
        i = np.arange(1, n + 1)
        north = 4 * i + 16
        nlon = np.concatenate([north, north[::-1]]).astype(np.int64)
        offsets = np.concatenate([[0], np.cumsum(nlon)]).astype(np.int64)
        return cls(n=int(n), lats=gaussian_latitudes(n), nlon=nlon, offsets=offsets)

    @property
    def name(self) -> str:
        return f"O{self.n}"

    @property
    def size(self) -> int:
        return int(self.offsets[-1])

    def point_lat_lon(self) -> tuple[np.ndarray, np.ndarray]:
        """Latitude and longitude of every point in the standard order (north to south, from 0 E)."""
        ring = np.repeat(np.arange(2 * self.n), self.nlon)
        j = np.arange(self.size) - self.offsets[ring]
        return self.lats[ring], 360.0 * j / self.nlon[ring]

    def index_of(self, lat, lon, *, tol_deg: float | None = None) -> np.ndarray:
        """Global index of each point (``lat``, ``lon``), checked against the grid.

        Raises ``ValueError`` when a point is farther than ``tol_deg`` from the grid point it is
        assigned to (default: a tenth of the finest ring spacing), so a wrong ``n`` or a
        different grid never passes silently.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
        # rings are sorted north to south: search in the ascending reversed array
        asc = self.lats[::-1]
        pos = np.clip(np.searchsorted(asc, lat), 1, len(asc) - 1)
        lower, upper = asc[pos - 1], asc[pos]
        pos = np.where(np.abs(lat - lower) <= np.abs(lat - upper), pos - 1, pos)
        ring = len(asc) - 1 - pos
        dlon = 360.0 / self.nlon[ring]
        j = np.mod(np.rint(lon / dlon).astype(np.int64), self.nlon[ring])
        if tol_deg is None:
            tol_deg = 0.1 * 180.0 / (2 * self.n)
        err_lat = np.abs(self.lats[ring] - lat)
        err_lon = np.abs(((lon - j * dlon) + 180.0) % 360.0 - 180.0)
        bad = (err_lat > tol_deg) | (err_lon > 0.1 * dlon)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} of {lat.size} points are not on {self.name} "
                f"(worst latitude miss {float(err_lat.max()):.4g} deg); pass the right grid"
            )
        return self.offsets[ring] + j


def regular_lat_lon(res_deg: float = 0.1, *, central_longitude: float = 0.0,
                    extent: tuple[float, float, float, float] | None = None):
    """Centres of a regular latitude-longitude grid, north to south and west to east.

    Global by default: 360/res columns starting at ``central_longitude - 180`` and 180/res + 1
    rows from 90 N to 90 S (0.1 degrees gives 3600 x 1801, the grid of ECMWF's 0.1 degree
    products). ``extent = (west, east, south, north)`` gives a box with both edges included.
    """
    if extent is None:
        nlon = int(round(360.0 / res_deg))
        lon = central_longitude - 180.0 + res_deg * np.arange(nlon)
        nlat = int(round(180.0 / res_deg)) + 1
        lat = 90.0 - res_deg * np.arange(nlat)
        return lat, lon
    west, east, south, north = (float(v) for v in extent)
    if east <= west:
        east += 360.0
    lon = west + res_deg * np.arange(int(round((east - west) / res_deg)) + 1)
    lat = north - res_deg * np.arange(int(round((north - south) / res_deg)) + 1)
    return lat, lon


class RingBilinear:
    """Precomputed ring-wise bilinear weights from O<N> to a regular grid.

    ``source_index`` (optional) is the global index of each value that will be passed to
    ``__call__``, for crops of the grid; without it the values must be the whole grid in the
    standard order.
    """

    def __init__(self, grid: OctahedralGrid, lat_t: np.ndarray, lon_t: np.ndarray,
                 source_index: np.ndarray | None = None):
        self.grid = grid
        self.shape = (lat_t.size, lon_t.size)
        self.source_index = None if source_index is None else np.asarray(source_index, dtype=np.int64)
        lats = grid.lats
        n_rings = lats.size

        # bracketing rings per target row (rings are north to south, lats decreasing)
        asc = lats[::-1]
        k = np.searchsorted(asc, lat_t)                       # asc[k-1] < lat <= asc[k]
        south_ring = np.clip(n_rings - k, 0, n_rings - 1)    # ring with latitude < lat (asc[k-1])
        north_ring = np.clip(n_rings - 1 - k, 0, n_rings - 1)
        north_ring = np.where(k >= n_rings, 0, north_ring)
        south_ring = np.where(k <= 0, n_rings - 1, south_ring)
        lat_n, lat_s = lats[north_ring], lats[south_ring]
        span = lat_n - lat_s
        wy = np.where(span > 0, (lat_n - lat_t) / np.where(span > 0, span, 1.0), 0.0)
        wy = np.clip(wy, 0.0, 1.0)                           # 0 -> north ring, 1 -> south ring

        lon_t = np.mod(np.asarray(lon_t, dtype=np.float64), 360.0)
        idx = np.empty((4,) + self.shape, dtype=np.int64)
        wts = np.empty((4,) + self.shape, dtype=np.float32)
        for slot, (rings, w_ring) in enumerate(((north_ring, 1.0 - wy), (south_ring, wy))):
            nl = grid.nlon[rings][:, None].astype(np.float64)
            p = lon_t[None, :] * nl / 360.0
            j0 = np.floor(p)
            wx = p - j0
            j0 = np.mod(j0.astype(np.int64), nl.astype(np.int64))
            j1 = np.mod(j0 + 1, nl.astype(np.int64))
            off = grid.offsets[rings][:, None]
            idx[2 * slot] = off + j0
            idx[2 * slot + 1] = off + j1
            wts[2 * slot] = (w_ring[:, None] * (1.0 - wx)).astype(np.float32)
            wts[2 * slot + 1] = (w_ring[:, None] * wx).astype(np.float32)

        if self.source_index is not None:
            lookup = np.full(grid.size, -1, dtype=np.int64)
            lookup[self.source_index] = np.arange(self.source_index.size)
            idx = lookup[idx]
            # a zero weight on a missing neighbour must not blank the target point
            idx = np.where((idx < 0) & (wts == 0), 0, idx)
            self.missing = (idx < 0).any(axis=0)
            idx = np.where(idx < 0, 0, idx)
        else:
            self.missing = None
        dtype = np.int32 if idx.max(initial=0) < 2**31 else np.int64
        self.idx = idx.astype(dtype)
        self.wts = wts

    def __call__(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32)
        out = np.zeros(self.shape, dtype=np.float32)
        for s in range(4):
            out += self.wts[s] * v[self.idx[s]]
        if self.missing is not None:
            out[self.missing] = np.nan
        return out
