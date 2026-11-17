#!/usr/bin/env python3
"""Grids of the CorrDiff baseline (17 Nov 2026): the E2 box [10, 70] N x [-80, 40] E on regular latitude-longitude grids.

Target grid  864 x 1728 cells of 60/864 = 120/1728 = 0.06944 deg (7.7 km in latitude; O1280 is 0.0703 deg in latitude
             and coarser in longitude away from the equator, so the grid does not smooth the 9 km truth). Both sizes are
             multiples of 16 (the U-Net halves the image 4 times, channel_mult has 5 levels).
Input grid   216 x 432 cells of 0.2778 deg (= 4 target cells; O320 is 0.281 deg in latitude), upsampled x4 in the
             dataset (bilinear with edge extrapolation, PhysicsNeMo hrrr_mini's method) = CorrDiff's own input path:
             the coarse field interpolated onto the target grid.
Cell centres, north to south and west to east. Longitudes in -180..180.

Regridding: unstructured -> regular by inverse squared chord distance over the 4 nearest points (qm_lib.idw_weights,
same as the QM baseline); regular -> evaluation points by bilinear interpolation inside the grid (points outside the
cell-centre hull of the target grid get NaN).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qm"))
from qm_lib import idw_weights  # noqa: E402

BOX = (10.0, 70.0, -80.0, 40.0)     # south, north, west, east
NY, NX = 864, 1728                  # target
FACTOR = 4
NYC, NXC = NY // FACTOR, NX // FACTOR  # input


def axes(ny, nx):
    s, n, w, e = BOX
    lat = n - (np.arange(ny) + 0.5) * (n - s) / ny
    lon = w + (np.arange(nx) + 0.5) * (e - w) / nx
    return lat, lon


def mesh(ny, nx):
    lat, lon = axes(ny, nx)
    return np.repeat(lat, nx), np.tile(lon, ny)


def wrap180(lon):
    return (np.asarray(lon, np.float64) + 180.0) % 360.0 - 180.0


def idw_to_grid(src_lat, src_lon, ny, nx, k=4):
    """(idx (ny*nx, k), w (ny*nx, k)) from unstructured source points to the regular grid."""
    la, lo = mesh(ny, nx)
    return idw_weights(np.asarray(src_lat, np.float64), wrap180(src_lon), la, lo, k=k)


def bilinear_weights(lat, lon, ny=NY, nx=NX):
    """Points -> (i0, j0, wy, wx, inside): bilinear interpolation on the regular grid (cell centres). Points outside the
    hull of the cell centres get inside=False (their value is NaN)."""
    glat, glon = axes(ny, nx)
    lat = np.asarray(lat, np.float64)
    lon = wrap180(lon)
    fy = (glat[0] - lat) / (glat[0] - glat[1])          # fractional row index (rows go north to south)
    fx = (lon - glon[0]) / (glon[1] - glon[0])
    inside = (fy >= 0) & (fy <= ny - 1) & (fx >= 0) & (fx <= nx - 1)
    i0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
    j0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
    return i0, j0, (fy - i0).astype(np.float32), (fx - j0).astype(np.float32), inside


def bilinear_apply(img, bw):
    """img (..., ny, nx) -> (..., n_points); NaN outside."""
    i0, j0, wy, wx, inside = bw
    a = img[..., i0, j0]
    b = img[..., i0, j0 + 1]
    c = img[..., i0 + 1, j0]
    d = img[..., i0 + 1, j0 + 1]
    out = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
    out = np.asarray(out, np.float32)
    out[..., ~inside] = np.nan
    return out


def check():
    """Self-test: a linear field is reproduced exactly by bilinear_apply inside the hull."""
    lat, lon = axes(NY, NX)
    img = (2.0 * lat[:, None] + 0.5 * lon[None, :]).astype(np.float64)
    rng = np.random.default_rng(0)
    pl = rng.uniform(lat[-1], lat[0], 1000)
    po = rng.uniform(lon[0], lon[-1], 1000)
    v = bilinear_apply(img, bilinear_weights(pl, po))
    assert np.allclose(v, 2.0 * pl + 0.5 * po, atol=1e-3), np.abs(v - (2.0 * pl + 0.5 * po)).max()
    out = bilinear_apply(img, bilinear_weights(np.array([75.0, 40.0]), np.array([0.0, 50.0])))
    assert np.isnan(out).all()
    assert NY % 16 == 0 and NX % 16 == 0 and NYC * FACTOR == NY
    print("cd_grid check OK", flush=True)


if __name__ == "__main__":
    check()
