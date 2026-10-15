"""tc_structure on O320 points (o96->o320 big eval): the bin-coverage rule needs the O320 cell.

A Rankine vortex (RMW 75 km, 40 m/s) on the real O320 points of the Franklin box. With the
default parameters (O1280 cell area, 10 km bins) almost every radial bin holds fewer points
than half of what an O1280 grid would put there, so the profile is empty and RMW is lost.
With ``params: {cell_area_km2: 1211.2, dr_km: 20}`` (the O320 cell, 4 pi a^2 / 421120)
RMW, the maximum wind and the vorticity inside 100 and 200 km are recovered.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from eval.evaluators.tc_structure.core import (
    StructureParams, gc_distance_km, measure_structure, o1280_points_in_box,
    outward_bearing_at_point,
)

BOX = (15.0, 38.0, -78.0, -58.0)
CLAT, CLON, RMW, VMAX = 25.0, -68.0, 75.0, 40.0
O320_CELL_KM2 = 4.0 * math.pi * 6371.0088 ** 2 / 421120


def _vortex():
    lat, lon = o1280_points_in_box(*BOX, n_lat=320)
    r = gc_distance_km(CLAT, CLON, lat, lon)
    beta = outward_bearing_at_point(CLAT, CLON, lat, lon)
    V = np.where(r < RMW, VMAX * r / RMW, VMAX * RMW / np.maximum(r, 1e-9))
    return lat, lon, -V * np.cos(beta), V * np.sin(beta)


def test_default_o1280_bins_lose_the_profile_on_o320():
    lat, lon, u, v = _vortex()
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON)
    assert sc["n_valid_bins"] < 10 or not np.isfinite(sc["rmw_km"]) or abs(sc["rmw_km"] - RMW) > 10


def test_o320_cell_and_20km_bins_recover_the_vortex():
    lat, lon, u, v = _vortex()
    p = StructureParams(cell_area_km2=O320_CELL_KM2, dr_km=20.0)
    sc, _ = measure_structure(lat, lon, u, v, CLAT, CLON, p)
    assert abs(sc["rmw_km"] - RMW) <= 20.0
    assert abs(sc["maxwind300_ms"] / VMAX - 1.0) <= 0.15
    for rk in (100.0, 200.0):     # mean vorticity inside r of a Rankine vortex: 2 vt(r) / r
        assert sc[f"zeta{int(rk)}_s"] == pytest.approx(2 * (VMAX * RMW / rk) / (rk * 1e3), rel=0.10)
    assert O320_CELL_KM2 == pytest.approx(1211.2, abs=0.1)
