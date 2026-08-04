"""Shared data structures for TC evaluation. Zero internal imports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import hashlib

import numpy as np

SupportMode = Literal["native", "regridded", "both"]
FORECAST_STEP_COUNT = 5


@dataclass(frozen=True)
class BoundingBox:
    """Spatial crop region. This is the EVALUATION box, not the data extent."""

    north: float
    south: float
    east: float
    west: float

    @property
    def crosses_dateline(self) -> bool:
        return self.east < self.west


@dataclass(frozen=True)
class CurveVectors:
    msl: np.ndarray
    wind: np.ndarray
    # Populated by the loaders. Directly constructed test/analysis curves remain
    # compatible, but the evaluator rejects them before writing comparison stats.
    support_mode: str = "unknown"
    support_signature: str = "unknown"


@dataclass(frozen=True)
class StructuredGrid:
    lat_axis: np.ndarray
    lon_axis: np.ndarray
    point_indices: np.ndarray


def curve_support_signature(support_mode: str, lon: np.ndarray, lat: np.ndarray) -> str:
    """Return a stable identity for the spatial support of a curve."""
    lon = np.ascontiguousarray(np.asarray(lon, dtype=np.float64).reshape(-1))
    lat = np.ascontiguousarray(np.asarray(lat, dtype=np.float64).reshape(-1))
    if lon.shape != lat.shape:
        raise ValueError("Curve support longitude/latitude arrays must have equal size")
    digest = hashlib.sha256(lon.tobytes() + lat.tobytes()).hexdigest()[:16]
    return f"{support_mode}:{lon.size}:{digest}"


def step_to_index(step: int) -> int:
    idx = (int(step) // 24) - 1
    if idx < 0 or idx > 4:
        raise ValueError(f"Unsupported step={step}; expected one of 24,48,72,96,120")
    return idx
