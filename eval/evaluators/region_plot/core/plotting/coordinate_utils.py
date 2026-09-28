"""Coordinate and region helpers for region plotting."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from .config import KNOWN_REGION_BOXES


def _coord_name_for_array(ds: xr.Dataset, da: xr.DataArray, axis: str) -> str:
    """Resolve the coordinate name for *axis* used by *da*."""
    attr_name = da.attrs.get(axis)
    if attr_name in ds.variables or attr_name in ds.coords:
        return str(attr_name)

    dim_candidates = {
        "grid_point_hres": f"{axis}_hres",
        "grid_point_lres": f"{axis}_lres",
    }
    for dim_name, coord_name in dim_candidates.items():
        if dim_name in da.dims and (coord_name in ds.variables or coord_name in ds.coords):
            return coord_name

    raise KeyError(f"Could not infer {axis} coordinate for {da.name}")


def resolve_region_box(region_box: str | Sequence[float]) -> list[float]:
    """Resolve a region name or explicit bounds into ``[lat_min, lat_max, lon_min, lon_max]``."""
    if isinstance(region_box, str):
        if region_box in KNOWN_REGION_BOXES:
            return list(KNOWN_REGION_BOXES[region_box])
        raise ValueError(f"Bounding box '{region_box}' is not predefined.")
    values = [float(value) for value in region_box]
    if len(values) != 4:
        raise ValueError("Bounding box list must have exactly 4 elements.")
    return values


def get_region_ds(ds: xr.Dataset, region_box: str | Sequence[float] = "default") -> xr.Dataset:
    """Subset a dataset to a named plotting region or explicit lat/lon bounds."""
    lat_min, lat_max, lon_min, lon_max = resolve_region_box(region_box)

    mask_hres = (
        (ds["lon_hres"] >= lon_min)
        & (ds["lon_hres"] <= lon_max)
        & (ds["lat_hres"] >= lat_min)
        & (ds["lat_hres"] <= lat_max)
    )
    region_hres = ds.isel(grid_point_hres=np.flatnonzero(np.asarray(mask_hres.values)))

    if "grid_point_lres" in region_hres.dims and ("lon_lres" in ds.variables or "lon_lres" in ds.coords):
        mask_lres = (
            (ds["lon_lres"] >= lon_min)
            & (ds["lon_lres"] <= lon_max)
            & (ds["lat_lres"] >= lat_min)
            & (ds["lat_lres"] <= lat_max)
        )
        region_ds = region_hres.isel(grid_point_lres=np.flatnonzero(np.asarray(mask_lres.values)))
    else:
        region_ds = region_hres

    region_ds.attrs["region"] = [lat_min, lat_max, lon_min, lon_max]
    return region_ds


def infer_grid_type(ds: xr.Dataset) -> str:
    """Infer the grid identifier from dataset attributes or high-resolution size."""
    grid = str(getattr(ds, "attrs", {}).get("grid", "")).strip()
    if grid:
        return grid
    hres = int(ds.sizes.get("grid_point_hres", 0))
    if hres >= 26_000_000:
        return "O2560"
    if hres >= 6_000_000:
        return "O1280"
    return ""


def default_region_for_grid(grid: str) -> str:
    """Return the default plotting region for a grid name."""
    return "amazon_forest"


_LEGACY_O320_PLOTTER_REGIONS = [
    "amazon_forest", "european_arctic", "himalayas", "rocky_mountains",
    "west_sahara", "pyrenees_alpes", "eastern_us", "central_africa",
]

_LEGACY_O1280_PLOTTER_REGIONS = [
    "rocky_mountains_central", "rocky_mountains_north", "rocky_mountains_south",
    "amazon_forest_central", "amazon_forest_west", "amazon_forest_east",
    "southeast_asia_central", "southeast_asia_mainland", "southeast_asia_maritime",
    "west_sahara_central", "west_sahara_coastal", "west_sahara_east",
    "himalayas_central", "himalayas_west", "himalayas_east",
    "greatbarrier_reef_central", "greatbarrier_reef_north", "greatbarrier_reef_south",
    "eastern_us_central", "eastern_us_north", "eastern_us_south",
    "central_africa_congo", "central_africa_north", "central_africa_south",
]


def legacy_plotter_regions_for_grid(grid: str) -> list[str]:
    """Return the legacy ``LocalInferencePlotter`` region list for a grid."""
    grid = str(grid).strip()
    if grid == "O320":
        return list(_LEGACY_O320_PLOTTER_REGIONS)
    if grid == "O1280":
        return list(_LEGACY_O1280_PLOTTER_REGIONS)
    raise ValueError(f"Unsupported grid type: {grid}. Please ensure grid is O320 or O1280.")
