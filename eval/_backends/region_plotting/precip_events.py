"""Find the heaviest-precipitation (date, step) slices in a predictions dir.

Used by the `precip_events` evaluator to drive bbox-cropped local plots of the
hardest precip cases. Pure logic — no plotting, no subprocess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

_FNAME_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc")


@dataclass
class Event:
    nc_path: Path
    date: str
    step: int
    peak_value: float
    lat: float
    lon: float
    bbox: list[float]   # [lat_min, lat_max, lon_min, lon_max]
    label: str


def _parse(path: Path) -> tuple[str, int] | None:
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def find_precip_events(
    predictions_dir,
    *,
    n_events: int,
    dlat: float,
    dlon: float,
    rank_by: str = "truth",
    var: str = "tp",
    member: int = 0,
) -> list[Event]:
    """Rank prediction NCs by max `var` (truth or pred) and return the top-N as Events."""
    predictions_dir = Path(predictions_dir)
    files = sorted(predictions_dir.glob("predictions_*_step*.nc"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    field = "y" if rank_by == "truth" else "y_pred"
    candidates: list[Event] = []
    for f in files:
        parsed = _parse(f)
        if parsed is None:
            continue
        date, step = parsed
        ds = xr.open_dataset(f)
        try:
            ws = list(ds["weather_state"].values)
            if var not in ws:
                continue
            vi = ws.index(var)
            arr = ds[field].values[0, member, :, vi]
            lats = ds["lat_hres"].values
            lons = ds["lon_hres"].values
        finally:
            ds.close()

        gp = int(np.argmax(arr))
        peak = float(arr[gp])
        lat = float(lats[gp])
        lon = float(lons[gp])
        bbox = [
            max(lat - dlat, -90.0), min(lat + dlat, 90.0),
            max(lon - dlon, -180.0), min(lon + dlon, 180.0),
        ]
        candidates.append(Event(
            nc_path=f, date=date, step=step, peak_value=peak,
            lat=lat, lon=lon, bbox=bbox, label="",
        ))

    if not candidates:
        raise FileNotFoundError(
            f"No prediction files in {predictions_dir} contain weather_state {var!r}"
        )

    candidates.sort(key=lambda e: e.peak_value, reverse=True)
    top = candidates[:n_events]
    for rank, e in enumerate(top, start=1):
        e.label = f"event{rank:02d}_{e.date}_step{e.step:03d}"
    return top
