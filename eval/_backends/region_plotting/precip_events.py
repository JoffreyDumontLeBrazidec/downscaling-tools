"""Find the heaviest-precipitation (date, step) slices in a predictions dir.

Used by the `precip_events` evaluator to drive bbox-cropped local plots of the
hardest precip cases. Pure logic — no plotting, no subprocess.

When ranking by truth and the predictions carry no tp truth (the o1280->o2560
main-lane case), the truth comes from the per-date GRIB template instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from eval._backends.precip.sources import PrecipTruthSource

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
    truth_grib_tpl: str = "",
) -> list[Event]:
    """Rank prediction NCs by max `var` (truth or pred) and return the top-N as Events."""
    predictions_dir = Path(predictions_dir)
    files = sorted(predictions_dir.glob("predictions_*_step*.nc"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    truth_src: PrecipTruthSource | None = None
    truth_mode: str | None = None
    candidates: list[Event] = []
    for f in files:
        parsed = _parse(f)
        if parsed is None:
            continue
        date, step = parsed
        ds = xr.open_dataset(f)
        try:
            ws = [str(s) for s in ds["weather_state"].values]
            if var not in ws:
                continue
            vi = ws.index(var)
            lats = ds["lat_hres"].values
            lons = ds["lon_hres"].values
            if rank_by == "truth":
                if truth_mode is None:
                    probe = ds["y"][0, member, :100_000].values[:, vi]
                    truth_mode = "embedded-y" if np.isfinite(probe).mean() > 0.99 \
                        else "grib"
                if truth_mode == "grib":
                    if not truth_grib_tpl:
                        raise RuntimeError(
                            f"rank_by=truth but predictions carry no {var} truth "
                            "and no truth_grib_tpl was configured (lane precip "
                            "block)")
                    truth_src = truth_src or PrecipTruthSource(truth_grib_tpl, var=var)
                    arr = truth_src.load(date, step)
                    truth_src.verify_grid(lats, lons)
                else:
                    arr = ds["y"][0, member].values[:, vi]
            else:
                arr = ds["y_pred"][0, member].values[:, vi]
        finally:
            ds.close()

        gp = int(np.nanargmax(arr))
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
