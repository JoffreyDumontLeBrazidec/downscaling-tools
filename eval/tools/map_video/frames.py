"""Read one field per frame from manual_inference prediction files.

A prediction file (``manual_inference.prediction.dataset.build_predictions_dataset``) holds
``y_pred`` (sample, ensemble_member, grid_point_hres, weather_state), optionally ``x_interp``
(the input interpolated to the output grid, same dims) and ``y`` (the truth), with
``lon_hres``/``lat_hres``. One file is one (base date, lead). Each frame of a video is one file;
the panels of a frame are sources of that file:

* ``pred``  -> ``y_pred``   (the downscaling model)
* ``input`` -> ``x_interp`` (the driving forecast, interpolated to the output grid)
* ``truth`` -> ``y``        (the high-resolution target)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SOURCE_VARS = {"pred": "y_pred", "input": "x_interp", "truth": "y"}
SOURCE_TITLES = {"pred": "Downscaled", "input": "Input (interpolated)", "truth": "Truth"}

_STEP_RES = (re.compile(r"step_?(\d{1,4})", re.I), re.compile(r"(?:^|_)L(\d{2,4})(?:_|\.|$)"),
             re.compile(r"lead_?(\d{1,4})", re.I))
_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?:T?(\d{2}))?(?!\d)")


@dataclass(frozen=True)
class FrameFile:
    path: Path
    init: datetime
    step: int

    @property
    def valid(self) -> datetime:
        return self.init + timedelta(hours=self.step)


def _parse_init(ds, path: Path, sample: int) -> datetime:
    for key in ("init_date", "date"):
        if key not in ds:
            continue
        try:
            raw = np.asarray(ds[key].values).reshape(-1)[sample]
            if np.issubdtype(np.asarray(raw).dtype, np.datetime64):
                return datetime.utcfromtimestamp(raw.astype("datetime64[s]").astype(int))
            text = str(raw.decode() if isinstance(raw, bytes) else raw)
            m = _DATE_RE.search(text.replace("-", "").replace(" ", "T"))
            if m:
                return datetime.strptime(m.group(1) + (m.group(2) or "00"), "%Y%m%d%H")
        except Exception:
            pass
    m = _DATE_RE.search(path.name)
    if m:
        return datetime.strptime(m.group(1) + (m.group(2) or "00"), "%Y%m%d%H")
    raise ValueError(f"{path}: no base date in the 'date' variable or the file name")


def _parse_step(ds, path: Path) -> int:
    if "lead_step_hours" in ds:
        try:
            raw = np.asarray(ds["lead_step_hours"].values).reshape(-1)[0]
            if np.issubdtype(np.asarray(raw).dtype, np.timedelta64):   # xarray decodes units "hours"
                return int(raw / np.timedelta64(1, "h"))
            return int(raw)
        except Exception:
            pass
    for key in ("step", "lead_time", "lead_hours", "step_hours"):
        if key in getattr(ds, "attrs", {}):
            return int(ds.attrs[key])
    for rx in _STEP_RES:
        m = rx.search(path.name)
        if m:
            return int(m.group(1))
    raise ValueError(f"{path}: no lead time (file name step<NNN>/L<NNN> or attribute 'step')")


def scan(paths, *, sample: int = 0) -> list[FrameFile]:
    """Frame files sorted by valid time (then lead), with base date and lead read from each file."""
    import xarray as xr

    out = []
    for p in paths:
        p = Path(p)
        with xr.open_dataset(p) as ds:
            out.append(FrameFile(path=p, init=_parse_init(ds, p, sample), step=_parse_step(ds, p)))
    return sorted(out, key=lambda f: (f.valid, f.step))


def read_coords(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return (np.asarray(ds["lat_hres"].values, dtype=np.float64),
                np.asarray(ds["lon_hres"].values, dtype=np.float64))


def _select(da, variable: str, member: int, sample: int):
    if "sample" in da.dims:
        da = da.isel(sample=sample)
    if "ensemble_member" in da.dims:
        da = da.isel(ensemble_member=member) if da.sizes["ensemble_member"] > 1 else da.isel(ensemble_member=0)
    names = [str(v) for v in da["weather_state"].values]
    want = [variable, f"{variable}_sfc"]
    hit = next((n for n in names if n in want), None)
    if hit is None:
        raise KeyError(f"{variable} not in weather_state {names}")
    return da.isel(weather_state=names.index(hit))


def read_field(path: Path, source: str, variable: str, *, member: int = 0, sample: int = 0) -> np.ndarray:
    """Native-unit values of ``variable`` from one source of one file, on the file's hres points.

    ``variable='10ff'`` (10 m wind speed) is computed from ``10u`` and ``10v``.
    """
    import xarray as xr

    name = SOURCE_VARS[source]
    with xr.open_dataset(path) as ds:
        if name not in ds:
            raise KeyError(f"{path.name} has no {name} ({source}); present: {sorted(ds.data_vars)}")
        da = ds[name]
        if variable == "10ff":
            u = np.asarray(_select(da, "10u", member, sample).values, dtype=np.float32)
            v = np.asarray(_select(da, "10v", member, sample).values, dtype=np.float32)
            return np.hypot(u, v)
        return np.asarray(_select(da, variable, member, sample).values, dtype=np.float32)


def member_label(path: Path, member: int) -> str:
    """The ensemble-member coordinate value at position ``member`` (for the title)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if "ensemble_member" in ds.coords and ds.sizes.get("ensemble_member", 0) > member:
            return str(ds["ensemble_member"].values[member])
    return str(member)
