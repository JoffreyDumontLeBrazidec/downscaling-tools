"""PhysicsNeMo CorrDiff dataset for the E2-v4 export (cd_export.py). Used as dataset.type "<this file>::E2v4Box".

Same contract as PhysicsNeMo's hrrr_mini reader (examples/weather/corrdiff/datasets/hrrrmini.py, tag v2.2.2):
__getitem__ -> (y, x) normalised; x = the coarse input upsampled x4 to the target grid (bilinear with edge
extrapolation, the same numba routine) followed by the invariants. Normalisation = per-channel mean / std of the
training rows (OUT.stats.json). No lead-time conditioning (the rows mix leads 12-72 h; GL512 has none either).

Options (all default off; the 46-channel input of the first regression run is unchanged without them):
  time_inputs: true  appends 5 per-row channels after the 46 (TIME_INPUTS: cos/sin julian day, cos/sin local time,
                     insolation), computed on the 864 x 1728 grid from the row's valid time with the earthkit-data
                     forcing formulas (time_fields), not normalised (in [-1, 1] / [0, 1], as GL512 is given them). The
                     stored invariants cos_local_time / sin_local_time stay as they are (constant, = -cos/-sin lon).
  spread: true       with max_rows: the max_rows rows evenly spread over the whole split (deterministic) instead of
                     the first max_rows (the first 64 valid rows are 29 Jun - 3 Aug 2022 only).
"""
from __future__ import annotations

import csv
import datetime
import json
import math
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

try:
    from datasets.base import ChannelMetadata, DownscalingDataset   # inside examples/weather/corrdiff
except ImportError:  # self-test outside PhysicsNeMo
    class DownscalingDataset:
        pass

    class ChannelMetadata:
        def __init__(self, name, level="", auxiliary=False):
            self.name, self.level, self.auxiliary = name, level, auxiliary

try:
    from numba import jit, prange
except ImportError:  # pragma: no cover
    def jit(**_):
        return lambda f: f
    prange = range


TIME_INPUTS = ["cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time", "insolation"]


def time_fields(valid: datetime.datetime, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(5, *lat.shape) float32, TIME_INPUTS order, at valid time `valid` (UTC, naive). Formulas of earthkit-data
    sources/forcings.py (julian_day, local_time, insolation = cos_solar_zenith_angle) and earthkit-meteo
    solar.cos_solar_zenith_angle (declination / time-correction series, clipped at 0, integer hour), which anemoi-datasets
    uses for its computed forcings (the stores GL512 reads)."""
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    d = valid - datetime.datetime(valid.year, 1, 1)
    jd = d.days + d.seconds / 86400.0
    a = jd / 365.25 * np.pi * 2
    d = valid - datetime.datetime(valid.year, valid.month, valid.day)
    lt = (lon / 360.0 * 24.0 + (d.days + d.seconds / 86400.0) * 24) % 24
    b = lt / 24 * np.pi * 2
    decl = np.deg2rad(0.396372 - 22.91327 * np.cos(a) + 4.025430 * np.sin(a) - 0.387205 * np.cos(2 * a)
                      + 0.051967 * np.sin(2 * a) - 0.154527 * np.cos(3 * a) + 0.084798 * np.sin(3 * a))
    tc = 0.004297 + 0.107029 * np.cos(a) - 1.837877 * np.sin(a) - 0.837378 * np.cos(2 * a) - 2.340475 * np.sin(2 * a)
    rlat = np.deg2rad(lat)
    csza = np.sin(decl) * np.sin(rlat) + np.cos(decl) * np.cos(rlat) * np.cos(np.deg2rad((valid.hour - 12) * 15 + lon + tc))
    one = np.ones_like(lat)
    return np.stack([np.cos(a) * one, np.sin(a) * one, np.cos(b), np.sin(b), np.clip(csza, 0.0, None)]).astype(np.float32)


def spread_index(n: int, k: int) -> np.ndarray:
    """k indices evenly spread over range(n), first and last included, deterministic."""
    return np.unique(np.round(np.linspace(0, n - 1, min(k, n))).astype(np.int64))


class E2v4Box(DownscalingDataset):
    def __init__(self, data_path: str, stats_path: str, input_variables: Union[List[str], None] = None,
                 output_variables: Union[List[str], None] = None, invariant_variables: Union[List[str], None] = None,
                 split: str = "train", max_rows: int = 0, time_inputs: bool = False, spread: bool = False):
        import zarr
        self.root = zarr.open_group(str(data_path), mode="r")
        at = self.root.attrs
        self.all_in, self.all_out, self.all_inv = list(at["input_variables"]), list(at["output_variables"]), list(at["invariant_variables"])
        self.input_variables = list(input_variables) if input_variables else self.all_in
        self.output_variables = list(output_variables) if output_variables else self.all_out
        self.invariant_variables = list(invariant_variables) if invariant_variables is not None else self.all_inv
        self.ii = [self.all_in.index(v) for v in self.input_variables]
        self.oi = [self.all_out.index(v) for v in self.output_variables]
        inv = np.asarray(self.root["invariant"][:], np.float32)
        self.invariants = inv[[self.all_inv.index(v) for v in self.invariant_variables]]
        self.factor = int(at["factor"])
        self.img_shape = (int(at["ny"]), int(at["nx"]))
        rows = list(csv.DictReader(open(Path(str(data_path)).with_suffix(".rows.csv"))))
        sel = [r for r in rows if r["split"] == split]
        if max_rows:
            sel = [sel[i] for i in spread_index(len(sel), max_rows)] if spread else sel[:max_rows]
        if not sel:
            raise ValueError(f"no rows in split {split}")
        self.rows = np.array([int(r["row"]) for r in sel], np.int64)
        self.valid = [datetime.datetime.fromisoformat(r["valid"]) for r in sel]
        self.steps = [int(r["step"]) for r in sel]
        self.time_inputs = bool(time_inputs)
        self.time_variables = list(TIME_INPUTS) if self.time_inputs else []
        nt = len(self.time_variables)
        st = json.load(open(stats_path))
        # time inputs: mean 0, std 1 = left as they are by (de)normalize_input
        self.input_mean = np.array([st["input"][v]["mean"] for v in self.input_variables] +
                                   [st["invariant"][v]["mean"] for v in self.invariant_variables] + [0.0] * nt,
                                   np.float32)[:, None, None]
        self.input_std = np.array([st["input"][v]["std"] for v in self.input_variables] +
                                  [st["invariant"][v]["std"] for v in self.invariant_variables] + [1.0] * nt,
                                  np.float32)[:, None, None]
        self.output_mean = np.array([st["output"][v]["mean"] for v in self.output_variables], np.float32)[:, None, None]
        self.output_std = np.array([st["output"][v]["std"] for v in self.output_variables], np.float32)[:, None, None]
        lat0, lon0 = float(at["lat0"]), float(at["lon0"])
        s, n, w, e = at["box"]
        ny, nx = self.img_shape
        self._lat = np.repeat((n - (np.arange(ny) + 0.5) * (n - s) / ny)[:, None], nx, 1)
        self._lon = np.repeat((w + (np.arange(nx) + 0.5) * (e - w) / nx)[None, :], ny, 0)
        assert abs(self._lat[0, 0] - lat0) < 1e-9 and abs(self._lon[0, 0] - lon0) < 1e-9

    def coarse_input(self, x_coarse: np.ndarray, valid: Union[datetime.datetime, None] = None) -> np.ndarray:
        """(C_all_in, NYC, NXC) physical -> normalised network input (C_in + C_inv [+ 5 time inputs], NY, NX).
        valid (UTC) is required with time_inputs."""
        x = self.upsample(np.ascontiguousarray(x_coarse[self.ii], np.float32))
        parts = [x, self.invariants]
        if self.time_inputs:
            if valid is None:
                raise ValueError("time_inputs: coarse_input needs the valid time")
            parts.append(time_fields(valid, self._lat, self._lon))
        return self.normalize_input(np.concatenate(parts, axis=0))

    def __getitem__(self, idx):
        r = int(self.rows[idx])
        x = self.coarse_input(np.asarray(self.root["input"][r], np.float32), self.valid[idx])
        y = self.normalize_output(np.asarray(self.root["output"][r], np.float32)[self.oi])
        return (y, x)

    def __len__(self):
        return len(self.rows)

    def longitude(self) -> np.ndarray:
        return self._lon

    def latitude(self) -> np.ndarray:
        return self._lat

    def input_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v) for v in self.input_variables] + \
               [ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables] + \
               [ChannelMetadata(name=v, auxiliary=True) for v in self.time_variables]

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> List:
        from helpers.train_helpers import _convert_datetime_to_cftime
        return [_convert_datetime_to_cftime(t) for t in self.valid]

    def image_shape(self) -> Tuple[int, int]:
        return self.img_shape

    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x):
        return x * self.input_std + self.input_mean

    def normalize_output(self, x):
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x):
        return x * self.output_std + self.output_mean

    def upsample(self, x):
        y = np.empty((x.shape[0], x.shape[1] * self.factor, x.shape[2] * self.factor), dtype=np.float32)
        _zoom_extrapolate(x, y, self.factor)
        return y


@jit(nopython=True)
def _zoom_extrapolate(x, y, factor):
    """Copy of PhysicsNeMo hrrrmini._zoom_extrapolate (v2.2.2): bilinear zoom with linear extrapolation at the edges."""
    s = 1 / factor
    for k in prange(y.shape[0]):
        for iy in range(y.shape[1]):
            ix = (iy + 0.5) * s - 0.5
            ix0 = int(math.floor(ix))
            ix0 = max(0, min(ix0, x.shape[1] - 2))
            ix1 = ix0 + 1
            for jy in range(y.shape[2]):
                jx = (jy + 0.5) * s - 0.5
                jx0 = int(math.floor(jx))
                jx0 = max(0, min(jx0, x.shape[2] - 2))
                jx1 = jx0 + 1
                x00 = x[k, ix0, jx0]
                x01 = x[k, ix0, jx1]
                x10 = x[k, ix1, jx0]
                x11 = x[k, ix1, jx1]
                djx = jx - jx0
                x0 = x00 + djx * (x01 - x00)
                x1 = x10 + djx * (x11 - x10)
                y[k, iy, jy] = x0 + (ix - ix0) * (x1 - x0)
