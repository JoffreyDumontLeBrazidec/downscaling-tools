"""PhysicsNeMo CorrDiff dataset for the E2-v4 export (cd_export.py). Used as dataset.type "<this file>::E2v4Box".

Same contract as PhysicsNeMo's hrrr_mini reader (examples/weather/corrdiff/datasets/hrrrmini.py, tag v2.2.2):
__getitem__ -> (y, x) normalised; x = the coarse input upsampled x4 to the target grid (bilinear with edge
extrapolation, the same numba routine) followed by the invariants. Normalisation = per-channel mean / std of the
training rows (OUT.stats.json). No lead-time conditioning (the rows mix leads 12-72 h; GL512 has none either).
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


class E2v4Box(DownscalingDataset):
    def __init__(self, data_path: str, stats_path: str, input_variables: Union[List[str], None] = None,
                 output_variables: Union[List[str], None] = None, invariant_variables: Union[List[str], None] = None,
                 split: str = "train", max_rows: int = 0):
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
            sel = sel[:max_rows]
        if not sel:
            raise ValueError(f"no rows in split {split}")
        self.rows = np.array([int(r["row"]) for r in sel], np.int64)
        self.valid = [datetime.datetime.fromisoformat(r["valid"]) for r in sel]
        self.steps = [int(r["step"]) for r in sel]
        st = json.load(open(stats_path))
        self.input_mean = np.array([st["input"][v]["mean"] for v in self.input_variables] +
                                   [st["invariant"][v]["mean"] for v in self.invariant_variables], np.float32)[:, None, None]
        self.input_std = np.array([st["input"][v]["std"] for v in self.input_variables] +
                                  [st["invariant"][v]["std"] for v in self.invariant_variables], np.float32)[:, None, None]
        self.output_mean = np.array([st["output"][v]["mean"] for v in self.output_variables], np.float32)[:, None, None]
        self.output_std = np.array([st["output"][v]["std"] for v in self.output_variables], np.float32)[:, None, None]
        lat0, lon0 = float(at["lat0"]), float(at["lon0"])
        s, n, w, e = at["box"]
        ny, nx = self.img_shape
        self._lat = np.repeat((n - (np.arange(ny) + 0.5) * (n - s) / ny)[:, None], nx, 1)
        self._lon = np.repeat((w + (np.arange(nx) + 0.5) * (e - w) / nx)[None, :], ny, 0)
        assert abs(self._lat[0, 0] - lat0) < 1e-9 and abs(self._lon[0, 0] - lon0) < 1e-9

    def coarse_input(self, x_coarse: np.ndarray) -> np.ndarray:
        """(C_all_in, NYC, NXC) physical -> normalised network input (C_in + C_inv, NY, NX)."""
        x = self.upsample(np.ascontiguousarray(x_coarse[self.ii], np.float32))
        return self.normalize_input(np.concatenate([x, self.invariants], axis=0))

    def __getitem__(self, idx):
        r = int(self.rows[idx])
        x = self.coarse_input(np.asarray(self.root["input"][r], np.float32))
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
               [ChannelMetadata(name=v, auxiliary=True) for v in self.invariant_variables]

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
