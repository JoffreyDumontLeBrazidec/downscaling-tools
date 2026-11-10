"""The input ensemble on its own grid (owner, 9 Nov 2026, standing rule 41).

Prediction files carry the model draws ``y_pred`` and the truth ``y`` on the output grid (``lat_hres``/``lon_hres``),
the input interpolated to the output grid ``x_interp`` (sometimes per member ``x_interp_members``) and, in newer
files, the NATIVE input ``x`` (sample, ensemble_member, grid_point_lres, weather_state) on ``lat_lres``/``lon_lres``.
Rule 41: the input is scored on its own grid, never interpolated to the output grid first. Interpolation smooths
the members, narrows their spread by 3-7 % and flatters the model (6 h tp at 4 km: +5 % against the interpolated
input, 0 % against the native one; docs o2560 note s19.7). ``x_interp`` stays legitimate only to DEFINE the
residual (``y - x_interp``, ``y_pred - x_interp``) that some evaluators study.

Method: for every verification point of the output grid, the input value is the value of the NEAREST NATIVE input
point (``nearest_native_index``; great-circle nearest neighbour through a 3-D unit-vector cKDTree, as ``xyz`` in
``obs_points/core/stations.py``). This is ECMWF quaver's method (the nearest native grid point to each
verification location) extended to a dense set of verification points. It keeps every native value unsmoothed
(no averaging, no interpolation: each output point carries one native value), and it keeps the verification
points, the domains and the area weights identical for every row (model, truth, input), so the rows stay
comparable point for point.

Detection follows commit 617d98a (obs_points, box_spectra): a file has a native input when it carries ``x`` and
``lat_lres``. Without it, readers fall back to ``x_interp`` (or what the evaluator used before) and record the
source as ``SOURCE_INTERP``; evaluators log a WARNING (``warn_fallback``) and write ``input_source`` into their
run_meta / summary JSON. A config key ``native_input`` (default True) set to False reproduces the old tables.

The nearest-point index is memoised in-process and cached on disk as ``.npz``, keyed by a hash of the two
coordinate sets, in ``$EVAL_NATIVE_INDEX_CACHE`` (default ``~/.cache/downscaling-tools/native_index``). An
unwritable or unreadable cache directory never fails a run; the index is then recomputed.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

SOURCE_NATIVE = "x (native input grid)"
SOURCE_INTERP = "x_interp (input interpolated to the output grid; no native x in the file)"
CACHE_ENV = "EVAL_NATIVE_INDEX_CACHE"
DEFAULT_CACHE_DIR = "~/.cache/downscaling-tools/native_index"
_MEMO: dict[str, np.ndarray] = {}
_MEMO_MAX = 8


def interp_source(variable: str = "x_interp", reason: str = "no native x in the file") -> str:
    """Provenance string of an input row read from an interpolated variable (``SOURCE_INTERP`` by default)."""
    return f"{variable} (input interpolated to the output grid; {reason})"


def grib_nn_source(var: str = "tp") -> str:
    """Provenance string of the precipitation evaluators' GRIB input (``LresInterpBaseline``): the driving model's
    member field at the nearest native point, the same method as ``nearest_native_index``."""
    return f"driver member {var} from the input GRIB at the nearest native point (native input grid)"


def has_native_input(ds) -> bool:
    """True when the file (netCDF4 or xarray Dataset) carries the native input ``x`` on ``lat_lres``/``lon_lres``."""
    names = ds.variables
    return "x" in names and "lat_lres" in names and "lon_lres" in names


def is_interpolated(source: str) -> bool:
    return "interpolated" in str(source)


def warn_fallback(logger: logging.Logger, evaluator: str, source: str) -> str:
    """Log the rule-41 WARNING for an input row that is not read from the native grid; returns the line."""
    line = (f"{evaluator}: input row from {source}; rule 41 says the input is scored on its own grid, so this "
            "input row is smoothed (narrower spread) and not quotable against the model")
    logger.warning(line)
    return line


# ----------------------------------------------------------------------------- nearest native point

def _xyz(lat, lon) -> np.ndarray:
    """Unit-sphere coordinates, so that the nearest-neighbour search ignores longitude wrapping."""
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _key(*arrays: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=20)
    for a in arrays:
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR).expanduser()


def _load_cached(path: Path, n_out: int, n_in: int) -> np.ndarray | None:
    try:
        with np.load(path) as z:
            idx = np.asarray(z["idx"], dtype=np.int64)
            if idx.shape == (n_out,) and int(z["n_in"]) == n_in and (n_out == 0 or idx.max() < n_in):
                return idx
    except Exception:  # noqa: BLE001 - a bad cache file is recomputed, never fatal
        LOG.debug("native_input: unreadable index cache %s", path, exc_info=True)
    return None


def _save_cached(path: Path, idx: np.ndarray, n_in: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        dtype = np.int32 if n_in < 2**31 else np.int64
        np.savez(tmp, idx=idx.astype(dtype), n_in=np.int64(n_in))
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - an unwritable cache never fails a run
        LOG.debug("native_input: could not write index cache %s", path, exc_info=True)


def nearest_native_index(lat_lres, lon_lres, lat_out, lon_out) -> np.ndarray:
    """For every output point, the index of the nearest native input point (int64, shape of ``lat_out`` flat).

    Great-circle nearest neighbour (3-D unit vectors, cKDTree). Memoised in-process and cached on disk under
    ``cache_dir()`` keyed by a hash of the four coordinate arrays; the cache never fails a run."""
    la_in = np.asarray(lat_lres, np.float64).reshape(-1)
    lo_in = np.asarray(lon_lres, np.float64).reshape(-1)
    la_out = np.asarray(lat_out, np.float64).reshape(-1)
    lo_out = np.asarray(lon_out, np.float64).reshape(-1)
    if la_in.size != lo_in.size or la_out.size != lo_out.size:
        raise ValueError("nearest_native_index: lat and lon sizes differ")
    if la_in.size == 0:
        raise ValueError("nearest_native_index: the native grid has no points")
    key = _key(la_in, lo_in, la_out, lo_out)
    if key in _MEMO:
        return _MEMO[key]
    path = cache_dir() / f"{key}.npz"
    idx = _load_cached(path, la_out.size, la_in.size) if path.exists() else None
    if idx is None:
        from scipy.spatial import cKDTree

        tree = cKDTree(_xyz(la_in, lo_in))
        _, idx = tree.query(_xyz(la_out, lo_out), workers=-1)
        idx = np.asarray(idx, dtype=np.int64)
        _save_cached(path, idx, la_in.size)
        LOG.info("native_input: nearest native index for %d output points from %d native points", la_out.size,
                 la_in.size)
    if len(_MEMO) >= _MEMO_MAX:
        _MEMO.pop(next(iter(_MEMO)))
    _MEMO[key] = idx
    return idx


# ----------------------------------------------------------------------------- reading the input

def _dims(var) -> tuple[str, ...]:
    dims = getattr(var, "dims", None)
    return tuple(dims if dims is not None else var.dimensions)


def _values(var, sel: tuple) -> np.ndarray:
    a = var[sel]
    a = getattr(a, "values", a)  # xarray DataArray -> ndarray; netCDF4 gives an (masked) ndarray
    return np.ma.filled(np.ma.asarray(a), np.nan) if np.ma.isMaskedArray(a) else np.asarray(a)


def _coords(ds, name: str) -> np.ndarray:
    v = ds[name]
    return np.asarray(_values(v, tuple(slice(None) for _ in _dims(v))), np.float64).reshape(-1)


def decode_states(ds) -> list[str]:
    """The ``weather_state`` names of a file (netCDF4 or xarray)."""
    v = ds["weather_state"]
    out = []
    for s in np.asarray(_values(v, tuple(slice(None) for _ in _dims(v)))).reshape(-1):
        if isinstance(s, bytes):
            out.append(s.decode())
        else:
            out.append(str(s))
    return out


class InputReader:
    """The input members of one prediction file on the output points (rule 41).

    With a native input (``has_native_input``) and ``native_input`` true, ``channel(ch)`` returns
    ``x[sample, member, nearest_native_index, ch]``: native values, no averaging, no interpolation. Otherwise it
    reads the first of ``fallback`` that the file carries (``x_interp`` by default) and ``source`` says so.
    ``points`` (index or boolean mask into the output points) restricts the output points read. ``source`` is
    ``SOURCE_NATIVE``, an interpolated source string, or None when the file has no input at all.
    """

    def __init__(self, ds, *, native_input: bool = True, fallback: Sequence[str] = ("x_interp",),
                 lat_out=None, lon_out=None, points=None, sample: int = 0):
        self.ds, self.sample = ds, int(sample)
        self.native = bool(native_input) and has_native_input(ds)
        self.index: np.ndarray | None = None
        if self.native:
            self.variable = "x"
            self.source = SOURCE_NATIVE
            if lat_out is None or lon_out is None:
                lat_out, lon_out = _coords(ds, "lat_hres"), _coords(ds, "lon_hres")
            idx = nearest_native_index(_coords(ds, "lat_lres"), _coords(ds, "lon_lres"), lat_out, lon_out)
            self.index = idx if points is None else idx[np.asarray(points)]
            self.points = None
        else:
            self.variable = next((v for v in fallback if v in ds.variables), None)
            reason = "native_input: false" if native_input is False and has_native_input(ds) else \
                "no native x in the file"
            self.source = None if self.variable is None else interp_source(self.variable, reason)
            self.points = None if points is None else np.asarray(points)
        self._states: list[str] | None = None

    @property
    def available(self) -> bool:
        return self.variable is not None

    @property
    def states(self) -> list[str]:
        if self._states is None:
            self._states = decode_states(self.ds)
        return self._states

    def _layout(self) -> tuple[tuple[str, ...], str, list[str]]:
        dims = _dims(self.ds[self.variable])
        point = next(d for d in dims if d.startswith("grid_point") or d in ("values", "points", "point"))
        members = [d for d in dims if d not in ("sample", "weather_state", point)]
        return dims, point, members

    @property
    def n_members(self) -> int:
        if not self.available:
            return 0
        dims, _, members = self._layout()
        shape = dict(zip(dims, self.ds[self.variable].shape))
        return int(np.prod([shape[d] for d in members])) if members else 1

    def channel(self, ch, *, member: int | None = None, dtype=np.float64) -> np.ndarray:
        """(members, output points) of one channel (name or index of weather_state); with ``member``, only that
        member as (output points,) (a field without a member axis serves every member)."""
        if not self.available:
            raise KeyError("the file carries no input field")
        ci = self.states.index(ch) if isinstance(ch, str) else int(ch)
        dims, point, members = self._layout()
        var = self.ds[self.variable]
        one = member is not None and len(members) == 1

        def pick(d):
            if d == "sample":
                return self.sample
            if d == "weather_state":
                return ci
            if one and d == members[0]:
                return int(member)
            return slice(None)

        a = _values(var, tuple(pick(d) for d in dims))
        kept = [d for d in dims if d not in ("sample", "weather_state") and not (one and d == members[0])]
        a = np.moveaxis(a, kept.index(point), -1).reshape(-1, a.shape[kept.index(point)])
        if self.index is not None:
            a = a[:, self.index]
        elif self.points is not None:
            a = a[:, self.points]
        if member is not None:
            a = a[0] if (one or a.shape[0] == 1) else a[int(member)]
        return a.astype(dtype, copy=False) if dtype is not None else a

    def describe(self) -> dict[str, Any]:
        return {"input_source": self.source, "variable": self.variable, "native": self.native,
                "n_members": self.n_members}


def read_input(ds, channel, **kwargs) -> tuple[np.ndarray, str]:
    """``(input members on the output points (M, n), source)`` for one channel; see ``InputReader``."""
    r = InputReader(ds, **kwargs)
    return r.channel(channel), r.source
