"""Truth and interp-baseline sources for precipitation evaluation.

Why this module exists (o1280->o2560 "definitive tp" fix, 2026-08-26):

* The main-lane prediction NetCDFs carry NO tp truth (the bundles never
  embedded it), so evaluators that read truth from `y` draw blank panels.
  `PrecipTruthSource` supplies the 6h-window tp truth per (date, step)
  straight from the per-date `_tp_dea` GRIB — verified grid-identical to the
  predictions' hres grid.
* tp/cp are OUTPUT-ONLY channels on this lane (the checkpoint's in_lres has
  no tp/cp), so the exported `x_interp` tp channel is identically zero and
  must never be used as an "input" series. `LresInterpBaseline` builds the
  honest interpolation baseline instead: the driving o1280 ENFO member tp,
  mapped to the hres grid through a cached nearest-neighbour index.

Both sources return values in the native GRIB unit (metres of water per 6h
window). Conversion to mm is a presentation concern and stays in the callers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

# Relative tolerance for the "series never decreases" accumulation test.
_ACCUM_REL_TOL = 1e-6
# A genuinely accumulated series keeps growing; require the last-step mean to
# exceed the first-step mean by this factor before auto-deaccumulating.
_ACCUM_GROWTH_FACTOR = 1.5
# Grid agreement tolerance in degrees between GRIB coords and NC coords.
_GRID_TOL_DEG = 1e-3


def _read_grib_var(path: Path, var: str):
    """Read all messages of `var` from a GRIB file.

    Returns (by_key, lats, lons) where by_key maps (number, endStep) ->
    float32 values. `number` is 0 for messages without a perturbation number.
    """
    import eccodes as ec

    by_key: dict[tuple[int, int], np.ndarray] = {}
    lats = lons = None
    with open(path, "rb") as f:
        while True:
            gid = ec.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                if ec.codes_get(gid, "shortName") != var:
                    continue
                step = int(ec.codes_get(gid, "endStep"))
                try:
                    number = int(ec.codes_get(gid, "number"))
                except Exception:
                    number = 0
                if lats is None:
                    lats = ec.codes_get_array(gid, "latitudes")
                    lons = ec.codes_get_array(gid, "longitudes")
                by_key[(number, step)] = ec.codes_get_values(gid).astype(np.float32)
            finally:
                ec.codes_release(gid)
    if not by_key:
        raise FileNotFoundError(f"No '{var}' messages in {path}")
    return by_key, lats, lons


def check_grid_match(lat_ref, lon_ref, lat_grib, lon_grib, *, context: str) -> None:
    """Refuse to score on a mismatched or reordered grid.

    Coordinates must agree pointwise (same ordering) within _GRID_TOL_DEG.
    A mismatch here means every downstream number would be silently wrong,
    which is exactly the class of error this pipeline exists to end.
    """
    if len(lat_ref) != len(lat_grib):
        raise ValueError(
            f"{context}: grid size mismatch — reference {len(lat_ref)} points "
            f"vs GRIB {len(lat_grib)} points"
        )
    sl = slice(None, None, max(1, len(lat_ref) // 4001))
    dlat = float(np.max(np.abs(np.asarray(lat_ref)[sl] - np.asarray(lat_grib)[sl])))
    dlon_raw = np.abs(np.asarray(lon_ref)[sl] - np.asarray(lon_grib)[sl])
    dlon = float(np.max(np.minimum(dlon_raw, np.abs(dlon_raw - 360.0))))
    if dlat > _GRID_TOL_DEG or dlon > _GRID_TOL_DEG:
        raise ValueError(
            f"{context}: grid ordering mismatch — max|dlat|={dlat:.6f} deg, "
            f"max|dlon|={dlon:.6f} deg (tolerance {_GRID_TOL_DEG})"
        )


def maybe_deaccumulate(values_by_step: dict[int, np.ndarray], *, context: str):
    """Convert an accumulated-from-init series to per-window values if needed.

    Correct-default behaviour (house rule): a per-step precip series whose
    grid mean never decreases AND grows substantially over the forecast is an
    accumulation-since-start; the caller wants 6h windows, so difference it,
    warn loudly, and carry on. A genuine 6h-window series is non-monotonic
    and passes through untouched.

    Returns (values_by_step, was_accumulated).
    """
    steps = sorted(values_by_step)
    if len(steps) < 3:
        return values_by_step, False
    means = np.array([float(np.mean(values_by_step[s])) for s in steps])
    scale = float(np.max(np.abs(means))) or 1.0
    non_decreasing = bool(np.all(np.diff(means) >= -_ACCUM_REL_TOL * scale))
    grows = means[-1] > _ACCUM_GROWTH_FACTOR * max(means[0], _ACCUM_REL_TOL * scale)
    if not (non_decreasing and grows):
        return values_by_step, False
    LOG.warning(
        "%s: per-step series is ACCUMULATED-FROM-INIT (grid-mean grows "
        "monotonically %.3g -> %.3g); auto-deaccumulating to per-window values. "
        "Fix the source data — this correction should not be load-bearing.",
        context, means[0], means[-1],
    )
    out = {steps[0]: values_by_step[steps[0]]}
    for prev, cur in zip(steps[:-1], steps[1:]):
        out[cur] = values_by_step[cur] - values_by_step[prev]
    return out, True


class PrecipTruthSource:
    """6h-window tp truth per (date, step) from a per-date GRIB template.

    `grib_tpl` must contain "{date}". The grid is verified against the
    reference coordinates on the first load; an accumulated series is
    auto-deaccumulated with a loud warning (see maybe_deaccumulate).
    """

    def __init__(self, grib_tpl: str, *, var: str = "tp", _reader=None):
        self.grib_tpl = str(grib_tpl)
        self.var = var
        self._reader = _reader or (lambda date: _read_grib_var(
            Path(self.grib_tpl.format(date=date)), self.var))
        self._cache_date: str | None = None
        self._cache: dict[int, np.ndarray] = {}
        self._lats = self._lons = None
        self._grid_checked = False

    def preload(self, date: str) -> dict[int, np.ndarray]:
        if self._cache_date == date:
            return self._cache
        by_key, lats, lons = self._reader(date)
        by_step = {step: vals for (_num, step), vals in by_key.items()}
        by_step, was_acc = maybe_deaccumulate(
            by_step, context=f"tp truth {self.grib_tpl.format(date=date)}")
        if not was_acc:
            LOG.info("tp truth %s: per-window (deaccumulated) series confirmed, "
                     "%d steps", date, len(by_step))
        self._cache_date, self._cache = date, by_step
        self._lats, self._lons = lats, lons
        return self._cache

    def steps(self, date: str) -> list[int]:
        return sorted(self.preload(date))

    def load(self, date: str, step: int) -> np.ndarray:
        cache = self.preload(date)
        if step not in cache:
            raise KeyError(
                f"tp truth for date {date} has no step {step}; "
                f"available: {sorted(cache)}")
        return cache[step]

    def verify_grid(self, lat_ref, lon_ref) -> None:
        """Check GRIB grid against reference coords once (idempotent)."""
        if self._grid_checked:
            return
        if self._lats is None:
            raise RuntimeError("verify_grid called before any preload()")
        check_grid_match(lat_ref, lon_ref, self._lats, self._lons,
                         context="tp truth GRIB")
        self._grid_checked = True

    def release(self) -> None:
        self._cache_date, self._cache = None, {}


class LresInterpBaseline:
    """o1280 ENFO member tp interpolated to the hres grid (nearest neighbour).

    `grib_tpl` must contain "{date}" and resolve to a per-date GRIB holding
    all members and steps of the deaccumulated driver tp. The source->target
    nearest-neighbour index is built once (unit-sphere KD-tree) and cached at
    `index_cache` so later runs skip the build.
    """

    def __init__(self, grib_tpl: str, index_cache: str | Path | None,
                 *, var: str = "tp", _reader=None):
        self.grib_tpl = str(grib_tpl)
        self.index_cache = Path(index_cache) if index_cache else None
        self.var = var
        self._reader = _reader or (lambda date: _read_grib_var(
            Path(self.grib_tpl.format(date=date)), self.var))
        self._cache_date: str | None = None
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._src_lats = self._src_lons = None
        self._nn_index: np.ndarray | None = None

    # -- data access ---------------------------------------------------------

    def _preload(self, date: str) -> dict[tuple[int, int], np.ndarray]:
        if self._cache_date == date:
            return self._cache
        by_key, lats, lons = self._reader(date)
        by_key, was_acc = maybe_deaccumulate_by_member(
            by_key, context=f"baseline lres tp {self.grib_tpl.format(date=date)}")
        self._cache_date, self._cache = date, by_key
        self._src_lats, self._src_lons = lats, lons
        return self._cache

    def members(self, date: str) -> list[int]:
        return sorted({num for num, _ in self._preload(date)})

    def load(self, date: str, step: int, member: int) -> np.ndarray:
        """Return member tp on the TARGET grid (requires ensure_index first)."""
        if self._nn_index is None:
            raise RuntimeError("ensure_index(dst_lat, dst_lon) must run before load()")
        cache = self._preload(date)
        key = (member, step)
        if key not in cache:
            raise KeyError(
                f"baseline lres tp {date}: no (member={member}, step={step}); "
                f"members={self.members(date)} "
                f"steps={sorted({s for _m, s in cache})}")
        return cache[key][self._nn_index]

    def release(self) -> None:
        self._cache_date, self._cache = None, {}

    # -- interpolation index -------------------------------------------------

    def ensure_index(self, dst_lat, dst_lon, *, probe_date: str) -> np.ndarray:
        """Build or load the source->target nearest-neighbour index."""
        if self._nn_index is not None:
            return self._nn_index
        self._preload(probe_date)  # populates source coords
        n_src, n_dst = len(self._src_lats), len(dst_lat)
        if self.index_cache is not None and self.index_cache.exists():
            with np.load(self.index_cache) as z:
                if int(z["n_src"]) == n_src and int(z["n_dst"]) == n_dst:
                    self._nn_index = z["index"].astype(np.int64)
                    LOG.info("interp index loaded from %s (src=%d dst=%d)",
                             self.index_cache, n_src, n_dst)
                    return self._nn_index
                LOG.warning("interp index cache %s does not match grids "
                            "(cache src=%d dst=%d, need src=%d dst=%d); rebuilding",
                            self.index_cache, int(z["n_src"]), int(z["n_dst"]),
                            n_src, n_dst)
        self._nn_index = build_nn_index(self._src_lats, self._src_lons,
                                        dst_lat, dst_lon)
        if self.index_cache is not None:
            self.index_cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.index_cache.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, index=self._nn_index.astype(np.int32),
                                n_src=n_src, n_dst=n_dst)
            tmp.replace(self.index_cache)
            LOG.info("interp index built and cached at %s", self.index_cache)
        return self._nn_index


def maybe_deaccumulate_by_member(by_key: dict[tuple[int, int], np.ndarray],
                                 *, context: str):
    """Per-member variant of maybe_deaccumulate for (number, step) keyed data."""
    members = sorted({num for num, _ in by_key})
    out: dict[tuple[int, int], np.ndarray] = {}
    any_acc = False
    for m in members:
        per_step = {s: v for (num, s), v in by_key.items() if num == m}
        per_step, was_acc = maybe_deaccumulate(
            per_step, context=f"{context} member {m}")
        any_acc |= was_acc
        for s, v in per_step.items():
            out[(m, s)] = v
    return out, any_acc


def build_nn_index(src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """Great-circle nearest-neighbour index via a unit-sphere KD-tree."""
    from scipy.spatial import cKDTree

    def to_xyz(lat, lon):
        la = np.radians(np.asarray(lat, dtype=np.float64))
        lo = np.radians(np.asarray(lon, dtype=np.float64))
        cos_la = np.cos(la)
        return np.column_stack((cos_la * np.cos(lo), cos_la * np.sin(lo),
                                np.sin(la)))

    tree = cKDTree(to_xyz(src_lat, src_lon))
    _dist, idx = tree.query(to_xyz(dst_lat, dst_lon), k=1, workers=-1)
    return idx.astype(np.int64)


def is_degenerate_channel(values: np.ndarray) -> bool:
    """True when a channel is unusable as a data series (all zero / all NaN).

    The o2560 lane exports x_interp for OUTPUT-ONLY channels (tp/cp) as
    all-zero; plotting that as an "input" series is the silent-zero trap this
    predicate exists to catch.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(np.all(finite == 0.0))
