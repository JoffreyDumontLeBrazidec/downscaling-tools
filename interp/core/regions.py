"""Masked scalar functionals over the irregular hres output grid.

This module is the shared primitive for the extreme-event & high-variability
region interpretability program (see
``~/dev/docs/epics/mechanistic-interpretability/in-progress/20260528_extreme_region_interpretability.md``).

Every interp tool (feature permutation, conditioning ablation, integrated
gradients, activation patching) reduces the model output field to a scalar.
This module replaces the usual *global* reduction with a *targeted* one:
restrict the scalar to a named region, an event box, an extreme percentile, a
high-wavenumber spectral band, or the per-cell ensemble spread.

The hres OUTPUT grid is an O320 reduced-Gaussian grid (~421,120 cells), which is
IRREGULAR (not a regular lat/lon raster). Per-cell coordinates come from the
datamodule: ``datamodule.ds_valid.data.latitudes[2]`` / ``.longitudes[2]`` (each
of shape ``(n_cells,)``). Masks are boolean tensors of shape ``(n_cells,)``.

All functionals are torch ops where possible so they stay differentiable for
Integrated Gradients. The lone approximation is ``spectral_high``: because the
grid is irregular you cannot FFT raw cells, so the masked patch is regridded
onto a small regular lat/lon raster (nearest-neighbour) before the 2D FFT.
"""

from __future__ import annotations

import argparse
import math
from typing import Callable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Named region registry
# ---------------------------------------------------------------------------
# Boxes are [lat_min, lat_max, lon_min, lon_max] with lon in [-180, 180].
# Extensible: add entries here or pass an ad-hoc "bbox:..." spec.
NAMED_REGIONS: dict[str, list[float]] = {
    # "global" is handled specially (selects all cells) but kept here for discoverability.
    "global": [-90.0, 90.0, -180.0, 180.0],
    "amazon": [-15.0, 5.0, -75.0, -45.0],
    # Tighter dense central rainforest (drops Cerrado/savanna edges).
    "amazon_rainforest": [-10.0, 2.0, -72.0, -50.0],
    "tropics": [-20.0, 20.0, -180.0, 180.0],
}

# Lane-default TC event boxes. (lat_deg, lon_deg, radius_km) at storm peak intensity
# on the bundle date used. Use build_tc_box_mask() to materialise a mask.
TC_BOXES = {
    "tc_franklin": (30.0, -65.0, 500.0),   # Atl 2023-08-28 Cat 4
    "tc_humberto": (36.0, -58.0, 500.0),   # Atl 2025-09-28 peak
}


# ---------------------------------------------------------------------------
# Longitude normalization
# ---------------------------------------------------------------------------
def _to_tensor(x, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def normalize_lon_pm180(lon: torch.Tensor) -> torch.Tensor:
    """Wrap longitudes into [-180, 180).

    Handles both [0, 360] and [-180, 180] input conventions: values are wrapped
    via modulo so 350 -> -10, -190 -> 170, etc.
    """
    lon = _to_tensor(lon)
    return ((lon + 180.0) % 360.0) - 180.0


def normalize_lon_0360(lon: torch.Tensor) -> torch.Tensor:
    """Wrap longitudes into [0, 360)."""
    lon = _to_tensor(lon)
    return lon % 360.0


# ---------------------------------------------------------------------------
# Region / box / extreme masks
# ---------------------------------------------------------------------------
def _bbox_mask(
    lat: torch.Tensor,
    lon: torch.Tensor,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> torch.Tensor:
    """Boolean mask for cells inside a lat/lon bbox.

    The bbox lon bounds are given in [-180, 180]. Cell longitudes are normalized
    to the same convention. A bbox that crosses the antimeridian (lon_min >
    lon_max after normalization) is handled by OR-ing the two wrapped halves.
    """
    lat = _to_tensor(lat)
    lon_n = normalize_lon_pm180(lon)
    lat_lo, lat_hi = sorted((float(lat_min), float(lat_max)))
    lat_ok = (lat >= lat_lo) & (lat <= lat_hi)

    # A full-width longitude span (>= ~360 deg) selects every longitude. This
    # also avoids the antimeridian-collapse where [-180, 180] both wrap to -180.
    if abs(float(lon_max) - float(lon_min)) >= 359.999:
        return lat_ok

    lon_lo = normalize_lon_pm180(torch.tensor(float(lon_min))).item()
    lon_hi = normalize_lon_pm180(torch.tensor(float(lon_max))).item()
    if lon_lo <= lon_hi:
        lon_ok = (lon_n >= lon_lo) & (lon_n <= lon_hi)
    else:
        # Box wraps the antimeridian, e.g. lon in [170, -170].
        lon_ok = (lon_n >= lon_lo) | (lon_n <= lon_hi)
    return lat_ok & lon_ok


def build_region_mask(lat, lon, region_spec: str) -> torch.Tensor:
    """Build a boolean mask ``(n_cells,)`` for a region spec.

    Parameters
    ----------
    lat, lon : array-like (n_cells,)
        Per-cell latitudes / longitudes of the hres grid.
    region_spec : str
        One of:
          * ``"global"`` -> all cells.
          * a key in :data:`NAMED_REGIONS` (e.g. ``"amazon"``, ``"tropics"``).
          * ``"bbox:latmin,latmax,lonmin,lonmax"`` -> ad-hoc box.

    Returns
    -------
    torch.BoolTensor of shape ``(n_cells,)``.
    """
    lat = _to_tensor(lat)
    spec = region_spec.strip()
    if spec == "global":
        return torch.ones(lat.shape[0], dtype=torch.bool, device=lat.device)
    if spec.startswith("bbox:"):
        parts = spec[len("bbox:"):].split(",")
        if len(parts) != 4:
            raise ValueError(
                f"bbox spec must be 'bbox:latmin,latmax,lonmin,lonmax', got {region_spec!r}"
            )
        latmin, latmax, lonmin, lonmax = (float(p) for p in parts)
        return _bbox_mask(lat, lon, latmin, latmax, lonmin, lonmax)
    if spec in NAMED_REGIONS:
        latmin, latmax, lonmin, lonmax = NAMED_REGIONS[spec]
        if spec == "global":
            return torch.ones(lat.shape[0], dtype=torch.bool, device=lat.device)
        return _bbox_mask(lat, lon, latmin, latmax, lonmin, lonmax)
    raise ValueError(
        f"Unknown region spec {region_spec!r}. Known names: "
        f"{sorted(NAMED_REGIONS)}; or use 'bbox:latmin,latmax,lonmin,lonmax'."
    )


def _great_circle_km(
    lat1: torch.Tensor, lon1: torch.Tensor, lat2: float, lon2: float
) -> torch.Tensor:
    """Great-circle distance (km) from each (lat1, lon1) cell to a center point."""
    r_earth = 6371.0088  # mean Earth radius, km
    lat1 = _to_tensor(lat1)
    lon1 = normalize_lon_pm180(lon1)
    phi1 = torch.deg2rad(lat1)
    phi2 = math.radians(float(lat2))
    lon2n = normalize_lon_pm180(torch.tensor(float(lon2))).item()
    dphi = torch.deg2rad(lat1 - float(lat2))
    dlmb = torch.deg2rad(lon1 - lon2n)
    a = torch.sin(dphi / 2.0) ** 2 + math.cos(phi2) * torch.cos(phi1) * torch.sin(dlmb / 2.0) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    c = 2.0 * torch.asin(torch.sqrt(a))
    return r_earth * c


def build_box_mask(
    lat, lon, center_lat: float, center_lon: float, radius_km: float
) -> torch.Tensor:
    """Boolean mask of cells within ``radius_km`` great-circle of a center point.

    Used for storm-centered event boxes (TC cores). CLI form is
    ``"name:lat,lon,radiuskm"`` (parsed by :func:`parse_box_spec`).
    """
    dist = _great_circle_km(lat, lon, center_lat, center_lon)
    return dist <= float(radius_km)


def parse_box_spec(spec: str) -> tuple[str, float, float, float]:
    """Parse ``"name:lat,lon,radiuskm"`` -> (name, lat, lon, radius_km)."""
    if ":" not in spec:
        raise ValueError(f"box spec must be 'name:lat,lon,radiuskm', got {spec!r}")
    name, rest = spec.split(":", 1)
    parts = rest.split(",")
    if len(parts) != 3:
        raise ValueError(f"box spec must be 'name:lat,lon,radiuskm', got {spec!r}")
    lat, lon, radius = (float(p) for p in parts)
    return name.strip(), lat, lon, radius


def build_extreme_mask(
    field_1d, percentile: float, region_mask: torch.Tensor | None = None,
    side: str = "abs",
) -> torch.Tensor:
    """Mask the extreme-tail cells of ``field``.

    Parameters
    ----------
    field_1d : array-like (n_cells,)
        The field whose extremes to select (e.g. one surface target).
    percentile : float
        In ``[0, 100)``. ``95`` selects the top 5%.
    region_mask : optional BoolTensor (n_cells,)
        If given, the percentile is computed over (and the result restricted to)
        only the cells inside ``region_mask``.
    side : {"abs", "high", "low"}
        Which tail: ``abs`` = top of ``|field|`` (right for winds/tp);
        ``high`` = top of the signed field; ``low`` = bottom of the signed
        field (right for msl cyclones — ``abs`` on raw msl selects
        ANTICYCLONES, the highest absolute pressures).

    Returns
    -------
    BoolTensor (n_cells,).
    """
    field = _to_tensor(field_1d).reshape(-1)
    if side == "abs":
        mag = field.abs()
    elif side == "high":
        mag = field
    elif side == "low":
        mag = -field
    else:
        raise ValueError(f"side must be abs|high|low, got {side!r}")
    n = field.shape[0]
    out = torch.zeros(n, dtype=torch.bool, device=field.device)
    if region_mask is not None:
        region_mask = region_mask.to(device=field.device).bool().reshape(-1)
        if region_mask.sum() == 0:
            return out
        vals = mag[region_mask]
    else:
        vals = mag
    thresh = torch.quantile(vals.double(), float(percentile) / 100.0).to(mag.dtype)
    above = mag >= thresh
    if region_mask is not None:
        above = above & region_mask
    return above


# ---------------------------------------------------------------------------
# Area weights
# ---------------------------------------------------------------------------
def get_area_weights(lat, bundle=None) -> torch.Tensor:
    """Per-cell area weights for the hres grid.

    Prefers real grid weights from the graph metadata if discoverable on
    ``bundle`` (an ``area_weight`` / ``area`` node attribute on the hres/data
    nodes), else falls back to ``cos(lat)`` (O320 cells have a
    latitude-dependent area). Weights are NOT normalized here; the functionals
    normalize over the masked subset.
    """
    lat = _to_tensor(lat)
    n = lat.shape[0]
    if bundle is not None:
        w = _try_graph_area_weights(bundle, n)
        if w is not None:
            return _to_tensor(w).to(lat.device)
    return torch.cos(torch.deg2rad(lat)).clamp(min=0.0)


def _try_graph_area_weights(bundle, n_cells: int):
    """Best-effort lookup of real hres-node area weights from graph_data.

    Returns a numpy array of length ``n_cells`` or ``None``. Never raises.
    """
    try:
        graph = getattr(bundle, "graph_data", None)
        if graph is None:
            return None
        # graph_data behaves like a dict of node/edge stores keyed by name.
        candidate_keys = ("area_weight", "area", "area_weights")
        # Iterate node stores; pick the one matching n_cells with an area attr.
        try:
            node_items = graph.node_items()
        except Exception:
            node_items = []
        for _name, store in node_items:
            try:
                num = int(store.num_nodes)
            except Exception:
                num = None
            if num is not None and num != n_cells:
                continue
            for key in candidate_keys:
                if key in store:
                    arr = np.asarray(store[key]).reshape(-1)
                    if arr.shape[0] == n_cells and np.all(np.isfinite(arr)):
                        return arr
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Scalar functionals
# ---------------------------------------------------------------------------
def masked_area_mean(
    field_1d: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Area-weighted mean of ``field_1d`` over the masked cells (scalar tensor).

    Differentiable in ``field_1d``.
    """
    field = field_1d.reshape(-1)
    mask = mask.to(device=field.device).bool().reshape(-1)
    if weights is None:
        weights = torch.ones_like(field)
    weights = weights.to(device=field.device, dtype=field.dtype).reshape(-1)
    w = weights * mask.to(field.dtype)
    denom = w.sum()
    if float(denom) == 0.0:
        return field.sum() * 0.0  # zero, keeps graph
    return (field * w).sum() / denom


def functional_mean(
    field_1d: torch.Tensor,
    mask: torch.Tensor,
    lat=None,
    bundle=None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """"mean": area-weighted mean over masked cells."""
    if weights is None and lat is not None:
        weights = get_area_weights(lat, bundle=bundle)
    return masked_area_mean(field_1d, mask, weights)


def functional_box(
    field_1d: torch.Tensor,
    box_mask: torch.Tensor,
    lat=None,
    bundle=None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """"box": area-weighted mean over a box mask (alias of mean over a box)."""
    return functional_mean(field_1d, box_mask, lat=lat, bundle=bundle, weights=weights)


def _regrid_patch_to_raster(
    field_1d: torch.Tensor,
    mask: torch.Tensor,
    lat,
    lon,
    n_grid: int = 64,
) -> torch.Tensor:
    """Regrid the masked irregular patch onto an ``n_grid x n_grid`` regular raster.

    APPROXIMATION: the hres grid is an irregular reduced-Gaussian grid, so a
    direct 2D FFT on raw cells is meaningless. We bin the masked cells onto a
    regular lat/lon raster spanning the patch bbox via NEAREST-neighbour
    assignment (each raster pixel takes the mean of cells falling in it; empty
    pixels are filled by nearest-filled-pixel propagation). This preserves
    differentiability w.r.t. ``field_1d`` (it is a linear scatter-mean) at the
    cost of a coarse, axis-aligned resampling — adequate for relative
    high-vs-low wavenumber power comparisons, not for absolute spectra.

    Returns an ``(n_grid, n_grid)`` tensor.
    """
    field = field_1d.reshape(-1)
    mask = mask.to(device=field.device).bool().reshape(-1)
    lat_t = _to_tensor(lat).to(field.device)
    lon_t = normalize_lon_pm180(lon).to(field.device)

    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return torch.zeros((n_grid, n_grid), dtype=field.dtype, device=field.device)

    plat = lat_t[idx]
    plon = lon_t[idx]
    pval = field[idx]

    lat_lo, lat_hi = plat.min(), plat.max()
    lon_lo, lon_hi = plon.min(), plon.max()
    lat_span = torch.clamp(lat_hi - lat_lo, min=1e-6)
    lon_span = torch.clamp(lon_hi - lon_lo, min=1e-6)

    iy = torch.clamp(((plat - lat_lo) / lat_span * (n_grid - 1)).round().long(), 0, n_grid - 1)
    ix = torch.clamp(((plon - lon_lo) / lon_span * (n_grid - 1)).round().long(), 0, n_grid - 1)
    flat = iy * n_grid + ix

    acc = torch.zeros(n_grid * n_grid, dtype=field.dtype, device=field.device)
    cnt = torch.zeros(n_grid * n_grid, dtype=field.dtype, device=field.device)
    acc = acc.scatter_add(0, flat, pval)
    cnt = cnt.scatter_add(0, flat, torch.ones_like(pval))
    filled = cnt > 0
    raster = torch.where(filled, acc / torch.clamp(cnt, min=1.0), torch.zeros_like(acc))

    # Fill empty pixels with the global patch mean (keeps FFT well-defined and
    # differentiable; avoids spurious high-k edges from zeros).
    if (~filled).any():
        patch_mean = pval.mean()
        raster = torch.where(filled, raster, patch_mean.expand_as(raster))
    return raster.reshape(n_grid, n_grid)


def functional_spectral_high(
    field_1d: torch.Tensor,
    mask: torch.Tensor,
    lat,
    lon,
    cutoff_frac: float = 0.5,
    n_grid: int = 64,
    bundle=None,
) -> torch.Tensor:
    """"spectral_high": high-wavenumber power of the masked region.

    The masked irregular patch is regridded onto a small regular lat/lon raster
    (see :func:`_regrid_patch_to_raster` for the approximation), a 2D real FFT
    is taken, and the summed power above a radial wavenumber cutoff is returned.

    Parameters
    ----------
    cutoff_frac : float
        Fraction of the max radial wavenumber above which power is summed.
        Default 0.5 -> "top 50% of k".
    n_grid : int
        Raster resolution per side.

    Returns
    -------
    scalar tensor (high-k power). Differentiable in ``field_1d``.
    """
    raster = _regrid_patch_to_raster(field_1d, mask, lat, lon, n_grid=n_grid)
    raster = raster - raster.mean()  # drop DC so it does not dominate
    spec = torch.fft.rfft2(raster.to(torch.float32))
    power = (spec.real ** 2 + spec.imag ** 2)

    ny, nxr = power.shape
    # Radial wavenumber grid (FFT freq indices, with rfft on last axis).
    ky = torch.fft.fftfreq(n_grid, d=1.0).to(power.device)[:ny].abs()
    kx = torch.fft.rfftfreq(n_grid, d=1.0).to(power.device)[:nxr].abs()
    kk = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    kmax = kk.max()
    band = kk >= (float(cutoff_frac) * kmax)
    return (power * band.to(power.dtype)).sum()


def functional_spread(
    ensemble: torch.Tensor,
    mask: torch.Tensor,
    lat=None,
    bundle=None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """"spread": per-cell variance across the ensemble axis, area-meaned over mask.

    Parameters
    ----------
    ensemble : tensor (S, ..., n_cells)
        First axis is the ensemble-sample axis ``S``; the last axis is the grid.
        Any middle dims are reduced/broadcast away by flattening to (S, n_cells)
        if they are singleton, else averaged.
    mask : BoolTensor (n_cells,)

    Returns
    -------
    scalar tensor (area-mean of per-cell ensemble variance).
    """
    ens = ensemble
    if ens.dim() == 1:
        raise ValueError("spread needs an ensemble axis; got a 1D tensor")
    # Collapse everything between S and n_cells into the grid mean per member.
    S = ens.shape[0]
    n_cells = ens.shape[-1]
    ens2 = ens.reshape(S, -1, n_cells)
    ens2 = ens2.mean(dim=1)  # (S, n_cells)
    var = ens2.var(dim=0, unbiased=False)  # (n_cells,)
    if weights is None and lat is not None:
        weights = get_area_weights(lat, bundle=bundle)
    return masked_area_mean(var, mask, weights)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def add_region_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the shared region/functional flags to an argparse parser.

    Flags
    -----
    --region            repeatable region spec ("global", "amazon", "tropics",
                        or "bbox:latmin,latmax,lonmin,lonmax").
    --box               repeatable event box "name:lat,lon,radiuskm".
    --extreme-percentile  restrict to cells above this percentile of |field|.
    --functional        one of {mean, box, spectral_high, spread}.
    --spectral-cutoff   high-k cutoff fraction for spectral_high (default 0.5).
    --spectral-ngrid    raster size per side for spectral_high (default 64).
    --n-samples         ensemble size S for spread.
    """
    g = parser.add_argument_group("region / functional")
    g.add_argument(
        "--region",
        action="append",
        default=None,
        help="Region spec; repeatable. 'global', a named region "
        f"({sorted(NAMED_REGIONS)}), or 'bbox:latmin,latmax,lonmin,lonmax'.",
    )
    g.add_argument(
        "--box",
        action="append",
        default=None,
        help="Event box 'name:lat,lon,radiuskm'; repeatable.",
    )
    g.add_argument(
        "--extreme-percentile",
        type=float,
        default=None,
        help="Restrict to cells above this percentile of |field| (0-100).",
    )
    g.add_argument(
        "--functional",
        choices=["mean", "box", "spectral_high", "spread"],
        default="mean",
        help="Scalar reduction over the masked field.",
    )
    g.add_argument(
        "--spectral-cutoff",
        type=float,
        default=0.5,
        help="High-k cutoff fraction for spectral_high (default 0.5).",
    )
    g.add_argument(
        "--spectral-ngrid",
        type=int,
        default=64,
        help="Raster resolution per side for spectral_high (default 64).",
    )
    g.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Ensemble size S for the 'spread' functional (default 8).",
    )
    return parser


def get_hres_lat_lon(bundle, split: str = "valid"):
    """Return (lat, lon) numpy arrays for the hres output grid from the datamodule.

    Mirrors predict.py: ``data.latitudes[2]`` / ``data.longitudes[2]``.
    """
    dm = bundle.datamodule
    ds = dm.ds_valid if split == "valid" else dm.ds_train
    data = ds.data
    lat = np.asarray(data.latitudes[2])
    lon = np.asarray(data.longitudes[2])
    return lat, lon


def resolve_functional(args: argparse.Namespace, bundle) -> Callable:
    """Build a callable scalar functional from parsed CLI args + a model bundle.

    The returned callable signature depends on ``args.functional``:

      * mean / box / spectral_high:
            ``fn(field_1d) -> scalar tensor``
        where ``field_1d`` is a single surface target's field, shape (n_cells,).
      * spread:
            ``fn(ensemble) -> scalar tensor``
        where ``ensemble`` is (S, ..., n_cells).

    The region/box/extreme masks are resolved once here against the real hres
    grid coords so the returned callable is cheap to apply repeatedly (e.g.
    inside an IG loop). When multiple --region / --box are given they are OR-ed.

    Returns a callable with attributes ``.mask`` (BoolTensor), ``.lat``,
    ``.lon`` for introspection.
    """
    lat_np, lon_np = get_hres_lat_lon(bundle)
    lat = torch.as_tensor(lat_np, dtype=torch.float64)
    lon = torch.as_tensor(lon_np, dtype=torch.float64)
    n_cells = lat.shape[0]

    # Base region mask: OR of all --region specs; default "global".
    region_specs = args.region if args.region else ["global"]
    region_mask = torch.zeros(n_cells, dtype=torch.bool)
    for spec in region_specs:
        region_mask |= build_region_mask(lat, lon, spec)

    # OR in any event boxes.
    if getattr(args, "box", None):
        for box_spec in args.box:
            _name, clat, clon, radius = parse_box_spec(box_spec)
            region_mask |= build_box_mask(lat, lon, clat, clon, radius)

    base_mask = region_mask
    extreme_pct = getattr(args, "extreme_percentile", None)

    func = args.functional
    cutoff = getattr(args, "spectral_cutoff", 0.5)
    ngrid = getattr(args, "spectral_ngrid", 64)
    weights = get_area_weights(lat, bundle=bundle)

    def _resolve_mask(field_1d: torch.Tensor) -> torch.Tensor:
        m = base_mask.to(field_1d.device)
        if extreme_pct is not None:
            m = build_extreme_mask(field_1d, extreme_pct, region_mask=m)
        return m

    if func == "spread":
        def fn(ensemble: torch.Tensor) -> torch.Tensor:
            # Use the per-cell mean across S to resolve an extreme mask if asked.
            ref = ensemble.reshape(ensemble.shape[0], -1, ensemble.shape[-1]).mean(dim=(0, 1))
            m = _resolve_mask(ref) if extreme_pct is not None else base_mask.to(ensemble.device)
            return functional_spread(ensemble, m, lat=lat, bundle=bundle, weights=weights.to(ensemble.device))

        fn.mask = base_mask
    elif func == "spectral_high":
        def fn(field_1d: torch.Tensor) -> torch.Tensor:
            m = _resolve_mask(field_1d)
            return functional_spectral_high(
                field_1d, m, lat=lat.to(field_1d.device), lon=lon.to(field_1d.device),
                cutoff_frac=cutoff, n_grid=ngrid, bundle=bundle,
            )

        fn.mask = base_mask
    else:  # mean or box
        def fn(field_1d: torch.Tensor) -> torch.Tensor:
            m = _resolve_mask(field_1d)
            return functional_mean(
                field_1d, m, lat=lat.to(field_1d.device), bundle=bundle,
                weights=weights.to(field_1d.device),
            )

        fn.mask = base_mask

    fn.lat = lat
    fn.lon = lon
    fn.region_mask = base_mask
    fn.functional = func
    return fn


def build_tc_box_mask(lat, lon, name: str):
    """Convenience: great-circle box mask for a named TC event in TC_BOXES."""
    if name not in TC_BOXES:
        raise ValueError(f"Unknown TC event {name!r}; options: {sorted(TC_BOXES)}")
    clat, clon, r_km = TC_BOXES[name]
    return build_box_mask(lat, lon, clat, clon, r_km)
