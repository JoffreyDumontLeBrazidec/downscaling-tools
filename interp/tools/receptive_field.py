"""Effective receptive field of the denoiser — impulse response vs noise level.

The architecture is encoder / processor / decoder on a single-level icosahedral
hidden mesh with one-hop nearest-neighbour edges, no pooling and no global
attention, so its THEORETICAL reach is bounded by (processor depth + 2) hidden
mesh spacings.  This tool measures the EFFECTIVE reach: it perturbs the
denoiser's input inside a small disc, runs ONE denoiser call at a fixed noise
level sigma with a fixed noise draw, differences the result against an
unperturbed reference computed with the SAME noise draw, and reads the radial
profile of the response around the perturbation centre.

Two perturbation types:

  conditioning — a constant delta added to the PREPROCESSED interpolated
                 low-resolution input ``x_interp`` (normalised units) on every
                 high-resolution node inside the disc, on the four surface
                 variables 10u / 10v / 2t / msl jointly.
  state        — the same delta added to the noised residual ``y_noised``.
                 Because the noise draw is fixed, adding delta to the residual
                 before the noise is added is arithmetically identical to
                 adding it to ``y_noised`` itself, which is how it is done.

The static high-resolution forcings ``x_hres`` are never perturbed.

Everything is reported in the model's NORMALISED units: the perturbation is
defined there and the denoiser output D is the normalised residual estimate, so
the gain (response amplitude divided by perturbation amplitude) is
dimensionless and directly comparable across variables and lanes.

The radial analysis runs rank-locally on each rank's own grid shard and the
per-bin sums are all-reduced, so no full-grid field is materialised for the
statistics.  The response field is gathered only for the requested field dumps.

Usage
-----
    cd ~/dev/downscaling-tools-interp-rf
    srun --ntasks=4 --ntasks-per-node=4 python -m interp receptive_field \
        --checkpoint <ckpt> --output-dir ~/perm/interp/<id>/receptive_field \
        --event franklin_o320_o1280 --theoretical-reach-km 510 \
        --r0-km 50 --max-radius-km 1500
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.cli import add_event_args, add_model_args, setup_logging
from interp.core.data import collect_event_bundles, resolve_event_args
from interp.core.geometry import (DEFAULT_AUTO_WINDOW, detect_min_center,
                                  haversine_km, norm_lon)
from interp.core.model import (denoise_at_sigma, get_surface_target_indices,
                               get_variable_names, is_dict_api, load_model)
from interp.core.regions import get_area_weights
from interp.core.runmeta import write_run_meta

LOGGER = logging.getLogger(__name__)

# The four surface variables that carry the perturbation, in a fixed order.
PERTURB_VARS = ["10u", "10v", "2t", "msl"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _shard_bounds(shard_sizes, rank):
    """[lo, hi) global row range owned by `rank` under `shard_sizes`."""
    lo = int(sum(int(s) for s in shard_sizes[:rank]))
    return lo, lo + int(shard_sizes[rank])


def log_mem(tag):
    if not torch.cuda.is_available():
        return
    g = 1024 ** 3
    LOGGER.info("MEM %-28s alloc=%6.2f GiB  reserved=%6.2f GiB  peak=%6.2f GiB", tag,
                torch.cuda.memory_allocated() / g, torch.cuda.memory_reserved() / g,
                torch.cuda.max_memory_allocated() / g)


def _store_num_nodes(store):
    """Node count of one graph node store, whatever container the runtime uses."""
    n = getattr(store, "num_nodes", None)
    if isinstance(n, int):
        return n
    for key in ("x", "latlons", "coords"):
        v = None
        try:
            v = store[key]
        except Exception:
            v = getattr(store, key, None)
        if v is not None and hasattr(v, "shape") and len(v.shape) >= 1:
            return int(v.shape[0])
    return None


def _walk_graph(graph, prefix, out, depth=0):
    """Collect node-set sizes, descending into per-dataset sub-graphs.

    ``graph_data`` is a torch_geometric HeteroData on some runtimes and, on the
    unified runtime, a plain dict mapping each dataset name to its own
    HeteroData.  A HeteroData's own ``num_nodes`` is the sum over all its node
    types, so the individual node sets — which is where the hidden mesh size
    lives — are only visible one level down.  Edge stores, whose key is a tuple
    of three names rather than a single string, are skipped.
    """
    if depth > 2:
        return
    items = None
    fn = getattr(graph, "node_items", None)
    if callable(fn):
        try:
            items = list(fn())
        except Exception:
            items = None
    if items is None and isinstance(graph, dict):
        items = list(graph.items())
    if items is None:
        LOGGER.warning("could not enumerate graph node stores (type %s)", type(graph))
        return
    for name, store in items:
        if not isinstance(name, str):
            continue
        key = f"{prefix}{name}"
        sub = getattr(store, "node_items", None)
        if callable(sub) or isinstance(store, dict):
            before = len(out)
            _walk_graph(store, f"{key}/", out, depth + 1)
            if len(out) > before:
                continue
        try:
            n = _store_num_nodes(store)
        except Exception as exc:  # pragma: no cover - diagnostic only
            LOGGER.warning("node count for %r failed: %s", key, exc)
            n = None
        if n is not None:
            out[key] = int(n)


def graph_node_counts(bundle) -> dict:
    """Node counts per node set of the loaded graph — the pinned-graph check."""
    out = {}
    graph = getattr(bundle, "graph_data", None)
    if graph is not None:
        _walk_graph(graph, "", out)
    return out


# ---------------------------------------------------------------------------
# perturbation centres
# ---------------------------------------------------------------------------

def _cell_centres(cells, nlon, raster_deg):
    a, b = cells // nlon, cells % nlon
    return (-90.0 + (a + 0.5) * raster_deg, -180.0 + (b + 0.5) * raster_deg)


def _remote_ocean_point(lat, lon, lsm, storm_latlon, min_land_km=1500.0,
                        min_storm_km=1500.0, raster_deg=2.0):
    """Pick an open-ocean node far from any land and from the storm.

    A coarse lat/lon raster is built first: a raster cell counts as land if ANY
    grid node inside it is land, which over-counts land and so makes the
    distance-to-land estimate conservative.  The best ocean raster cell is then
    snapped to the nearest real ocean grid node, and that one node is checked
    exactly against the full set of land nodes before it is accepted.
    """
    lonn = norm_lon(lon)
    land = lsm > 0.5
    nlat = int(round(180.0 / raster_deg))
    nlon = int(round(360.0 / raster_deg))
    ilat = np.clip(((lat + 90.0) / raster_deg).astype(int), 0, nlat - 1)
    ilon = np.clip(((lonn + 180.0) / raster_deg).astype(int), 0, nlon - 1)
    cell = ilat * nlon + ilon
    land_cells = np.unique(cell[land])
    ocean_cells = np.setdiff1d(np.unique(cell), land_cells)
    lc_lat, lc_lon = _cell_centres(land_cells, nlon, raster_deg)
    oc_lat, oc_lon = _cell_centres(ocean_cells, nlon, raster_deg)

    best_d = np.empty(oc_lat.shape, dtype=float)
    step = 512
    for i in range(0, oc_lat.size, step):
        j = min(i + step, oc_lat.size)
        d = haversine_km(oc_lat[i:j, None], oc_lon[i:j, None],
                         lc_lat[None, :], lc_lon[None, :])
        best_d[i:j] = d.min(axis=1)

    d_storm = haversine_km(storm_latlon[0], norm_lon(storm_latlon[1]), oc_lat, oc_lon)
    # a cell centre can stand for a node up to ~raster_deg away in great circle
    margin = 1.2 * raster_deg * 111.2
    ok = (best_d > (min_land_km + margin)) & (d_storm > (min_storm_km + margin))
    if not ok.any():
        LOGGER.warning("no 2-deg ocean cell satisfies >%.0f km land / >%.0f km storm; "
                       "falling back to the most remote cell available",
                       min_land_km, min_storm_km)
        ok = best_d >= best_d.max()
    cand = np.where(ok)[0]
    pick = int(cand[int(np.argmax(best_d[cand]))])
    clat, clon = float(oc_lat[pick]), float(oc_lon[pick])

    d_nodes = np.where(land, np.inf, haversine_km(clat, clon, lat, lonn))
    j = int(np.argmin(d_nodes))
    node_lat, node_lon = float(lat[j]), float(lonn[j])

    land_idx = np.where(land)[0]
    exact = np.inf
    for i in range(0, land_idx.size, 2_000_000):
        chunk = land_idx[i:i + 2_000_000]
        exact = min(exact, float(haversine_km(node_lat, node_lon,
                                              lat[chunk], lonn[chunk]).min()))
    LOGGER.info("ocean centre: node (%.3f, %.3f); exact distance to nearest land "
                "%.0f km", node_lat, node_lon, exact)
    return node_lat, node_lon, exact


def _snap_to_node(lat, lon, clat, clon):
    lonn = norm_lon(lon)
    d = haversine_km(clat, norm_lon(clon), lat, lonn)
    j = int(np.argmin(d))
    return float(lat[j]), float(lonn[j]), j


def resolve_centres(args, lat, lon, lsm, msl_truth):
    """Build the three perturbation centres: storm, flat land, open ocean."""
    window = (tuple(float(x) for x in args.auto_window.split(","))
              if args.auto_window else DEFAULT_AUTO_WINDOW)
    slat, slon = detect_min_center(msl_truth, lat, lon, window)
    slat, slon, sj = _snap_to_node(lat, lon, slat, slon)
    centres = {"storm": {
        "lat": slat, "lon": slon,
        "how": f"argmin(truth msl) inside window {window}",
        "truth_msl_hPa": float(msl_truth[sj]) / 100.0,
        "lsm": float(lsm[sj])}}

    llat, llon, lj = _snap_to_node(lat, lon, args.land_lat, args.land_lon)
    centres["land"] = {
        "lat": llat, "lon": llon,
        "how": f"nearest grid node to the requested flat-land point "
               f"({args.land_lat}, {args.land_lon})",
        "lsm": float(lsm[lj]),
        "distance_to_storm_km": float(haversine_km(llat, llon, slat, slon))}

    olat, olon, od = _remote_ocean_point(lat, lon, lsm, (slat, slon),
                                         min_land_km=args.ocean_min_land_km,
                                         min_storm_km=args.ocean_min_storm_km)
    _, _, oj = _snap_to_node(lat, lon, olat, olon)
    centres["ocean"] = {
        "lat": olat, "lon": olon,
        "how": "most remote open-ocean node (2 deg land raster search, then exact "
               "verification of the chosen node against every land node)",
        "distance_to_land_km": od,
        "distance_to_storm_km": float(haversine_km(olat, olon, slat, slon)),
        "lsm": float(lsm[oj])}
    return centres


# ---------------------------------------------------------------------------
# radial reduction (rank-local sums; all-reduced by the caller)
# ---------------------------------------------------------------------------

class RadialAccumulator:
    """Per-bin area-weighted sums of the squared response on one rank's rows."""

    def __init__(self, dist_km, weights, bin_km, max_radius_km, r0_km,
                 reach_km, device):
        self.n_bins = int(round(max_radius_km / bin_km))
        self.bin_km = float(bin_km)
        self.device = device
        idx = torch.clamp((dist_km / bin_km).long(), min=0)
        self.in_range = idx < self.n_bins
        self.bin_idx = torch.where(self.in_range, idx,
                                   torch.full_like(idx, self.n_bins - 1))
        self.w = weights
        sel = self.in_range
        self.w_bin = torch.zeros(self.n_bins, device=device, dtype=torch.float64)
        self.w_bin.index_add_(0, self.bin_idx[sel], self.w[sel].double())
        # scalar shells, computed over the WHOLE grid (no max-radius cap)
        self.m_core = dist_km <= r0_km
        self.m_mid = (dist_km > 1.5 * reach_km) & (dist_km <= 3.0 * reach_km)
        self.m_far = dist_km > 3.0 * reach_km

    def reduce(self, sq):
        """sq: (N,) squared response on this rank's rows -> (profile, scalars)."""
        sq = sq.double()
        wsq = self.w.double() * sq
        prof = torch.zeros(self.n_bins, device=self.device, dtype=torch.float64)
        sel = self.in_range
        prof.index_add_(0, self.bin_idx[sel], wsq[sel])
        z = torch.zeros((), device=self.device, dtype=torch.float64)

        def _max(t):
            return t.max() if t.numel() else z

        scal = torch.stack([
            wsq.sum(),                              # 0 total weighted sq, whole grid
            wsq[self.m_core].sum(),                 # 1 inside r0
            self.w[self.m_core].double().sum(),     # 2 weight inside r0
            wsq[self.m_mid].sum(),                  # 3 shell 1.5R..3R
            self.w[self.m_mid].double().sum(),      # 4
            wsq[self.m_far].sum(),                  # 5 beyond 3R
            self.w[self.m_far].double().sum(),      # 6
            _max(sq),                               # 7 max sq anywhere
            _max(sq[self.m_far]),                   # 8 max sq beyond 3R
        ])
        return prof, scal


def radii_from_profile(prof_w_sq, bin_km, fractions=(0.5, 0.9)):
    """Radii enclosing the given fractions of the binned squared response."""
    total = float(np.sum(prof_w_sq))
    out = {}
    if not np.isfinite(total) or total <= 0:
        return {f: float("nan") for f in fractions}, total
    cum = np.cumsum(prof_w_sq) / total
    edges = np.arange(len(prof_w_sq) + 1) * bin_km
    for f in fractions:
        k = int(np.searchsorted(cum, f))
        if k >= len(cum):
            out[f] = float("nan")
            continue
        c0 = cum[k - 1] if k > 0 else 0.0
        c1 = cum[k]
        frac = 0.0 if c1 <= c0 else (f - c0) / (c1 - c0)
        out[f] = float(edges[k] + frac * bin_km)
    return out, total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="interp receptive_field",
        description="Effective receptive field of the denoiser (impulse response vs sigma).")
    add_model_args(p)
    add_event_args(p)
    g = p.add_argument_group("probe")
    g.add_argument("--sigmas", nargs="+", type=float, default=[80.0, 20.0, 5.0, 1.0, 0.2])
    g.add_argument("--r0-km", type=float, default=50.0,
                   help="radius of the perturbation disc (km)")
    g.add_argument("--delta", type=float, default=0.5,
                   help="perturbation amplitude in NORMALISED units")
    g.add_argument("--linearity-delta", type=float, default=0.25)
    g.add_argument("--linearity-sigma", type=float, default=5.0)
    g.add_argument("--linearity-centre", default="storm")
    g.add_argument("--state-sigmas", nargs="+", type=float, default=[80.0, 1.0])
    g.add_argument("--state-centre", default="storm")
    g.add_argument("--max-radius-km", type=float, default=1500.0)
    g.add_argument("--bin-km", type=float, default=10.0)
    g.add_argument("--theoretical-reach-km", type=float, required=True,
                   help="processor reach including the two mapper hops (km)")
    g.add_argument("--seed", type=int, default=20260911,
                   help="seed of the ONE noise draw reused for every call in this lane")
    g.add_argument("--centres", nargs="+", default=["ocean", "land", "storm"])
    g.add_argument("--land-lat", type=float, default=25.0,
                   help="flat-land centre latitude (default: central Sahara)")
    g.add_argument("--land-lon", type=float, default=10.0)
    g.add_argument("--ocean-min-land-km", type=float, default=1500.0)
    g.add_argument("--ocean-min-storm-km", type=float, default=1500.0)
    g.add_argument("--auto-window", default=None,
                   help="lat0,lat1,lon0,lon1 (lon 0..360) for the storm-centre search")
    g.add_argument("--dump-sigmas", nargs="+", type=float, default=[80.0, 1.0])
    g.add_argument("--no-dump", action="store_true")
    g.add_argument("--smoke", action="store_true",
                   help="one sigma, one centre, no state / linearity / dump arms")
    return p


def main(argv=None):
    setup_logging()
    return run(build_parser().parse_args(argv))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(args):
    out_path = Path(args.output_dir)

    from manual_inference.prediction.predict import (_get_parallel_info,
                                                     _init_model_comm_group)
    global_rank, local_rank, world_size = _get_parallel_info()
    sharded = world_size > 1
    device = args.device
    mcg = None
    if sharded:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(int(local_rank))
            device = f"cuda:{int(local_rank)}"
        mcg = _init_model_comm_group(device, global_rank, world_size)
        LOGGER.info("rank %d/%d (local %d) on %s — GRID-SHARDED inference",
                    global_rank, world_size, local_rank, device)
    import torch.distributed as dist

    LOGGER.info("Loading model from %s", args.checkpoint)
    bundle = load_model(args.checkpoint, device=device, precision=args.precision,
                        num_gpus_per_model=world_size)
    log_mem("after load_model")
    inner = bundle.inner_model
    dict_api = is_dict_api(inner)
    node_counts = graph_node_counts(bundle)
    LOGGER.info("GRAPH NODE COUNTS %s", node_counts)
    LOGGER.info("model class %s (dict API: %s)", type(inner).__name__, dict_api)

    target_indices = get_surface_target_indices(bundle)
    vnames = get_variable_names(bundle)
    name2in = {v: k for k, v in vnames["input_lres"].items()}
    LOGGER.info("surface targets: %s", target_indices)
    missing = [v for v in PERTURB_VARS if v not in name2in or v not in target_indices]
    if missing:
        raise SystemExit(f"receptive_field needs {PERTURB_VARS} in both the lres input "
                         f"and the output schema; missing {missing}")
    cond_channels = [int(name2in[v]) for v in PERTURB_VARS]
    state_channels = [int(target_indices[v]) for v in PERTURB_VARS]

    bundle_dir, dates, members, steps, _label = resolve_event_args(args)
    eb = collect_event_bundles(bundle, bundle_dir, dates, members, steps)
    if eb.x_lres.shape[0] != 1:
        raise SystemExit("receptive_field runs on ONE bundle (batch 1)")
    _, _, lat_full, lon_full = eb.coords
    lat_full = np.asarray(lat_full, dtype=np.float64)
    lon_full = norm_lon(np.asarray(lon_full, dtype=np.float64))
    n_hres = int(lat_full.shape[0])
    LOGGER.info("hres grid: %d nodes", n_hres)

    hres_n2i = {v: k for k, v in vnames["input_hres"].items()}
    if "lsm" not in hres_n2i:
        raise SystemExit("the hres forcings carry no lsm channel; cannot place an "
                         "open-ocean centre")
    lsm = eb.x_hres[0, 0, 0, :, hres_n2i["lsm"]].cpu().numpy().astype(np.float64)
    msl_truth = eb.y[0, 0, 0, :, target_indices["msl"]].cpu().numpy().astype(np.float64)

    centres_all = resolve_centres(args, lat_full, lon_full, lsm, msl_truth)
    wanted = ["storm"] if args.smoke else list(args.centres)
    centres = {k: centres_all[k] for k in wanted if k in centres_all}
    LOGGER.info("centres:\n%s", json.dumps(centres, indent=2, default=str))

    # ---- sharded conditioning tensors -------------------------------------
    y0 = eb.y[0:1].to(device)
    if sharded and dict_api:
        from anemoi.models.distributed.graph import gather_tensor
        from anemoi.models.distributed.shapes import get_shard_sizes
        from interp.tools.trajectory import (row_sharded_upsample,
                                             select_residual_channels)
        out_sizes = get_shard_sizes(y0, -2, mcg)
        gss = {"in_lres": out_sizes, "in_hres": out_sizes, "out_hres": out_sizes}
        shard_sizes = [int(s) for s in out_sizes]
        lo, hi = _shard_bounds(shard_sizes, global_rank)
        with torch.no_grad():
            x_lres_dev = eb.x_lres[0:1].to(device)
            x_interp_raw_sh = row_sharded_upsample(inner, x_lres_dev, lo, hi)
            del x_lres_dev
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            x_interp = bundle.pre_processors["in_lres"](x_interp_raw_sh, in_place=False)
            x_hres_p = bundle.pre_processors["in_hres"](
                eb.x_hres[0:1][:, :, :, lo:hi, :].to(device), in_place=False)
            prt = getattr(bundle.model, "pre_processors_tendencies", None)
            y_residual = inner.compute_residuals(
                y0[:, :, :, lo:hi, :], select_residual_channels(inner, x_interp_raw_sh),
                bundle.pre_processors["out_hres"], prt["out_hres"],
                target_dataset="out_hres")
            del x_interp_raw_sh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _gather(field_sh):
            return gather_tensor(field_sh, -2, out_sizes, mcg)

    elif sharded:
        from anemoi.models.distributed.graph import gather_tensor, shard_tensor
        from anemoi.models.distributed.shapes import apply_shard_shapes, get_shard_shapes
        with torch.no_grad():
            (x_interp, x_hres_p, x_interp_raw_sh), gss = inner._before_sampling(
                eb.x_lres[0:1].to(device), eb.x_hres[0:1].to(device),
                bundle.pre_processors, 1, model_comm_group=mcg)
            shard_sizes = [int(s) for s in gss]
            lo, hi = _shard_bounds(shard_sizes, global_rank)
            y_sh = shard_tensor(y0, -2, get_shard_shapes(y0, -2, mcg), mcg)
            y_residual = inner.compute_residuals(
                y_sh[:, 0, ...], x_interp_raw_sh[:, 0, ...])[:, None, ...]
            del y_sh, x_interp_raw_sh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _gss = gss

        def _gather(field_sh):
            return gather_tensor(field_sh, -2,
                                 apply_shard_shapes(field_sh, -2, _gss), mcg)

    elif dict_api:
        # Unified (dict-API) single-GPU preparation, mirroring the single-GPU
        # dict-API branch of interp.tools.trajectory. _before_sampling returns the
        # NORMALISED interpolated conditioning, while compute_residuals wants the
        # RAW interpolated input together with the out_hres state and tendency
        # normalisers. prepare_batch below implements the two-argument ds-tensor
        # signature instead, which this model does not have.
        from interp.tools.trajectory import select_residual_channels
        with torch.no_grad():
            batch = {"in_lres": eb.x_lres[0:1].to(device),
                     "in_hres": eb.x_hres[0:1].to(device)}
            (x_interp, x_hres_p), _ = inner._before_sampling(
                batch, bundle.pre_processors, 1)
            x_interp_raw = inner.apply_interpolate_to_high_res(
                eb.x_lres[0:1].to(device)[:, 0, ...])[:, None, ...]
            prt = getattr(bundle.model, "pre_processors_tendencies", None)
            y_residual = inner.compute_residuals(
                y0, select_residual_channels(inner, x_interp_raw),
                bundle.pre_processors["out_hres"], prt["out_hres"],
                target_dataset="out_hres")
            del x_interp_raw
        gss = None
        shard_sizes = [n_hres]
        lo, hi = 0, n_hres

        def _gather(field_sh):
            return field_sh

    else:
        from interp.core.model import prepare_batch
        prepared = prepare_batch(bundle, eb.x_lres[0:1], eb.x_hres[0:1], eb.y[0:1])
        x_interp, x_hres_p = prepared["x_interp"], prepared["x_hres"]
        y_residual = prepared["y_residual"]
        gss = None
        shard_sizes = [n_hres]
        lo, hi = 0, n_hres

        def _gather(field_sh):
            return field_sh

    del y0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.info("rank %d owns hres rows [%d, %d) of %d", global_rank, lo, hi, n_hres)
    log_mem("after conditioning setup")
    gss_arg = gss if sharded else None

    if tuple(y_residual.shape[:3]) != (1, 1, 1):
        raise SystemExit(f"expected a (1,1,1,grid,vars) residual, got "
                         f"{tuple(y_residual.shape)}")
    n_out = int(y_residual.shape[-1])
    if max(target_indices.values()) >= n_out:
        raise SystemExit(f"surface target indices {target_indices} exceed the "
                         f"{n_out} residual channels")
    LOGGER.info("residual / output channels: %d", n_out)

    # ---- the ONE noise draw, reused by every call in this lane ------------
    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
    noise_full = torch.randn(n_hres, n_out, generator=gen, dtype=torch.float32)
    noise = (noise_full[lo:hi].reshape(1, 1, 1, hi - lo, n_out)
             .to(device).to(y_residual.dtype).contiguous())
    del noise_full

    # ---- rank-local geometry ---------------------------------------------
    w_full = get_area_weights(lat_full, bundle=bundle).float().cpu().numpy()
    weights = torch.from_numpy(np.ascontiguousarray(w_full[lo:hi])).to(device).float()
    LOGGER.info("area weights: %d values, range [%.4g, %.4g], sum %.6g",
                int(w_full.size), float(w_full.min()), float(w_full.max()),
                float(w_full.sum()))

    accs, disc_masks = {}, {}
    for cname, c in centres.items():
        d = torch.from_numpy(np.ascontiguousarray(
            haversine_km(c["lat"], c["lon"], lat_full[lo:hi], lon_full[lo:hi])
        )).to(device).float()
        accs[cname] = RadialAccumulator(d, weights, args.bin_km, args.max_radius_km,
                                        args.r0_km, args.theoretical_reach_km, device)
        disc_masks[cname] = torch.nonzero(d <= args.r0_km, as_tuple=False).flatten()
        n_disc = torch.tensor([float(disc_masks[cname].numel())], device=device)
        if sharded:
            dist.all_reduce(n_disc)
        c["n_disc_nodes"] = int(n_disc.item())
        LOGGER.info("centre %s: %d hres nodes inside r0 = %.0f km",
                    cname, c["n_disc_nodes"], args.r0_km)
        del d

    # ---- denoiser calls ---------------------------------------------------
    def denoise(sigma, x_i=None, y_r=None):
        return denoise_at_sigma(bundle,
                                x_interp if x_i is None else x_i,
                                x_hres_p,
                                y_residual if y_r is None else y_r,
                                float(sigma), noise, model_comm_group=mcg,
                                grid_shard_shapes=gss_arg)

    def _add_on_disc(t, cname, channels, delta):
        out = t.clone()
        idx = disc_masks[cname]
        if idx.numel():
            for ch in channels:
                out[..., idx, ch] += float(delta)
        return out

    ref_cache = {}
    ref_stats = {}

    def reference(sigma):
        """Unperturbed denoiser output at this sigma, computed once and reused.

        The reference magnitude is recorded as well, because the response is the
        difference of two fp32 forwards: anything smaller than roughly 1e-7
        times the local |D| simply cannot be resolved, and that measurement
        floor has to be stated next to any zero that is reported.
        """
        key = float(sigma)
        if key not in ref_cache:
            ref_cache[key] = denoise(key)
            r = ref_cache[key][0, 0, 0].float()
            st = torch.stack([(r ** 2).sum().double(),
                              torch.tensor(float(r.numel()), device=r.device,
                                           dtype=torch.float64),
                              r.abs().max().double()])
            if sharded:
                sums = st[0:2].contiguous()
                mx = st[2:3].contiguous()
                dist.all_reduce(sums)
                dist.all_reduce(mx, op=dist.ReduceOp.MAX)
                st = torch.cat([sums, mx])
            ref_stats[key] = {
                "rms_D_ref_all_channels": float(torch.sqrt(st[0] / st[1]).item()),
                "max_abs_D_ref": float(st[2].item()),
                "fp32_resolution_floor_on_max": float(st[2].item()) * 1.2e-7,
            }
        return ref_cache[key]

    rows, profiles, sanity = [], {}, {}

    def analyse(dD, cname, sigma, ptype, delta):
        acc = accs[cname]
        d = dD[0, 0, 0].float()                       # (Nshard, V)
        prof_l, scal_l, names = [], [], []
        for v in PERTURB_VARS:
            p, s = acc.reduce(d[:, target_indices[v]] ** 2)
            prof_l.append(p); scal_l.append(s); names.append(v)
        p, s = acc.reduce((d ** 2).mean(dim=-1))
        prof_l.append(p); scal_l.append(s); names.append("all_channels")
        prof = torch.stack(prof_l)
        scal = torch.stack(scal_l)
        wbin = acc.w_bin.clone()
        if sharded:
            dist.all_reduce(prof)
            dist.all_reduce(wbin)
            sums = scal[:, 0:7].contiguous()
            maxes = scal[:, 7:9].contiguous()
            dist.all_reduce(sums)
            dist.all_reduce(maxes, op=dist.ReduceOp.MAX)
            scal = torch.cat([sums, maxes], dim=1)
        if global_rank != 0:
            return
        prof_np = prof.cpu().numpy()
        wbin_np = wbin.cpu().numpy()
        sc = scal.cpu().numpy()
        for i, v in enumerate(names):
            radii, tot_binned = radii_from_profile(prof_np[i], args.bin_km)
            with np.errstate(divide="ignore", invalid="ignore"):
                rms_prof = np.sqrt(np.where(wbin_np > 0, prof_np[i] / wbin_np, np.nan))
            core = float(np.sqrt(sc[i, 1] / sc[i, 2])) if sc[i, 2] > 0 else float("nan")
            mid = float(np.sqrt(sc[i, 3] / sc[i, 4])) if sc[i, 4] > 0 else float("nan")
            far = float(np.sqrt(sc[i, 5] / sc[i, 6])) if sc[i, 6] > 0 else float("nan")
            key = f"{ptype}|{cname}|sigma{sigma:g}|delta{delta:g}|{v}"
            profiles[key] = {
                "bin_edges_km": (np.arange(len(rms_prof) + 1) * args.bin_km).tolist(),
                "rms": [None if not np.isfinite(x) else float(x) for x in rms_prof]}
            rows.append({
                "centre": cname, "sigma": float(sigma), "type": ptype,
                "delta": float(delta), "variable": v,
                "R50_km": radii[0.5], "R90_km": radii[0.9],
                "rms_core_r0": core,
                "gain_core_over_delta": core / delta if delta else float("nan"),
                "rms_shell_1p5R_3R": mid,
                "rms_beyond_3R": far,
                "max_abs_beyond_3R": float(np.sqrt(max(sc[i, 8], 0.0))),
                "max_abs_global": float(np.sqrt(max(sc[i, 7], 0.0))),
                "total_weighted_sq_global": float(sc[i, 0]),
                "total_weighted_sq_binned": float(tot_binned),
                "fraction_inside_max_radius": (float(tot_binned / sc[i, 0])
                                               if sc[i, 0] > 0 else float("nan")),
            })

    # -- determinism of the (possibly sharded) reference path ---------------
    sigmas = [float(args.sigmas[0])] if args.smoke else [float(s) for s in args.sigmas]
    s0 = sigmas[0]
    ref0 = reference(s0)
    ref0b = denoise(s0)
    stats = torch.tensor([float((ref0 - ref0b).abs().max()),
                          float(ref0.abs().max())], device=device)
    if sharded:
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    sanity["determinism"] = {
        "sigma": s0,
        "max_abs_diff_two_reference_calls": float(stats[0].item()),
        "max_abs_reference": float(stats[1].item()),
        "bit_exact": bool(stats[0].item() == 0.0)}
    LOGGER.info("DETERMINISM sigma=%g: max|ref1-ref2| = %.6e (max|D_ref| = %.6e)",
                s0, float(stats[0].item()), float(stats[1].item()))
    del ref0b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dumps = {}
    surf_idx = torch.tensor([target_indices[v] for v in PERTURB_VARS], device=device)

    # -- conditioning arm ---------------------------------------------------
    for sigma in sigmas:
        ref = reference(sigma)
        for cname in centres:
            D = denoise(sigma, x_i=_add_on_disc(x_interp, cname, cond_channels,
                                                args.delta))
            dD = D - ref
            analyse(dD, cname, sigma, "conditioning", args.delta)
            if (not args.smoke and not args.no_dump and cname == "storm"
                    and any(abs(sigma - s) < 1e-9 for s in args.dump_sigmas)):
                full = _gather(dD[..., surf_idx].contiguous())
                if global_rank == 0:
                    dumps[f"dD_sigma{sigma:g}"] = full[0, 0, 0].float().cpu().numpy()
                del full
            del D, dD
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        LOGGER.info("conditioning arm done at sigma %g", sigma)

    # -- linearity control --------------------------------------------------
    if not args.smoke and args.linearity_centre in centres:
        sig = float(args.linearity_sigma)
        ref = reference(sig)
        D = denoise(sig, x_i=_add_on_disc(x_interp, args.linearity_centre,
                                          cond_channels, args.linearity_delta))
        analyse(D - ref, args.linearity_centre, sig, "conditioning",
                args.linearity_delta)
        del D
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("linearity control done (delta %g at sigma %g)",
                    args.linearity_delta, sig)

    # -- state arm ----------------------------------------------------------
    if not args.smoke and args.state_centre in centres:
        for sigma in [float(s) for s in args.state_sigmas]:
            ref = reference(sigma)
            D = denoise(sigma, y_r=_add_on_disc(y_residual, args.state_centre,
                                                state_channels, args.delta))
            analyse(D - ref, args.state_centre, sigma, "state", args.delta)
            del D
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        LOGGER.info("state arm done")

    # ---- write results ----------------------------------------------------
    if global_rank == 0:
        out_path.mkdir(parents=True, exist_ok=True)
        result = {
            "checkpoint": args.checkpoint,
            "event": args.event,
            "bundles": eb.paths,
            "world_size": world_size,
            "precision": args.precision,
            "units": "normalised model units for both the perturbation and the "
                     "response, so the gain is dimensionless",
            "graph_node_counts": node_counts,
            "model_class": type(inner).__name__,
            "dict_api": dict_api,
            "n_hres_nodes": n_hres,
            "n_output_channels": n_out,
            "theoretical_reach_km": args.theoretical_reach_km,
            "r0_km": args.r0_km,
            "delta": args.delta,
            "bin_km": args.bin_km,
            "max_radius_km": args.max_radius_km,
            "noise_seed": args.seed,
            "perturbed_variables": PERTURB_VARS,
            "conditioning_channels_in_lres": cond_channels,
            "state_channels_in_output": state_channels,
            "centres": centres,
            "sanity": sanity,
            "reference_stats": ref_stats,
            "rows": rows,
            "profiles": profiles,
        }
        with open(out_path / "receptive_field.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        cols = ["centre", "sigma", "type", "delta", "variable", "R50_km", "R90_km",
                "rms_core_r0", "gain_core_over_delta", "rms_shell_1p5R_3R",
                "rms_beyond_3R", "max_abs_beyond_3R", "max_abs_global",
                "total_weighted_sq_global", "total_weighted_sq_binned",
                "fraction_inside_max_radius"]
        with open(out_path / "receptive_field.csv", "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=["lane"] + cols)
            wtr.writeheader()
            lane = args.event or "event"
            for r in rows:
                wtr.writerow({"lane": lane, **{c: r[c] for c in cols}})
        if dumps:
            np.savez_compressed(
                out_path / "dD_fields.npz",
                lat=lat_full.astype(np.float32), lon=lon_full.astype(np.float32),
                variables=np.array(PERTURB_VARS),
                centre_lat=np.float32(centres["storm"]["lat"]),
                centre_lon=np.float32(centres["storm"]["lon"]),
                **{k: v.astype(np.float32) for k, v in dumps.items()})
            LOGGER.info("field dumps written: %s", sorted(dumps))
        write_run_meta(out_path, "receptive_field", args,
                       extra={"graph_node_counts": node_counts,
                              "world_size": world_size,
                              "centres": centres, "sanity": sanity})
        LOGGER.info("receptive_field complete -> %s", out_path)

    if sharded and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
