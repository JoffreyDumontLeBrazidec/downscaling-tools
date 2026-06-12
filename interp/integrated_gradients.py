"""Integrated Gradients (Tier-2) for AIFSDD — per surface target, storm-centered.

Reduces each surface-target OUTPUT field to a SCALAR functional, then attributes
that scalar back to the full conditioning INPUT fields (x_interp lres + x_hres)
via a hand-rolled Riemann-sum Integrated Gradients (Captum optional).

The per-target output is a field over ~421k hres cells; attributing the whole
field would need one backward per cell. Instead each surface target's output
field is collapsed to a SCALAR functional F and IG attributes F back to the
full preprocessed conditioning tensors:
  - x_interp : lres conditioning, shape (B, 1, E, N_lres_grid, V_lres)
  - x_hres   : hres forcing,      shape (B, 1, E, N_hres_grid, V_hres)

Functionals (config-selectable via --functionals):
  - global_mean : area-mean of the target field over all hres cells.
  - box         : mean of the target field over hres cells within radius R km of a
                  probe center (lat0, lon0). The storm-centered probe — "what drives
                  <target> right HERE?". Specified as "name:lat,lon,radiuskm".

Baseline is zeros (default) or a per-variable climatology mean (--baseline mean).
Attribution is computed at several sigmas since it is noise-regime dependent. The
diffusion noise is held FIXED across the baseline->input path so the path integral
is well defined.

Data loading mirrors interp.feature_permutation: batches come from the validation
dataloader (no shuffling needed for IG).

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp.integrated_gradients \
        --checkpoint /home/ecm5702/scratch/aifs/checkpoint/59e4.../anemoi-...step_300000.ckpt \
        --output-dir ~/perm/interp/59e4_300k/integrated_gradients \
        --sigmas 0.1 1.0 5.0 20.0 80.0 \
        --n-batches 2 \
        --functionals global_mean,box \
        --boxes "idalia:27.0,-83.0,500" \
        --ig-steps 32 --device cuda --precision fp32
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_DT_ROOT = Path(__file__).resolve().parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.model_utils import (
    load_model,
    get_variable_names,
    prepare_batch,
    get_surface_target_indices,
)
from manual_inference.input_data_construction.bundle import (
    load_inputs_from_bundle_numpy,
    extract_target_from_bundle,
)

LOGGER = logging.getLogger(__name__)

# NW-Atlantic search window for TC-center auto-detection (lat0,lat1,lon0,lon1; lon 0..360).
# Covers Franklin 2023 (~-70 W) and excludes Idalia's Gulf landfall (~-83 W = 277 E).
DEFAULT_AUTO_WINDOW = (10.0, 45.0, 280.0, 320.0)

# Optional Captum acceleration; falls back to the hand-rolled Riemann IG below.
try:
    import captum  # noqa: F401
    _HAVE_CAPTUM = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_CAPTUM = False


# ---------------------------------------------------------------------------
# checkpoint id helper  (parent-dir hex + step  ->  "59e4_300k")
# ---------------------------------------------------------------------------

def ckpt_id_from_path(checkpoint_path: str) -> str:
    """Build the ckpt_id used in the perm output convention.

    parent-dir "59e40596..." + filename ".../step_300000.ckpt"
        -> "59e4_300k"  (first 4 hex chars + "_" + step//1000 + "k")
    """
    p = Path(checkpoint_path)
    hex4 = p.parent.name[:4]
    step = None
    stem = p.stem
    if "step_" in stem:
        tail = stem.split("step_")[-1]
        digits = "".join(c for c in tail if c.isdigit())
        if digits:
            step = int(digits)
    if step is None:
        for tok in stem.replace("-", "_").split("_"):
            if tok.startswith("step") and tok[4:].isdigit():
                step = int(tok[4:])
    if step is None:
        return hex4
    return f"{hex4}_{step // 1000}k"


# ---------------------------------------------------------------------------
# geometry helpers (probe boxes)
# ---------------------------------------------------------------------------

def _norm_lon(lon):
    """Normalize longitudes to [-180, 180) so explicit western-hemisphere centers
    (e.g. lon=-83.0 for Idalia) compare correctly against the grid convention."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def _haversine_km(lat0, lon0, lat, lon):
    """Great-circle distance (km) from a single point to arrays lat/lon (degrees)."""
    r = 6371.0
    dlon_deg = (lon - lon0 + 180.0) % 360.0 - 180.0
    lat0r, latr = np.radians(lat0), np.radians(lat)
    dlatr, dlonr = np.radians(lat - lat0), np.radians(dlon_deg)
    a = np.sin(dlatr / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlonr / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _box_mask(lat, lon, lat0, lon0, radius_km):
    """Boolean mask over the grid for cells within radius_km of (lat0, lon0)."""
    d = _haversine_km(lat0, _norm_lon(lon0), _norm_lon(lat), _norm_lon(lon))
    return d <= float(radius_km)


def _detect_center(msl_field, lat, lon, window):
    """Locate a storm center as the argmin of msl within a lat/lon window."""
    la0, la1, lo0, lo1 = window
    lonn = np.asarray(lon) % 360.0
    sel = (lat >= la0) & (lat <= la1) & (lonn >= lo0) & (lonn <= lo1)
    if not sel.any():
        raise ValueError(f"auto-detect window {window} contains no grid cells")
    idx = int(np.argmin(np.where(sel, msl_field, np.inf)))
    return float(lat[idx]), float(lonn[idx])


def _parse_boxes(box_specs, lat_hres, lon_hres, msl_field=None, window=None):
    """Parse --boxes specs into {name: dict(lat, lon, radius_km, mask, n_cells)}.

    Spec form: "name:lat,lon,radius_km" (explicit) or "name:auto[,radius_km]"
    (storm-centered: center = argmin(msl) inside `window`, needs `msl_field`).
    """
    boxes = {}
    for spec in box_specs:
        name, _, rest = spec.partition(":")
        parts = [p.strip() for p in rest.split(",")]
        if parts and parts[0].lower() == "auto":
            if msl_field is None:
                raise ValueError(f"box {name!r} uses 'auto' but no msl field available "
                                 "(auto-detect needs bundle target data)")
            radius_km = float(parts[1]) if len(parts) > 1 else 500.0
            lat0, lon0 = _detect_center(msl_field, lat_hres, lon_hres,
                                        window or DEFAULT_AUTO_WINDOW)
            LOGGER.info("box %s: auto-detected storm center (%.2f,%.2f)", name, lat0, lon0)
        elif len(parts) == 3:
            lat0, lon0, radius_km = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            raise ValueError(f"box spec {spec!r} must be 'name:lat,lon,radiuskm' or 'name:auto'")
        mask = _box_mask(lat_hres, lon_hres, lat0, lon0, radius_km)
        n = int(mask.sum())
        if n == 0:
            raise ValueError(
                f"box {name} ({lat0},{lon0},{radius_km}km) has 0 hres cells")
        boxes[name] = {"lat": lat0, "lon": lon0 % 360.0, "radius_km": radius_km,
                       "mask": mask, "n_cells": n}
        LOGGER.info("box %s: center=(%.2f,%.2f) R=%.0fkm -> %d hres cells",
                    name, lat0, lon0 % 360.0, radius_km, n)
    return boxes


# ---------------------------------------------------------------------------
# grad-enabled denoiser (twin of model_utils.denoise_at_sigma, no no_grad)
# ---------------------------------------------------------------------------

def _denoise_grad(bundle, x_interp, x_hres, y_residual, sigma, noise):
    """Grad-enabled denoise at fixed sigma. Copies denoise_at_sigma's exact
    calling pattern (sigma_4d shape, no time dim on y_noised) but WITHOUT
    torch.no_grad / autocast so IG can backprop into x_interp / x_hres."""
    inner = bundle.inner_model
    device = bundle.device
    b, e = x_interp.shape[0], x_interp.shape[2]
    sigma_t = torch.tensor(sigma, device=device, dtype=x_interp.dtype)
    sigma_4d = sigma_t.view(1, 1, 1, 1).expand(b, e, 1, 1)
    y_noised = y_residual.to(x_interp.dtype) + sigma_t * noise.to(x_interp.dtype)
    return inner.fwd_with_preconditioning(x_interp, x_hres, y_noised, sigma_4d)


def _functional(out, t_idx, mask_t):
    """Reduce the per-target output FIELD to a SCALAR.

    out: (B, 1, E, N_hres_grid, V_out).  mask_t None => global mean.
    """
    field = out[..., t_idx]            # (B, 1, E, N_hres_grid)
    if mask_t is None:
        return field.mean()
    return field[..., mask_t].mean()


# ---------------------------------------------------------------------------
# Integrated Gradients core (hand-rolled Riemann midpoint; shared forward)
# ---------------------------------------------------------------------------

def integrated_gradients(bundle, prepared, sigma, noise, target_indices,
                         functional_masks, n_steps, baseline_li, baseline_h):
    """Return {(functional, target): (attr_li, attr_h)} for one sigma.

    A single forward per path-step is shared across all (functional, target)
    scalars; one backward per scalar (retain_graph) reuses that forward graph.
    Riemann midpoint rule: alpha = (k + 0.5) / n_steps.
    """
    x_interp = prepared["x_interp"]
    x_hres = prepared["x_hres"]
    y_res = prepared["y_residual"]

    diff_li = x_interp - baseline_li
    diff_h = x_hres - baseline_h

    pairs = [(f, t) for f in functional_masks for t in target_indices]
    grads_li = {key: torch.zeros_like(x_interp) for key in pairs}
    grads_h = {key: torch.zeros_like(x_hres) for key in pairs}

    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        xi = (baseline_li + alpha * diff_li).detach().requires_grad_(True)
        xh = (baseline_h + alpha * diff_h).detach().requires_grad_(True)
        out = _denoise_grad(bundle, xi, xh, y_res, sigma, noise)
        for i, (fkey, tname) in enumerate(pairs):
            scalar = _functional(out, target_indices[tname], functional_masks[fkey])
            gi, gh = torch.autograd.grad(
                scalar, [xi, xh], retain_graph=(i < len(pairs) - 1))
            grads_li[fkey, tname] += gi.detach()
            grads_h[fkey, tname] += gh.detach()
        del out

    attrs = {}
    for key in pairs:
        attr_li = diff_li * (grads_li[key] / n_steps)
        attr_h = diff_h * (grads_h[key] / n_steps)
        attrs[key] = (attr_li.detach(), attr_h.detach())
    return attrs


def _functional_values(bundle, prepared, sigma, noise, target_indices, functional_masks):
    """Scalar F at the true input (alpha=1), for reporting."""
    with torch.no_grad():
        out = _denoise_grad(bundle, prepared["x_interp"], prepared["x_hres"],
                            prepared["y_residual"], sigma, noise)
        return {(f, t): float(_functional(out, target_indices[t], functional_masks[f]).item())
                for f in functional_masks for t in target_indices}


# ---------------------------------------------------------------------------
# summarization
# ---------------------------------------------------------------------------

def _summarize(attr, names):
    """attr: (B,1,E,Ngrid,V) -> per-var {name, mean_abs, signed_mean} + per-cell mean."""
    a = attr.float()
    mean_abs = a.abs().mean(dim=(0, 1, 2, 3)).cpu().numpy()        # (V,)
    signed_mean = a.mean(dim=(0, 1, 2, 3)).cpu().numpy()           # (V,)
    per_cell = a.mean(dim=(0, 1, 2)).cpu().numpy()                 # (Ngrid, V) signed
    per_var = {
        str(v): {"name": names.get(v, f"var_{v}"),
                 "mean_abs": float(mean_abs[v]),
                 "signed_mean": float(signed_mean[v])}
        for v in range(a.shape[-1])
    }
    return per_var, per_cell


def _topk_maps(per_var, per_cell, lat, lon, topk, max_points):
    """Store (downsampled) per-cell signed attribution for the top-K vars by
    mean_abs, so attribution maps can be drawn later by the viz renderer."""
    ranked = sorted(per_var.items(), key=lambda kv: kv[1]["mean_abs"], reverse=True)
    n = per_cell.shape[0]
    stride = max(1, n // max_points) if max_points else 1
    sl = slice(None, None, stride)
    lat_ds = np.asarray(lat)[sl].astype(float).tolist()
    lon_ds = _norm_lon(np.asarray(lon))[sl].astype(float).tolist()
    maps = {}
    for vidx_str, info in ranked[:topk]:
        v = int(vidx_str)
        maps[vidx_str] = {
            "name": info["name"],
            "stride": stride,
            "lat": lat_ds,
            "lon": lon_ds,
            "attr": per_cell[sl, v].astype(float).tolist(),
        }
    return maps


# ---------------------------------------------------------------------------
# data loading (validation dataloader, mirrors interp.feature_permutation)
# ---------------------------------------------------------------------------

def _collect_batches(bundle, n_batches):
    """Stack n_batches validation batches into a single (x_lres, x_hres, y)."""
    bundle.datamodule.setup(stage="predict")
    dl = bundle.datamodule.val_dataloader()
    batches = []
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        batches.append(batch)
    if not batches:
        raise SystemExit("validation dataloader yielded no batches")
    x_lres = torch.cat([b[0] for b in batches], dim=0)
    x_hres = torch.cat([b[1] for b in batches], dim=0)
    y = torch.cat([b[2] for b in batches], dim=0)
    LOGGER.info("collected %d batch(es): x_lres=%s x_hres=%s y=%s",
                len(batches), tuple(x_lres.shape), tuple(x_hres.shape), tuple(y.shape))
    return x_lres, x_hres, y


def _grid_coords(bundle):
    """(lat_lres, lon_lres, lat_hres, lon_hres) from the datamodule.

    Per the AIFSDD convention, latitudes[0]=lres grid, latitudes[2]=hres grid.
    """
    data = bundle.datamodule.ds_valid.data
    lat_lres = np.asarray(data.latitudes[0])
    lon_lres = np.asarray(data.longitudes[0])
    lat_hres = np.asarray(data.latitudes[2])
    lon_hres = np.asarray(data.longitudes[2])
    return lat_lres, lon_lres, lat_hres, lon_hres


# ---------------------------------------------------------------------------
# data loading from REAL event bundles (needed for storm-centered probes;
# the val dataloader has no guaranteed storm at the probe location)
# ---------------------------------------------------------------------------

def _find_bundles(bundle_dir, dates, members, steps):
    paths = []
    for d in dates:
        for m in members:
            for s in steps:
                pat = f"*date{d}*mem{m}*step{s}h*input_bundle.nc"
                hits = sorted(glob.glob(str(Path(bundle_dir) / pat)))
                if not hits:
                    raise FileNotFoundError(f"No bundle matching {pat} in {bundle_dir}")
                paths.append(hits[0])
    return paths


def _collect_bundles(bundle, bundle_dir, dates, members, steps):
    """Build (x_lres, x_hres, y) + grid coords from real event bundles."""
    vn = get_variable_names(bundle)
    n2i_lres = {name: idx for idx, name in vn["input_lres"].items()}
    n2i_hres = {name: idx for idx, name in vn["input_hres"].items()}
    out_states = [vn["output"][i] for i in sorted(vn["output"])]

    paths = _find_bundles(bundle_dir, dates, members, steps)
    xl, xh, ys, coords = [], [], [], None
    for p in paths:
        x_lres_np, x_hres_np, _lon_l, _lat_l, lon_hres, lat_hres = (
            load_inputs_from_bundle_numpy(p, n2i_lres, n2i_hres))
        target_np, _found = extract_target_from_bundle(p, out_states)
        if target_np is None:
            raise SystemExit(f"Could not extract target from bundle {p}")
        xl.append(torch.from_numpy(x_lres_np)[None, None, None, ...])
        xh.append(torch.from_numpy(x_hres_np)[None, None, None, ...])
        ys.append(torch.from_numpy(target_np)[None, None, None, ...])
        coords = (np.asarray(_lat_l), np.asarray(_lon_l),
                  np.asarray(lat_hres), np.asarray(lon_hres))
    x_lres, x_hres, y = torch.cat(xl), torch.cat(xh), torch.cat(ys)
    LOGGER.info("loaded %d bundle(s) from %s: x_lres=%s x_hres=%s y=%s",
                len(paths), bundle_dir, tuple(x_lres.shape), tuple(x_hres.shape), tuple(y.shape))
    return x_lres, x_hres, y, coords, paths


# ---------------------------------------------------------------------------
# spatial coherence + zoom-window attribution maps
# ---------------------------------------------------------------------------

def _coherence(agg, lat, lon, eye_lat, eye_lon):
    """Spatial-coherence summary of a nonneg per-cell influence field `agg`.

    Answers: is the model's influence on the probe LOCAL to the storm or remote?
    """
    d = _haversine_km(eye_lat, _norm_lon(eye_lon), _norm_lon(lat), _norm_lon(lon))
    total = float(agg.sum())
    if total <= 0:
        return {"total_influence": 0.0}
    w = agg / total
    lon_r = np.radians(_norm_lon(lon))
    cx, cy = float((w * np.cos(lon_r)).sum()), float((w * np.sin(lon_r)).sum())
    centroid_lon = float(np.degrees(np.arctan2(cy, cx)))
    centroid_lat = float((w * lat).sum())
    centroid_dist = float(_haversine_km(
        eye_lat, _norm_lon(eye_lon), np.array([centroid_lat]), np.array([centroid_lon]))[0])
    return {
        "total_influence": total,
        "frac_within_200km": float(agg[d <= 200].sum() / total),
        "frac_within_500km": float(agg[d <= 500].sum() / total),
        "frac_within_1000km": float(agg[d <= 1000].sum() / total),
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon % 360.0,
        "centroid_offset_km": centroid_dist,
        "effective_radius_km": float((w * d).sum()),
    }


def _zoom_maps(per_cell, names, lat, lon, center, zoom_deg, topk):
    """Full-resolution windowed attribution maps around the storm center.

    Stores, for cells within `zoom_deg` of `center`: the aggregate influence
    (sum_v |attr|) and the signed attribution of the top-K input variables.
    """
    dlon = (np.asarray(lon) - center[1] + 180.0) % 360.0 - 180.0
    win = (np.abs(lat - center[0]) <= zoom_deg) & (np.abs(dlon) <= zoom_deg)
    agg_full = np.abs(per_cell).sum(axis=1)
    win_abs = np.abs(per_cell[win]).mean(axis=0)
    order = np.argsort(-win_abs)[:topk]
    out = {
        "center_lat": float(center[0]),
        "center_lon": float(center[1] % 360.0),
        "zoom_deg": zoom_deg,
        "n_cells": int(win.sum()),
        "lat": lat[win].astype(float).tolist(),
        "lon": (_norm_lon(lon)[win]).astype(float).tolist(),
        "agg": agg_full[win].astype(float).tolist(),
        "vars": {names.get(int(v), f"var_{v}"): per_cell[win, int(v)].astype(float).tolist()
                 for v in order},
    }
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_integrated_gradients(
    checkpoint_path, output_dir, sigmas, functionals, box_specs, ig_steps,
    n_batches, device, precision, baseline, map_sigma, topk_grid, max_map_points,
    bundle_dir=None, dates=None, members=None, steps=None,
    eye_radius_km=150.0, zoom_deg=12.0, topk_zoom=3, auto_window=None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    auto_window = auto_window or DEFAULT_AUTO_WINDOW

    LOGGER.info("Loading model from %s", checkpoint_path)
    bundle = load_model(checkpoint_path, device=device, precision=precision)
    vn = get_variable_names(bundle)
    target_indices = get_surface_target_indices(bundle)
    LOGGER.info("surface targets: %s", target_indices)

    # Data source: real event bundles (for storm-centered probes) or the val
    # dataloader. Storm-centered work REQUIRES bundles (a real storm in the batch).
    bundle_paths = []
    if bundle_dir:
        x_lres, x_hres, y, coords, bundle_paths = _collect_bundles(
            bundle, bundle_dir, dates or ["20230826"], members or ["01"], steps or ["048"])
        lat_lres, lon_lres, lat_hres, lon_hres = coords
    else:
        x_lres, x_hres, y = _collect_batches(bundle, n_batches)
        lat_lres, lon_lres, lat_hres, lon_hres = _grid_coords(bundle)

    prepared = prepare_batch(bundle, x_lres, x_hres, y)
    xi_grid = prepared["x_interp"].shape[-2]
    xh_grid = prepared["x_hres"].shape[-2]
    LOGGER.info("prepared: x_interp grid=%d (lres=%d hres=%d), x_hres grid=%d",
                xi_grid, len(lat_lres), len(lat_hres), xh_grid)

    # x_interp lives on the HRES grid (lres is interpolated up); x_hres too.
    def _coords_for(n):
        if n == len(lat_hres):
            return lat_hres, lon_hres
        if n == len(lat_lres):
            return lat_lres, lon_lres
        raise ValueError(
            f"grid size {n} matches neither hres({len(lat_hres)}) nor lres({len(lat_lres)})")

    lat_xi, lon_xi = _coords_for(xi_grid)
    lat_xh, lon_xh = _coords_for(xh_grid)

    # Target msl on the hres grid (for storm-center auto-detection).
    msl_out_idx = target_indices.get("msl")
    msl_field = (y[0, 0, 0, :, msl_out_idx].cpu().numpy()
                 if (msl_out_idx is not None and y.shape[-2] == len(lat_hres)) else None)

    # Probe boxes / eye are defined on the OUTPUT (hres) grid the functional reduces over.
    boxes = {}
    if "box" in functionals:
        if not box_specs:
            raise SystemExit("--functionals includes 'box' but no --boxes given")
        boxes = _parse_boxes(box_specs, lat_hres, lon_hres, msl_field, auto_window)

    # functional name -> (output-grid mask tensor | None) and probe center (for coherence)
    functional_masks = {}
    functional_centers = {}
    if "global_mean" in functionals:
        functional_masks["global_mean"] = None
        functional_centers["global_mean"] = None
    for bname, b in boxes.items():
        functional_masks[f"box:{bname}"] = torch.from_numpy(b["mask"]).to(bundle.device)
        functional_centers[f"box:{bname}"] = (b["lat"], b["lon"])

    # The "eye" functional: a tight disk on the storm center = the extreme core.
    if "eye" in functionals:
        if msl_field is None:
            raise SystemExit("'eye' functional needs bundle target msl (use --bundle-dir)")
        eye_lat, eye_lon = _detect_center(msl_field, lat_hres, lon_hres, auto_window)
        eye_mask = _box_mask(lat_hres, lon_hres, eye_lat, eye_lon, eye_radius_km)
        functional_masks["eye"] = torch.from_numpy(eye_mask).to(bundle.device)
        functional_centers["eye"] = (eye_lat, eye_lon)
        boxes["eye"] = {"lat": eye_lat, "lon": eye_lon % 360.0,
                        "radius_km": eye_radius_km, "n_cells": int(eye_mask.sum())}
        LOGGER.info("eye: center=(%.2f,%.2f) R=%.0fkm -> %d hres cells",
                    eye_lat, eye_lon % 360.0, eye_radius_km, int(eye_mask.sum()))

    if not functional_masks:
        raise SystemExit(f"no valid functionals selected from {functionals!r}")

    # Baselines for the IG path integral.
    if baseline == "zeros":
        base_li = torch.zeros_like(prepared["x_interp"])
        base_h = torch.zeros_like(prepared["x_hres"])
    elif baseline == "mean":
        base_li = prepared["x_interp"].mean(dim=(0, 1, 2, 3), keepdim=True).expand_as(
            prepared["x_interp"]).contiguous()
        base_h = prepared["x_hres"].mean(dim=(0, 1, 2, 3), keepdim=True).expand_as(
            prepared["x_hres"]).contiguous()
    else:
        raise SystemExit(f"unknown --baseline {baseline!r}")

    lres_names = vn["input_lres"]   # idx -> name
    hres_names = vn["input_hres"]
    spatial = {f for f in functional_masks if functional_masks[f] is not None}

    all_results = {
        "checkpoint": checkpoint_path,
        "ckpt_id": ckpt_id_from_path(checkpoint_path),
        "captum": bool(_HAVE_CAPTUM),
        "ig_method": "captum" if _HAVE_CAPTUM else "riemann_midpoint",
        "data_source": ("bundles" if bundle_dir else "val_dataloader"),
        "bundle_paths": [str(p) for p in bundle_paths],
        "surface_targets": list(target_indices.keys()),
        "functionals": list(functional_masks.keys()),
        "boxes": {n: {k: v for k, v in b.items() if k != "mask"} for n, b in boxes.items()},
        "ig_steps": ig_steps,
        "baseline": baseline,
        "n_batches": n_batches,
        "batch_size": int(x_lres.shape[0]),
        "sigmas": list(sigmas),
        "map_sigma": map_sigma,
        "zoom_deg": zoom_deg,
        "lres_var_names": {str(k): v for k, v in lres_names.items()},
        "hres_var_names": {str(k): v for k, v in hres_names.items()},
        "results": [],
    }

    for sigma in sigmas:
        LOGGER.info("IG at sigma=%.3f", sigma)
        # noise held FIXED across the whole baseline->input path for this sigma
        noise = torch.randn_like(prepared["y_residual"])
        attrs = integrated_gradients(
            bundle, prepared, sigma, noise, target_indices, functional_masks,
            ig_steps, base_li, base_h)
        fvals = _functional_values(bundle, prepared, sigma, noise,
                                   target_indices, functional_masks)
        store_maps = (map_sigma is not None and abs(sigma - map_sigma) < 1e-9)

        for fkey in functional_masks:
            for tname in target_indices:
                attr_li, attr_h = attrs[fkey, tname]
                lres_pv, lres_cell = _summarize(attr_li, lres_names)
                hres_pv, hres_cell = _summarize(attr_h, hres_names)
                entry = {
                    "sigma": sigma,
                    "functional": fkey,
                    "target": tname,
                    "F_value": fvals[fkey, tname],
                    "lres": lres_pv,
                    "hres": hres_pv,
                }
                # Spatial coherence of the input influence on this probe (all sigmas).
                if fkey in spatial and functional_centers.get(fkey):
                    c = functional_centers[fkey]
                    agg_lres = np.abs(lres_cell).sum(axis=1)
                    entry["coherence"] = _coherence(agg_lres, lat_xi, lon_xi, c[0], c[1])
                # Per-cell maps only for storm-centered probes (full-res zoom of WHERE
                # the input drivers live). global_mean keeps only the per-var summaries
                # above — a global per-grid map would bloat the JSON and the renderer
                # does not use it.
                if store_maps and fkey in spatial and functional_centers.get(fkey):
                    entry["zoom"] = _zoom_maps(
                        lres_cell, lres_names, lat_xi, lon_xi,
                        functional_centers[fkey], zoom_deg, topk_zoom)
                all_results["results"].append(entry)

    results_file = output_path / "integrated_gradients.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info("Results saved to %s", results_file)
    _print_summary(all_results)
    return all_results


def _print_summary(all_results):
    print("\n" + "=" * 72)
    print("INTEGRATED GRADIENTS — top input vars by mean|attribution| per target")
    print("=" * 72)
    by_sig = {}
    for e in all_results["results"]:
        by_sig.setdefault(e["sigma"], []).append(e)
    for sigma in sorted(by_sig):
        print(f"\nSigma = {sigma:g}")
        for e in by_sig[sigma]:
            ranked = sorted(
                list(e["lres"].items()) + list(e["hres"].items()),
                key=lambda kv: kv[1]["mean_abs"], reverse=True)
            top = ranked[:5]
            label = f"{e['functional']} | {e['target']}"
            cells = ", ".join(
                f"{i['name']}({'+' if i['signed_mean'] >= 0 else '-'}{i['mean_abs']:.2e})"
                for _, i in top)
            print(f"  {label:24s}  F={e['F_value']:+.3e}  top5: {cells}")


def main():
    p = argparse.ArgumentParser(description="Integrated Gradients (Tier-2) for AIFSDD")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.1, 1.0, 5.0, 20.0, 80.0])
    p.add_argument("--n-batches", type=int, default=2)
    p.add_argument("--functionals", default="global_mean,box",
                   help="comma list of: global_mean, box")
    p.add_argument("--boxes", nargs="*", default=["franklin:auto"],
                   help="name:lat,lon,radiuskm  or  name:auto[,radiuskm]")
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--baseline", default="zeros", choices=["zeros", "mean"],
                   help="IG baseline: zeros (default) or per-variable climatology mean")
    p.add_argument("--map-sigma", type=float, default=5.0,
                   help="store per-cell attribution maps only at this sigma")
    p.add_argument("--topk-grid", type=int, default=8,
                   help="keep full per-grid attribution for the top-K vars (for maps)")
    p.add_argument("--max-map-points", type=int, default=20000,
                   help="downsample stored maps to at most this many grid points")
    # Storm-centered probe options (require real event bundles).
    p.add_argument("--bundle-dir", default=None,
                   help="event bundle dir, e.g. ~/hpcperm/data/input_data/o96_o320/idalia "
                        "(if set, loads real bundles instead of the val dataloader)")
    p.add_argument("--dates", nargs="+", default=["20230826"])
    p.add_argument("--members", nargs="+", default=["01"])
    p.add_argument("--steps", nargs="+", default=["048"], help="zero-padded step hours, e.g. 048")
    p.add_argument("--eye-radius-km", type=float, default=150.0,
                   help="radius of the 'eye' extreme-core functional")
    p.add_argument("--zoom-deg", type=float, default=12.0,
                   help="half-width (deg) of stored zoom maps around the storm center")
    p.add_argument("--topk-zoom", type=int, default=3,
                   help="per-variable zoom maps for the top-K input vars")
    p.add_argument("--auto-window", default=None,
                   help="lat0,lat1,lon0,lon1 (deg, lon 0..360) for storm auto-detect")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    window = tuple(float(x) for x in args.auto_window.split(",")) if args.auto_window else None

    run_integrated_gradients(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        sigmas=args.sigmas,
        functionals=[f.strip() for f in args.functionals.split(",") if f.strip()],
        box_specs=args.boxes,
        ig_steps=args.ig_steps,
        n_batches=args.n_batches,
        device=args.device,
        precision=args.precision,
        baseline=args.baseline,
        map_sigma=args.map_sigma,
        topk_grid=args.topk_grid,
        max_map_points=args.max_map_points,
        bundle_dir=args.bundle_dir,
        dates=args.dates,
        members=args.members,
        steps=args.steps,
        eye_radius_km=args.eye_radius_km,
        zoom_deg=args.zoom_deg,
        topk_zoom=args.topk_zoom,
        auto_window=window,
    )


if __name__ == "__main__":
    main()
