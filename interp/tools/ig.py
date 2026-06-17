"""Integrated Gradients (Tier-2) for AIFSDD — per surface target, storm-centered.

Reduces each surface-target OUTPUT field to a SCALAR functional, then attributes
that scalar back to the full conditioning INPUT fields (x_interp lres + x_hres)
via a hand-rolled Riemann-sum Integrated Gradients.

Functionals (--functionals, comma list):
  - global_mean : mean of the target field over all hres cells.
  - box         : mean over hres cells within R km of a probe center
                  ("name:lat,lon,radiuskm" or "name:auto[,radiuskm]").
  - eye         : tight extreme-core disk at the auto-detected storm center.
  - tail        : mean over the cells where |observed y_target| is above
                  --tail-percentile (within --tail-region). "What drives the
                  model exactly where the field is extreme?"
  - spectral    : high-wavenumber power of the target field over
                  --spectral-region (interp.core.regions.functional_spectral_high).
                  "Which inputs create the small scales?"

Baseline is zeros (default) or a per-variable climatology mean (--baseline mean).
Attribution is computed at several sigmas since it is noise-regime dependent. The
diffusion noise is held FIXED across the baseline->input path so the path
integral is well defined.

Data ALWAYS comes from real event bundles (--event / --bundle-dir); the val
dataloader path was removed (broken lres grid + no storm at the probe). Run
with ONE bundle: the grad-enabled forward is batched and the encoder
mis-assembles batches > 1.

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp ig \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir ~/perm/interp/<ckpt_id>/integrated_gradients \
        --event franklin_o96_o320 \
        --functionals global_mean,eye,tail,spectral --ig-steps 32
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.cli import add_event_args, add_model_args, add_sigma_args, setup_logging
from interp.core.data import collect_event_bundles, resolve_event_args
from interp.core.geometry import (
    DEFAULT_AUTO_WINDOW,
    box_mask_km,
    detect_min_center,
    haversine_km,
    norm_lon,
    parse_boxes,
)
from interp.core.model import (
    denoise_at_sigma_grad,
    get_surface_target_indices,
    get_variable_names,
    load_model,
    prepare_batch,
)
from interp.core.regions import (
    build_extreme_mask,
    build_region_mask,
    functional_spectral_high,
)
from interp.core.runmeta import ckpt_id_from_path, write_run_meta

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# functionals
# ---------------------------------------------------------------------------

def _apply_functional(out, t_idx, tname, spec):
    """Reduce the per-target output FIELD (B,1,E,G) to a SCALAR (differentiable)."""
    field = out[..., t_idx]
    if spec["kind"] == "spectral":
        f1 = field.reshape(-1, field.shape[-1]).mean(dim=0)  # (G,), keeps grad
        return functional_spectral_high(
            f1, spec["mask"], spec["lat"], spec["lon"],
            cutoff_frac=spec["cutoff"], n_grid=spec["ngrid"])
    mask = spec["mask"]
    if isinstance(mask, dict):
        mask = mask[tname]
    if mask is None:
        return field.mean()
    return field[..., mask].mean()


def build_functional_specs(args, bundle, target_indices, y, lat_hres, lon_hres,
                           probe_field, functionals, box_specs):
    """Resolve every requested functional into {fkey: spec dict}.

    spec = {"kind": "mean"|"spectral", "mask": Tensor|dict|None,
            "center": (lat,lon)|None, ...extras}
    """
    dev = bundle.device
    specs: dict[str, dict] = {}
    boxes_meta: dict[str, dict] = {}

    if "global_mean" in functionals:
        specs["global_mean"] = {"kind": "mean", "mask": None, "center": None}

    if "box" in functionals:
        if not box_specs:
            raise SystemExit("--functionals includes 'box' but no --boxes given")
        boxes = parse_boxes(box_specs, lat_hres, lon_hres, probe_field,
                            args_window(args))
        for bname, b in boxes.items():
            specs[f"box:{bname}"] = {
                "kind": "mean",
                "mask": torch.from_numpy(b["mask"]).to(dev),
                "center": (b["lat"], b["lon"]),
            }
            boxes_meta[bname] = {k: v for k, v in b.items() if k != "mask"}

    if "eye" in functionals:
        if probe_field is None:
            raise SystemExit(f"'eye' functional needs the bundle target "
                             f"{args.probe_field} field")
        eye_lat, eye_lon = detect_min_center(probe_field, lat_hres, lon_hres,
                                             args_window(args))
        eye_mask = box_mask_km(lat_hres, lon_hres, eye_lat, eye_lon, args.eye_radius_km)
        specs["eye"] = {"kind": "mean",
                        "mask": torch.from_numpy(eye_mask).to(dev),
                        "center": (eye_lat, eye_lon)}
        boxes_meta["eye"] = {"lat": eye_lat, "lon": eye_lon % 360.0,
                             "radius_km": args.eye_radius_km,
                             "n_cells": int(eye_mask.sum())}
        LOGGER.info("eye: center=(%.2f,%.2f) R=%.0fkm -> %d hres cells",
                    eye_lat, eye_lon % 360.0, args.eye_radius_km, int(eye_mask.sum()))

    lat_t = torch.as_tensor(lat_hres, dtype=torch.float64)
    lon_t = torch.as_tensor(lon_hres, dtype=torch.float64)

    if "tail" in functionals:
        region_mask = build_region_mask(lat_t, lon_t, args.tail_region)
        masks, sides = {}, {}
        for tname, tidx in target_indices.items():
            side = resolve_tail_side(args.tail_side, tname)
            field = y[..., tidx].reshape(-1, y.shape[-2]).mean(dim=0)  # (G,)
            m = build_extreme_mask(field, args.tail_percentile,
                                   region_mask=region_mask, side=side)
            masks[tname] = m.to(dev)
            sides[tname] = side
            LOGGER.info("tail mask %s: p%g (%s tail) in %s -> %d cells",
                        tname, args.tail_percentile, side, args.tail_region, int(m.sum()))
        specs["tail"] = {"kind": "mean", "mask": masks, "center": None,
                         "percentile": args.tail_percentile, "region": args.tail_region,
                         "sides": sides}

    if "spectral" in functionals:
        smask = build_region_mask(lat_t, lon_t, args.spectral_region).to(dev)
        specs[f"spectral:{args.spectral_region}"] = {
            "kind": "spectral",
            "mask": smask,
            "center": None,
            "lat": lat_t.to(dev),
            "lon": lon_t.to(dev),
            "cutoff": args.spectral_cutoff,
            "ngrid": args.spectral_ngrid,
        }
        LOGGER.info("spectral: region=%s (%d cells) cutoff=%.2f ngrid=%d",
                    args.spectral_region, int(smask.sum()),
                    args.spectral_cutoff, args.spectral_ngrid)

    if not specs:
        raise SystemExit(f"no valid functionals selected from {functionals!r}")
    return specs, boxes_meta


def args_window(args):
    if args.auto_window:
        return tuple(float(x) for x in args.auto_window.split(","))
    return DEFAULT_AUTO_WINDOW


def resolve_tail_side(side: str, tname: str) -> str:
    """'auto' = low tail for msl (cyclones; |msl| would select anticyclones),
    abs tail for everything else (winds, tp, temperature)."""
    if side != "auto":
        return side
    return "low" if tname == "msl" else "abs"


# ---------------------------------------------------------------------------
# Integrated Gradients core (hand-rolled Riemann midpoint; shared forward)
# ---------------------------------------------------------------------------

def integrated_gradients(bundle, prepared, sigma, noise, target_indices,
                         specs, n_steps, baseline_li, baseline_h, baseline_y,
                         pairs_per_pass=8):
    """Return {(functional, target): (attr_li, attr_h, attr_y)} for one sigma.

    Three conditioning pathways are attributed along the SAME baseline->input
    path: lres conditioning (x_interp), hres forcings (x_hres) and the NOISED
    TARGET itself (y_residual, the 'noisy_hres' pathway — what the denoiser is
    handed to refine). The diffusion noise is held fixed; only the clean signal
    y_residual is integrated from its baseline.

    One forward per path-step is shared across a CHUNK of (functional, target)
    scalars; one backward per scalar (retain_graph) reuses that forward graph.
    Chunking bounds GPU memory: each retained backward through an
    activation-checkpointed segment keeps its recomputed activations alive, so
    too many backwards on one graph OOMs (observed at 20 pairs on a 40 GB
    GA100). Extra forwards are cheap by comparison. Riemann midpoint rule:
    alpha = (k + 0.5) / n_steps.
    """
    x_interp = prepared["x_interp"]
    x_hres = prepared["x_hres"]
    y_res = prepared["y_residual"]

    diff_li = x_interp - baseline_li
    diff_h = x_hres - baseline_h
    diff_y = y_res - baseline_y

    pairs = [(f, t) for f in specs for t in target_indices]
    chunks = [pairs[i:i + pairs_per_pass] for i in range(0, len(pairs), pairs_per_pass)]
    grads_li = {key: torch.zeros_like(x_interp) for key in pairs}
    grads_h = {key: torch.zeros_like(x_hres) for key in pairs}
    grads_y = {key: torch.zeros_like(y_res) for key in pairs}

    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        for chunk in chunks:
            xi = (baseline_li + alpha * diff_li).detach().requires_grad_(True)
            xh = (baseline_h + alpha * diff_h).detach().requires_grad_(True)
            yt = (baseline_y + alpha * diff_y).detach().requires_grad_(True)
            out = denoise_at_sigma_grad(bundle, xi, xh, yt, sigma, noise)
            for i, (fkey, tname) in enumerate(chunk):
                scalar = _apply_functional(out, target_indices[tname], tname, specs[fkey])
                gi, gh, gy = torch.autograd.grad(
                    scalar, [xi, xh, yt], retain_graph=(i < len(chunk) - 1))
                grads_li[fkey, tname] += gi.detach()
                grads_h[fkey, tname] += gh.detach()
                grads_y[fkey, tname] += gy.detach()
            del out, xi, xh, yt

    attrs = {}
    for key in pairs:
        attr_li = diff_li * (grads_li[key] / n_steps)
        attr_h = diff_h * (grads_h[key] / n_steps)
        attr_y = diff_y * (grads_y[key] / n_steps)
        attrs[key] = (attr_li.detach(), attr_h.detach(), attr_y.detach())
    return attrs


def _functional_values(bundle, prepared, sigma, noise, target_indices, specs):
    """Scalar F at the true input (alpha=1), for reporting."""
    with torch.no_grad():
        out = denoise_at_sigma_grad(bundle, prepared["x_interp"], prepared["x_hres"],
                                    prepared["y_residual"], sigma, noise)
        return {(f, t): float(_apply_functional(out, target_indices[t], t, specs[f]).item())
                for f in specs for t in target_indices}


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


# Probe-relative locality buffer (km) added to the output-disk radius: "how
# much influence sits inside the disk, and inside the disk + this buffer".
PROBE_BUFFER_KM = 350.0
# Distance-grid edges (km) for the radial cumulative-attribution profiles.
RADIAL_EDGES_KM = list(range(0, 3001, 100))


def _coherence(agg, lat, lon, eye_lat, eye_lon, probe_r_km=None):
    """Spatial-coherence summary of a nonneg per-cell influence field `agg`.

    Answers: is the model's influence on the probe LOCAL to the storm or remote?
    Distances are measured from THIS member's own probe centre, so the summary
    is storm-relative and safe to average across ensemble members. When the
    output-disk radius is known, also reports the fraction inside the disk and
    inside disk + PROBE_BUFFER_KM (probe-relative locality, item 3).
    """
    d = haversine_km(eye_lat, norm_lon(eye_lon), norm_lon(lat), norm_lon(lon))
    total = float(agg.sum())
    if total <= 0:
        return {"total_influence": 0.0}
    w = agg / total
    lon_r = np.radians(norm_lon(lon))
    cx, cy = float((w * np.cos(lon_r)).sum()), float((w * np.sin(lon_r)).sum())
    centroid_lon = float(np.degrees(np.arctan2(cy, cx)))
    centroid_lat = float((w * lat).sum())
    centroid_dist = float(haversine_km(
        eye_lat, norm_lon(eye_lon), np.array([centroid_lat]), np.array([centroid_lon]))[0])
    out = {
        "total_influence": total,
        "frac_within_200km": float(agg[d <= 200].sum() / total),
        "frac_within_500km": float(agg[d <= 500].sum() / total),
        "frac_within_1000km": float(agg[d <= 1000].sum() / total),
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon % 360.0,
        "centroid_offset_km": centroid_dist,
        "effective_radius_km": float((w * d).sum()),
    }
    if probe_r_km:
        out["probe_radius_km"] = float(probe_r_km)
        out["frac_within_probe"] = float(agg[d <= probe_r_km].sum() / total)
        out["frac_within_probe_plus_buffer"] = float(
            agg[d <= probe_r_km + PROBE_BUFFER_KM].sum() / total)
        out["buffer_km"] = PROBE_BUFFER_KM
    return out


def _radial_cdf(agg_per_var, names, lat, lon, center, edges):
    """Cumulative |attribution| vs distance, per variable, storm-relative.

    agg_per_var: (n_cells, n_vars) NONNEG influence (already |.| and summed over
    targets). Returns {varname: [cum-fraction at each edge]} measured from
    `center`. Storm-relative, so averageable across members.
    """
    d = haversine_km(center[0], norm_lon(center[1]), norm_lon(lat), norm_lon(lon))
    edges = np.asarray(edges, dtype=float)
    within = d[:, None] <= edges[None, :]            # (n_cells, n_edges)
    cum = (agg_per_var[:, :, None] * within[:, None, :]).sum(axis=0)  # (n_vars, n_edges)
    tot = agg_per_var.sum(axis=0)                     # (n_vars,)
    tot = np.where(tot > 0, tot, 1.0)
    frac = cum / tot[:, None]
    return {names.get(v, f"var_{v}"): frac[v].astype(float).tolist()
            for v in range(agg_per_var.shape[1])}


def _zoom_maps(per_cell, names, lat, lon, center, zoom_deg, topk,
               obs=None, must_include=None):
    """Full-resolution windowed attribution maps around the probe center.

    `obs`: the OBSERVED target field (n_cells,) — stored so the renderer can
    overlay its contours (you see the storm, not just dots). `must_include`:
    input-variable index always kept among the maps (the target's own field),
    even when it is not in the top-k by |attr|.
    """
    dlon = (np.asarray(lon) - center[1] + 180.0) % 360.0 - 180.0
    win = (np.abs(lat - center[0]) <= zoom_deg) & (np.abs(dlon) <= zoom_deg)
    agg_full = np.abs(per_cell).sum(axis=1)
    win_abs = np.abs(per_cell[win]).mean(axis=0)
    order = list(np.argsort(-win_abs))
    if must_include is not None and must_include in order:
        order.remove(must_include)
        order.insert(0, must_include)
    order = order[:topk]
    out = {
        "center_lat": float(center[0]),
        "center_lon": float(center[1] % 360.0),
        "zoom_deg": zoom_deg,
        "n_cells": int(win.sum()),
        "lat": lat[win].astype(float).tolist(),
        "lon": (norm_lon(lon)[win]).astype(float).tolist(),
        "agg": agg_full[win].astype(float).tolist(),
        "vars": {names.get(int(v), f"var_{v}"): per_cell[win, int(v)].astype(float).tolist()
                 for v in order},
    }
    if obs is not None:
        out["obs"] = np.asarray(obs)[win].astype(float).tolist()
    return out


# ---------------------------------------------------------------------------
# multi-member averaging helpers (alignment-safe: scalars & storm-relative
# locality are averaged across ensemble members; maps come from member 0)
# ---------------------------------------------------------------------------

def _baselines(kind, prepared):
    xi, xh, yr = prepared["x_interp"], prepared["x_hres"], prepared["y_residual"]
    if kind == "zeros":
        return torch.zeros_like(xi), torch.zeros_like(xh), torch.zeros_like(yr)
    if kind == "mean":
        def _m(t):
            return t.mean(dim=(0, 1, 2, 3), keepdim=True).expand_as(t).contiguous()
        return _m(xi), _m(xh), _m(yr)
    raise SystemExit(f"unknown --baseline {kind!r}")


def _accum_pv(dst, pv):
    for k, v in pv.items():
        d = dst.setdefault(k, {"name": v["name"], "mean_abs": 0.0, "signed_mean": 0.0})
        d["mean_abs"] += v["mean_abs"]
        d["signed_mean"] += v["signed_mean"]


def _mean_pv(dst, n):
    return {k: {"name": v["name"], "mean_abs": v["mean_abs"] / n,
                "signed_mean": v["signed_mean"] / n} for k, v in dst.items()}


def _accum_dict(dst, d):
    for k, v in d.items():
        if isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0.0) + float(v)


def _accum_curves(dst, curves):
    for name, ys in curves.items():
        prev = dst.get(name)
        dst[name] = list(ys) if prev is None else [a + b for a, b in zip(prev, ys)]


def _probe_radius(fkey, boxes_meta):
    key = fkey.split(":", 1)[1] if fkey.startswith("box:") else fkey
    return (boxes_meta.get(key, {}) or {}).get("radius_km")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_integrated_gradients(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    functionals = [f.strip() for f in args.functionals.split(",") if f.strip()]

    LOGGER.info("Loading model from %s", args.checkpoint)
    bundle = load_model(args.checkpoint, device=args.device, precision=args.precision)
    vn = get_variable_names(bundle)
    target_indices = get_surface_target_indices(bundle)
    LOGGER.info("surface targets: %s", target_indices)

    bundle_dir, dates, members, steps, _ = resolve_event_args(args)
    eb = collect_event_bundles(bundle, bundle_dir, dates, members, steps)
    n_members = int(eb.x_lres.shape[0])
    lat_lres, lon_lres, lat_hres, lon_hres = eb.coords
    LOGGER.info("IG over %d ensemble member(s): maps + storm centre from member 0; "
                "driver bars, coherence and radial locality averaged storm-relative.",
                n_members)

    lres_names = vn["input_lres"]            # idx -> name (lres conditioning)
    hres_names = vn["input_hres"]            # idx -> name (hres forcings)
    ntgt_names = vn.get("output", {})        # idx -> name (noised-target channels)

    def _coords_for(n):
        if n == len(lat_hres):
            return lat_hres, lon_hres
        if n == len(lat_lres):
            return lat_lres, lon_lres
        raise ValueError(f"grid size {n} matches neither hres nor lres")

    # Cross-member accumulators (alignment-safe averaging).
    acc = {}            # (fkey, tname, sigma) -> running sums of scalars/coherence
    member0_zoom = {}   # (fkey, tname, sigma) -> zoom maps from member 0 only
    radial_acc = {}     # (fkey, sigma) -> running sums of per-var radial CDFs
    specs0 = boxes_meta0 = None
    lat_xi = lon_xi = None

    for m in range(n_members):
        prepared = prepare_batch(bundle, eb.x_lres[m:m + 1], eb.x_hres[m:m + 1],
                                 eb.y[m:m + 1])
        y_m = eb.y[m:m + 1].to(bundle.device)
        lat_xi, lon_xi = _coords_for(prepared["x_interp"].shape[-2])

        # Probe field for THIS member, so the eye is centred on its own storm.
        probe_field = None
        if y_m.shape[-2] == len(lat_hres):
            p_idx = target_indices.get(args.probe_field)
            if p_idx is not None:
                f = y_m[0, 0, 0, :, p_idx].cpu().numpy()
                probe_field = f if args.probe_field == "msl" else -f
        specs, boxes_meta = build_functional_specs(
            args, bundle, target_indices, y_m, lat_hres, lon_hres, probe_field,
            functionals, args.boxes)
        if m == 0:
            specs0, boxes_meta0 = specs, boxes_meta
        base_li, base_h, base_y = _baselines(args.baseline, prepared)

        for sigma in args.sigmas:
            LOGGER.info("member %d/%d, IG at sigma=%.3f", m, n_members, sigma)
            noise = torch.randn_like(prepared["y_residual"])
            attrs = integrated_gradients(
                bundle, prepared, sigma, noise, target_indices, specs,
                args.ig_steps, base_li, base_h, base_y,
                pairs_per_pass=args.pairs_per_pass)
            fvals = _functional_values(bundle, prepared, sigma, noise,
                                       target_indices, specs)

            for fkey, spec in specs.items():
                center = spec.get("center")
                probe_r = _probe_radius(fkey, boxes_meta)
                agg = {"lres": None, "hres": None, "ntgt": None}  # sum |attr| over targets
                cells = {}
                for tname in target_indices:
                    attr_li, attr_h, attr_y = attrs[fkey, tname]
                    lres_pv, lres_cell = _summarize(attr_li, lres_names)
                    hres_pv, hres_cell = _summarize(attr_h, hres_names)
                    ntgt_pv, ntgt_cell = _summarize(attr_y, ntgt_names)
                    key = (fkey, tname, sigma)
                    a = acc.setdefault(key, {"F": 0.0, "n": 0, "lres": {},
                                             "hres": {}, "ntgt": {}, "coh": {}})
                    a["n"] += 1
                    a["F"] += fvals[fkey, tname]
                    _accum_pv(a["lres"], lres_pv)
                    _accum_pv(a["hres"], hres_pv)
                    _accum_pv(a["ntgt"], ntgt_pv)
                    if center is not None:
                        coh = _coherence(np.abs(lres_cell).sum(axis=1), lat_xi, lon_xi,
                                         center[0], center[1], probe_r_km=probe_r)
                        _accum_dict(a["coh"], coh)
                        for grp, cell in (("lres", lres_cell), ("hres", hres_cell),
                                          ("ntgt", ntgt_cell)):
                            agg[grp] = np.abs(cell) if agg[grp] is None else agg[grp] + np.abs(cell)
                        if m == 0:
                            t_idx = target_indices[tname]
                            obs = (y_m[0, 0, 0, :, t_idx].cpu().numpy()
                                   if y_m.shape[-2] == len(lat_xi) else None)
                            self_idx = next((i for i, n in lres_names.items()
                                             if n == tname), None)
                            member0_zoom[key] = _zoom_maps(
                                lres_cell, lres_names, lat_xi, lon_xi, center,
                                args.zoom_deg, args.topk_zoom, obs=obs,
                                must_include=self_idx)
                if center is not None and agg["lres"] is not None:
                    rk = (fkey, sigma)
                    r = radial_acc.setdefault(rk, {"n": 0, "lres": {}, "hres": {},
                                                   "ntgt": {}})
                    r["n"] += 1
                    _accum_curves(r["lres"], _radial_cdf(agg["lres"], lres_names,
                                  lat_xi, lon_xi, center, RADIAL_EDGES_KM))
                    _accum_curves(r["hres"], _radial_cdf(agg["hres"], hres_names,
                                  lat_xi, lon_xi, center, RADIAL_EDGES_KM))
                    _accum_curves(r["ntgt"], _radial_cdf(agg["ntgt"], ntgt_names,
                                  lat_xi, lon_xi, center, RADIAL_EDGES_KM))

    # ---- assemble averaged results -----------------------------------------
    all_results = {
        "checkpoint": args.checkpoint,
        "ckpt_id": ckpt_id_from_path(args.checkpoint),
        "ig_method": "riemann_midpoint",
        "data_source": "bundles",
        "bundle_paths": [str(p) for p in eb.paths],
        "surface_targets": list(target_indices.keys()),
        "functionals": list(specs0.keys()),
        "boxes": boxes_meta0,
        "ig_steps": args.ig_steps,
        "baseline": args.baseline,
        "batch_size": n_members,
        "n_samples": n_members,
        "maps_from_member": 0,
        "sigmas": list(args.sigmas),
        "map_sigma": args.map_sigma,
        "zoom_deg": args.zoom_deg,
        "probe_buffer_km": PROBE_BUFFER_KM,
        "tail_percentile": getattr(args, "tail_percentile", None),
        "lres_var_names": {str(k): v for k, v in lres_names.items()},
        "hres_var_names": {str(k): v for k, v in hres_names.items()},
        "ntgt_var_names": {str(k): v for k, v in ntgt_names.items()},
        "results": [],
        "radial_locality": [],
    }
    for (fkey, tname, sigma), a in acc.items():
        n = a["n"]
        entry = {"sigma": sigma, "functional": fkey, "target": tname,
                 "F_value": a["F"] / n,
                 "lres": _mean_pv(a["lres"], n), "hres": _mean_pv(a["hres"], n),
                 "ntgt": _mean_pv(a["ntgt"], n)}
        if a["coh"]:
            entry["coherence"] = {k: v / n for k, v in a["coh"].items()}
        z = member0_zoom.get((fkey, tname, sigma))
        if z is not None:
            entry["zoom"] = z
        all_results["results"].append(entry)
    for (fkey, sigma), r in radial_acc.items():
        n = r["n"]
        all_results["radial_locality"].append({
            "functional": fkey, "sigma": sigma, "edges_km": RADIAL_EDGES_KM,
            "lres": {k: [y / n for y in ys] for k, ys in r["lres"].items()},
            "hres": {k: [y / n for y in ys] for k, ys in r["hres"].items()},
            "ntgt": {k: [y / n for y in ys] for k, ys in r["ntgt"].items()},
        })

    results_file = output_path / "integrated_gradients.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info("Results saved to %s", results_file)
    write_run_meta(output_path, "ig", args)
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
            print(f"  {label:28s}  F={e['F_value']:+.3e}  top5: {cells}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Integrated Gradients (Tier-2) for AIFSDD")
    add_model_args(p)
    add_event_args(p)
    # IG default sigma ladder: 1, 5, 50 (low / mid / high noise regime).
    # Always these three unless overridden on the CLI.
    add_sigma_args(p, default=[1.0, 5.0, 50.0])
    p.add_argument("--functionals", default="global_mean,box,eye",
                   help="comma list of: global_mean, box, eye, tail, spectral")
    p.add_argument("--boxes", nargs="*", default=["franklin:auto"],
                   help="name:lat,lon,radiuskm  or  name:auto[,radiuskm]")
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--pairs-per-pass", type=int, default=8,
                   help="(functional,target) backwards sharing one forward graph; "
                        "lower if CUDA OOM (memory grows with retained backwards)")
    p.add_argument("--baseline", default="zeros", choices=["zeros", "mean"],
                   help="IG baseline: zeros (default) or per-variable climatology mean")
    p.add_argument("--map-sigma", type=float, default=5.0,
                   help="the renderer's DEFAULT display sigma for the zoom maps "
                        "(maps themselves are stored at every sigma)")
    p.add_argument("--eye-radius-km", type=float, default=150.0,
                   help="radius of the 'eye' extreme-core functional")
    p.add_argument("--zoom-deg", type=float, default=12.0,
                   help="half-width (deg) of stored zoom maps around the storm center")
    p.add_argument("--topk-zoom", type=int, default=8,
                   help="per-variable zoom maps for the top-K input vars "
                        "(stored at every sigma; the renderer shows a subset)")
    p.add_argument("--auto-window", default=None,
                   help="lat0,lat1,lon0,lon1 (deg, lon 0..360) for storm auto-detect")
    p.add_argument("--probe-field", default="msl", choices=["msl", "tp"],
                   help="field whose extremum centers box:auto/eye probes: "
                        "msl minimum (cyclone) or tp maximum (intense precip)")
    p.add_argument("--tail-percentile", type=float, default=99.0,
                   help="[tail] percentile of y_target selecting extreme cells")
    p.add_argument("--tail-region", default="global",
                   help="[tail] region within which the percentile is taken")
    p.add_argument("--tail-side", default="auto", choices=["auto", "abs", "high", "low"],
                   help="[tail] which tail; auto = low for msl (cyclones), abs otherwise")
    p.add_argument("--spectral-region", default="amazon_rainforest",
                   help="[spectral] region whose high-k power is the functional")
    p.add_argument("--spectral-cutoff", type=float, default=0.5)
    p.add_argument("--spectral-ngrid", type=int, default=64)
    args = p.parse_args(argv)
    setup_logging()
    return run_integrated_gradients(args)


if __name__ == "__main__":
    main()
