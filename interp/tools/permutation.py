"""Feature permutation importance for AIFSDD.

Shuffle each input variable independently across the batch and measure output
degradation against a noise-paired baseline. No gradients, no hooks.

Two modes (previously two near-identical scripts):
  --mode sigma     which inputs matter for the DENOISER at each noise level?
                   (one denoiser call per sigma, fixed noise)
  --mode sampling  which inputs matter for the FINAL OUTPUT?
                   (full Heun trajectory, paired seeds)

Both modes load real event bundles (never val_dataloader), score per surface
target, per region, and — with --extreme-percentile — over the tail cells of
each target field (tail-conditioned importance: "which inputs matter for the
extremes").

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp permutation \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir /path/to/results/ \
        --event franklin_o96_o320_m4 \
        --mode sigma --n-repeats 5 --extreme-percentile 99
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
from interp.core.model import (
    denoise_at_sigma,
    get_surface_target_indices,
    get_variable_names,
    load_model,
    per_target_mse,
    prepare_batch,
    sample_full,
)
from interp.core.regions import (
    build_extreme_mask,
    build_region_mask,
    get_area_weights,
    get_hres_lat_lon,
    masked_area_mean,
)
from interp.core.runmeta import write_run_meta

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# masks + scoring
# ---------------------------------------------------------------------------

def resolve_tail_side(side: str, tname: str) -> str:
    """'auto' = low tail for msl (cyclones; |msl| would select anticyclones),
    abs tail for everything else (winds, tp, temperature)."""
    if side != "auto":
        return side
    return "low" if tname == "msl" else "abs"


def setup_masks(bundle, regions, extreme_percentile, y, extreme_side="auto"):
    """Resolve region masks, area weights and (optional) per-target tail masks.

    Tail masks select the extreme cells of the OBSERVED target field within
    each region — fixed once, so every permutation is scored on the same cells.
    """
    target_indices = get_surface_target_indices(bundle)
    lat, lon = get_hres_lat_lon(bundle)
    lat_t = torch.as_tensor(lat)
    lon_t = torch.as_tensor(lon)
    region_masks = {r: build_region_mask(lat_t, lon_t, r) for r in regions}
    area_weights = get_area_weights(lat_t, bundle)
    LOGGER.info("Regions: %s | surface targets: %s",
                {r: int(m.sum()) for r, m in region_masks.items()}, list(target_indices))

    extreme_masks = None
    if extreme_percentile is not None:
        extreme_masks = {}
        for rname, rmask in region_masks.items():
            extreme_masks[rname] = {}
            for tname, tidx in target_indices.items():
                side = resolve_tail_side(extreme_side, tname)
                field = y[..., tidx].reshape(-1, y.shape[-2]).mean(dim=0)  # (G,)
                m = build_extreme_mask(field, extreme_percentile, region_mask=rmask,
                                       side=side)
                extreme_masks[rname][tname] = m
                LOGGER.info("tail mask %s/%s: p%g (%s tail) -> %d cells",
                            rname, tname, extreme_percentile, side, int(m.sum()))
    return target_indices, region_masks, area_weights, extreme_masks


def make_pair_scorer(target_indices, region_masks, area_weights, extreme_masks):
    """Build fn(perm_out, base_out) -> paired-MSE dict (global/region/tail)."""

    def score(perm_out, base_out):
        rec = {"paired_mse_aggregate": float(((perm_out - base_out) ** 2).mean().item()),
               "paired_mse_per_target": {}, "region_paired_mse_per_target": {},
               "extreme_paired_mse_per_target": {}}
        for tname, tidx in target_indices.items():
            d = (perm_out[..., tidx] - base_out[..., tidx]) ** 2  # (..., G)
            rec["paired_mse_per_target"][tname] = float(d.mean().item())
            grid = d.reshape(-1, d.shape[-1]).mean(dim=0)  # (G,)
            for rname, rmask in (region_masks or {}).items():
                rec["region_paired_mse_per_target"].setdefault(rname, {})[tname] = float(
                    masked_area_mean(grid, rmask, area_weights))
                if extreme_masks is not None:
                    rec["extreme_paired_mse_per_target"].setdefault(rname, {})[tname] = float(
                        masked_area_mean(grid, extreme_masks[rname][tname], area_weights))
        return rec

    return score


def _agg_repeats(recs: list[dict]) -> dict:
    """Mean (and per-target std) over repeat scorer dicts."""
    out = {
        "paired_mse_aggregate": float(np.mean([r["paired_mse_aggregate"] for r in recs])),
        "paired_mse_per_target": {
            t: float(np.mean([r["paired_mse_per_target"][t] for r in recs]))
            for t in recs[0]["paired_mse_per_target"]},
        "paired_mse_per_target_std": {
            t: float(np.std([r["paired_mse_per_target"][t] for r in recs]))
            for t in recs[0]["paired_mse_per_target"]},
    }
    for key in ("region_paired_mse_per_target", "extreme_paired_mse_per_target"):
        if recs[0].get(key):
            out[key] = {
                rname: {t: float(np.mean([r[key][rname][t] for r in recs]))
                        for t in recs[0][key][rname]}
                for rname in recs[0][key]}
    return out


# ---------------------------------------------------------------------------
# the permutation loop (shared by both modes)
# ---------------------------------------------------------------------------

def permute_all_variables(prepared, run_pair, scorer, n_repeats,
                          var_names_lres, var_names_hres, var_names_ntgt=None,
                          on_progress=None):
    """For each input variable, permute it across the batch and score against
    the paired baseline.

    Covers up to THREE conditioning pathways: lres (x_interp), hres (x_hres) and
    — when ``var_names_ntgt`` is given — the noised target itself (y_residual,
    the 'noisy_hres' pathway). run_pair(group, x_perm, rep) -> (perm_out,
    base_out): forward the permuted input and return it with the repeat-paired
    baseline output.
    """
    x_interp = prepared["x_interp"]
    x_hres = prepared["x_hres"]
    batch_size = x_interp.shape[0]
    if batch_size < 2:
        raise ValueError(
            f"Need batch_size >= 2 for permutation, got {batch_size}. "
            "Pass more (date, member, step) bundles, e.g. --event franklin_o96_o320_m4.")

    groups = [("lres", x_interp, var_names_lres), ("hres", x_hres, var_names_hres)]
    if var_names_ntgt is not None:
        groups.append(("noisy", prepared["y_residual"], var_names_ntgt))
    importance = {g[0]: {} for g in groups}
    for group, x_ref, names in groups:
        n_vars = x_ref.shape[-1]
        for var_idx in range(n_vars):
            recs = []
            for rep in range(n_repeats):
                x_perm = x_ref.clone()
                perm = torch.randperm(batch_size, device=x_ref.device)
                x_perm[..., var_idx] = x_ref[perm][..., var_idx]
                perm_out, base_out = run_pair(group, x_perm, rep)
                recs.append(scorer(perm_out, base_out))
            name = (names or {}).get(var_idx, f"{group}_{var_idx}")
            entry = {"name": name}
            entry.update(_agg_repeats(recs))
            importance[group][str(var_idx)] = entry
            LOGGER.info("  %s[%d]=%s  paired_agg=%.4e", group, var_idx, name,
                        entry["paired_mse_aggregate"])
            if on_progress is not None:
                on_progress(group, var_idx, importance)
    return importance


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------

def run_sigma_mode(bundle, prepared, args, target_indices, region_masks,
                   area_weights, extreme_masks, var_names):
    """Per-sigma single-denoiser-call importance (fixed noise, paired baseline)."""
    scorer = make_pair_scorer(target_indices, region_masks, area_weights, extreme_masks)
    x_interp, x_hres = prepared["x_interp"], prepared["x_hres"]
    y_residual = prepared["y_residual"]

    sigma_results = []
    for sigma in args.sigmas:
        LOGGER.info("Computing permutation importance at sigma=%.2f", sigma)
        noise = torch.randn_like(y_residual)
        baseline_out = denoise_at_sigma(bundle, x_interp, x_hres, y_residual, sigma,
                                        noise=noise)
        y_clean = y_residual[..., : baseline_out.shape[-1]]
        baseline_agg_mse = ((baseline_out - y_clean) ** 2).mean().item()
        baseline_region = {}
        for rname, rmask in region_masks.items():
            baseline_region[rname] = {}
            for tname, tidx in target_indices.items():
                per_cell = (baseline_out[..., tidx] - y_clean[..., tidx]) ** 2
                grid = per_cell.reshape(-1, per_cell.shape[-1]).mean(dim=0)
                baseline_region[rname][tname] = float(
                    masked_area_mean(grid, rmask, area_weights))

        def run_pair(group, x_perm, rep):
            if group == "lres":
                out = denoise_at_sigma(bundle, x_perm, x_hres, y_residual, sigma, noise=noise)
            elif group == "hres":
                out = denoise_at_sigma(bundle, x_interp, x_perm, y_residual, sigma, noise=noise)
            else:  # 'noisy': x_perm is the permuted y_residual (noised target)
                out = denoise_at_sigma(bundle, x_interp, x_hres, x_perm, sigma, noise=noise)
            return out, baseline_out

        ntgt_names = var_names.get("output") if getattr(args, "permute_noisy", False) else None
        importance = permute_all_variables(
            prepared, run_pair, scorer, args.n_repeats,
            var_names.get("input_lres"), var_names.get("input_hres"),
            var_names_ntgt=ntgt_names)

        # Legacy aggregate ratio (vs clean target) kept for back-compat plots.
        for group in importance:
            for entry in importance[group].values():
                entry["mean_mse_aggregate"] = entry["paired_mse_aggregate"]
                entry["importance_ratio_aggregate"] = float(
                    entry["paired_mse_aggregate"] / max(baseline_agg_mse, 1e-10))

        res = {
            "sigma": sigma,
            "surface_targets": list(target_indices.keys()),
            "regions": list(region_masks.keys()),
            "baseline_mse_aggregate": baseline_agg_mse,
            "baseline_mse_per_target": per_target_mse(baseline_out, y_clean, target_indices),
            "baseline_region_per_target_mse": baseline_region,
            "lres_importance": importance["lres"],
            "hres_importance": importance["hres"],
        }
        if "noisy" in importance:
            res["noisy_importance"] = importance["noisy"]
        sigma_results.append(res)
    return {"sigma_results": sigma_results}


def run_sampling_mode(bundle, prepared, args, target_indices, region_masks,
                      area_weights, extreme_masks, var_names, partial_path):
    """End-to-end importance via full Heun sampling (paired seeds).

    The repeat-paired baseline depends only on the seed, so it is computed once
    per repeat and shared across all variables (the old script recomputed it
    per variable — ~half the runtime for identical numbers).
    """
    scorer = make_pair_scorer(target_indices, region_masks, area_weights, extreme_masks)
    x_interp, x_hres = prepared["x_interp"], prepared["x_hres"]
    y_residual = prepared["y_residual"]

    base_by_rep: dict[int, torch.Tensor] = {}

    def base_for(rep):
        if rep not in base_by_rep:
            base_by_rep[rep] = sample_full(bundle, x_interp, x_hres,
                                           num_steps=args.num_steps, seed=args.seed + rep)
        return base_by_rep[rep]

    def run_pair(group, x_perm, rep):
        if group == "lres":
            out = sample_full(bundle, x_perm, x_hres, num_steps=args.num_steps,
                              seed=args.seed + rep)
        else:
            out = sample_full(bundle, x_interp, x_perm, num_steps=args.num_steps,
                              seed=args.seed + rep)
        return out, base_for(rep)

    baseline_out = base_for(0)
    y_target = y_residual[..., : baseline_out.shape[-1]].squeeze(1)
    baseline_target_mse = {
        name: float(((baseline_out[..., idx] - y_target[..., idx]) ** 2).mean().item())
        for name, idx in target_indices.items()}

    result = {
        "num_steps": args.num_steps,
        "n_repeats": args.n_repeats,
        "seed_base": args.seed,
        "surface_targets": list(target_indices.keys()),
        "regions": list(region_masks.keys()),
        "baseline_mse_per_target": baseline_target_mse,
        "lres_importance": {},
        "hres_importance": {},
    }

    def on_progress(group, var_idx, importance):
        # Periodic partial save: a late crash must not lose hours of sampling.
        result["lres_importance"] = importance["lres"]
        result["hres_importance"] = importance["hres"]
        if (var_idx + 1) % 10 == 0 or (group == "lres"
                                       and var_idx + 1 == prepared["x_interp"].shape[-1]):
            with open(partial_path, "w") as f:
                json.dump(result, f, indent=2)
            LOGGER.info("  [checkpoint] partial results -> %s", partial_path)

    importance = permute_all_variables(
        prepared, run_pair, scorer, args.n_repeats,
        var_names.get("input_lres"), var_names.get("input_hres"), on_progress)
    result["lres_importance"] = importance["lres"]
    result["hres_importance"] = importance["hres"]
    return {"result": result}


def _print_summary(payload: dict, mode: str):
    print("\n" + "=" * 70)
    print(f"FEATURE PERMUTATION ({mode}) — per-surface-target paired-MSE")
    print("=" * 70)
    blocks = (payload["sigma_results"] if mode == "sigma" else [payload["result"]])
    for res in blocks:
        head = f"Sigma = {res['sigma']:.2f}" if mode == "sigma" else (
            f"Full sampling (num_steps={res['num_steps']}, reps={res['n_repeats']})")
        print(f"\n{head}")
        targets = res["surface_targets"]
        for target in targets:
            for group, k in (("lres", 10), ("hres", 5)):
                ranked = sorted(res[f"{group}_importance"].items(),
                                key=lambda x: x[1]["paired_mse_per_target"][target],
                                reverse=True)
                if not ranked:
                    continue
                print(f"  Top-{k} {group} for target {target}:")
                for i, (_, info) in enumerate(ranked[:k]):
                    print(f"    {i+1:2d}. {info['name']:30s}  "
                          f"paired_mse={info['paired_mse_per_target'][target]:.4e}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Feature permutation importance for AIFSDD")
    add_model_args(p)
    add_event_args(p)
    add_sigma_args(p)
    p.add_argument("--mode", choices=["sigma", "sampling"], default="sigma",
                   help="sigma: one denoiser call per sigma. sampling: full Heun trajectory.")
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=20,
                   help="[sampling] Heun steps (canonical inference uses 40)")
    p.add_argument("--seed", type=int, default=12345, help="[sampling] base seed")
    p.add_argument("--region", action="append", default=None,
                   help="Region spec; repeatable. 'global', 'amazon', 'tropics', "
                        "or 'bbox:latmin,latmax,lonmin,lonmax'. Default: amazon_rainforest.")
    p.add_argument("--extreme-percentile", type=float, default=None,
                   help="Also score over tail cells of y_target (per region) "
                        "-> extreme_paired_mse_per_target.")
    p.add_argument("--extreme-side", default="auto",
                   choices=["auto", "abs", "high", "low"],
                   help="Which tail; auto = low for msl (cyclones), abs otherwise.")
    p.add_argument("--permute-noisy", action="store_true",
                   help="Also permute the noised-target channels (the 'noisy_hres' "
                        "pathway) -> noisy_importance (sigma mode only).")
    args = p.parse_args(argv)
    setup_logging()

    bundle_dir, dates, members, steps, _ = resolve_event_args(args)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s", args.checkpoint)
    bundle = load_model(args.checkpoint, device=args.device, precision=args.precision)
    var_names = get_variable_names(bundle)

    batch = collect_event_bundles(bundle, bundle_dir, dates, members, steps)
    prepared = prepare_batch(bundle, batch.x_lres, batch.x_hres, batch.y)

    regions = args.region or ["amazon_rainforest"]
    target_indices, region_masks, area_weights, extreme_masks = setup_masks(
        bundle, regions, args.extreme_percentile, batch.y.to(bundle.device),
        extreme_side=args.extreme_side)

    common = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "n_batches": int(batch.x_lres.shape[0]),
        "batch_size": int(batch.x_lres.shape[0]),
        "regions": regions,
        "extreme_percentile": args.extreme_percentile,
        "bundle_paths": batch.paths,
    }

    if args.mode == "sigma":
        payload = run_sigma_mode(bundle, prepared, args, target_indices, region_masks,
                                 area_weights, extreme_masks, var_names)
        results_file = output_path / "permutation_importance.json"
    else:
        partial = output_path / "permutation_importance_full_sampling.partial.json"
        payload = run_sampling_mode(bundle, prepared, args, target_indices, region_masks,
                                    area_weights, extreme_masks, var_names, partial)
        results_file = output_path / "permutation_importance_full_sampling.json"

    all_results = {**common, **payload}
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info("Results saved to %s", results_file)
    write_run_meta(output_path, f"permutation:{args.mode}", args)
    _print_summary(all_results, args.mode)
    return all_results


if __name__ == "__main__":
    main()
