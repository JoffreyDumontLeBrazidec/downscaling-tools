"""Sigma-sweep conditioning ablation for AIFSDD.

At each sigma level, compare the model's output with:
  1. Full conditioning (normal)
  2. Zeroed lres conditioning (coarse atmospheric state)
  3. Zeroed hres forcing (static/time forcings)
  4. Zeroed noisy target (no target signal of any kind)

Measures how much the model relies on each conditioning pathway at each
denoising stage — per surface target, per region, optionally over tail cells.

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp ablation \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir /path/to/results/ \
        --event franklin_o96_o320_m4
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch

_DT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.cli import add_event_args, add_model_args, add_sigma_args, setup_logging
from interp.core.data import collect_event_bundles, resolve_event_args
from interp.core.model import denoise_at_sigma, load_model, prepare_batch
from interp.core.regions import masked_area_mean
from interp.core.runmeta import write_run_meta
from interp.tools.permutation import setup_masks

LOGGER = logging.getLogger(__name__)


def compute_conditioning_ablation(bundle, prepared, sigma, region_masks,
                                  target_indices, area_weights, extreme_masks=None):
    """Measure conditioning reliance at a given sigma."""
    x_interp = prepared["x_interp"]
    x_hres = prepared["x_hres"]
    y_residual = prepared["y_residual"]

    noise = torch.randn_like(y_residual)

    # 1. Full conditioning baseline (lres + hres + noisy target)
    out_full = denoise_at_sigma(bundle, x_interp, x_hres, y_residual, sigma, noise=noise)
    # 2. Zero in_lres (coarse atmospheric state)
    out_no_lres = denoise_at_sigma(bundle, torch.zeros_like(x_interp), x_hres,
                                   y_residual, sigma, noise=noise)
    # 3. Zero in_hres (high-res static/time forcings)
    out_no_hres = denoise_at_sigma(bundle, x_interp, torch.zeros_like(x_hres),
                                   y_residual, sigma, noise=noise)
    # 4. Zero noisy_hres: y_noised = y_residual + sigma*noise; zero both so the
    #    denoiser receives y_noised = 0 (no target signal of any kind).
    out_no_ytgt = denoise_at_sigma(bundle, x_interp, x_hres,
                                   torch.zeros_like(y_residual), sigma,
                                   noise=torch.zeros_like(noise))

    def _metrics(out_ablated):
        diff = (out_ablated - out_full).float()
        mse = (diff ** 2).mean().item()
        f = out_full.float().flatten()
        a = out_ablated.float().flatten()
        corr = torch.corrcoef(torch.stack([f, a]))[0, 1].item()
        # Per-output-channel MSE — full grid (kept for back-compat)
        var_mse = (diff ** 2).mean(dim=(0, 1, 2, 3)).cpu().numpy().tolist()
        region_pt, extreme_pt = {}, {}
        sq = diff ** 2  # (B, T, E, G, V)
        for tgt, idx in target_indices.items():
            per_cell = sq[..., idx]
            grid = per_cell.reshape(-1, per_cell.shape[-1]).mean(dim=0)  # (G,)
            for rname, rmask in region_masks.items():
                region_pt.setdefault(rname, {})[tgt] = float(
                    masked_area_mean(grid, rmask, area_weights))
                if extreme_masks is not None:
                    extreme_pt.setdefault(rname, {})[tgt] = float(
                        masked_area_mean(grid, extreme_masks[rname][tgt], area_weights))
        out = {"mse_vs_full": mse, "correlation_with_full": corr,
               "per_var_mse": var_mse, "region_per_target_mse": region_pt}
        if extreme_masks is not None:
            out["extreme_per_target_mse"] = extreme_pt
        return out

    baseline_mse = ((out_full.float()
                     - y_residual[..., :out_full.shape[-1]].float()) ** 2).mean().item()

    return {
        "sigma": sigma,
        "baseline_denoising_mse": baseline_mse,
        "ablate_lres": _metrics(out_no_lres),
        "ablate_hres": _metrics(out_no_hres),
        "ablate_noisy_hres": _metrics(out_no_ytgt),
    }


def _print_summary(all_results: dict):
    print("\n" + "=" * 70)
    print("CONDITIONING ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Sigma':>8s}  {'Baseline MSE':>14s}  {'No LRES corr':>14s}  "
          f"{'No HRES corr':>14s}  {'No noisy corr':>14s}")
    print("-" * 70)
    for res in all_results["sigma_results"]:
        print(f"{res['sigma']:8.2f}  {res['baseline_denoising_mse']:14.6f}  "
              f"{res['ablate_lres']['correlation_with_full']:14.4f}  "
              f"{res['ablate_hres']['correlation_with_full']:14.4f}  "
              f"{res['ablate_noisy_hres']['correlation_with_full']:14.4f}")
    print("\nInterpretation:")
    print("  corr ~ 1.0 -> ablation has little effect (model doesn't use this input)")
    print("  corr ~ 0.0 -> ablation destroys output (model heavily relies on this input)")
    print("  High sigma: expect high lres reliance (coarse structure from conditioning)")
    print("  Low sigma:  expect low lres reliance (fine detail generated freely)")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Conditioning ablation for AIFSDD")
    add_model_args(p)
    add_event_args(p)
    add_sigma_args(p, default=[0.1, 0.5, 1.0, 5.0, 20.0, 80.0])
    p.add_argument("--region", action="append", default=None,
                   help="Region spec; repeatable. Default: amazon_rainforest.")
    p.add_argument("--extreme-percentile", type=float, default=None,
                   help="Also score over tail cells of each target field.")
    args = p.parse_args(argv)
    setup_logging()

    bundle_dir, dates, members, steps, _ = resolve_event_args(args)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s", args.checkpoint)
    bundle = load_model(args.checkpoint, device=args.device, precision=args.precision)

    batch = collect_event_bundles(bundle, bundle_dir, dates, members, steps)
    prepared = prepare_batch(bundle, batch.x_lres, batch.x_hres, batch.y)

    regions = args.region or ["amazon_rainforest"]
    target_indices, region_masks, area_weights, extreme_masks = setup_masks(
        bundle, regions, args.extreme_percentile, batch.y.to(bundle.device))

    all_results = {
        "checkpoint": args.checkpoint,
        "n_batches": int(batch.x_lres.shape[0]),
        "batch_size": int(batch.x_lres.shape[0]),
        "regions": regions,
        "extreme_percentile": args.extreme_percentile,
        "surface_targets": list(target_indices),
        "sigma_results": [],
    }
    for sigma in args.sigmas:
        LOGGER.info("Conditioning ablation at sigma=%.2f", sigma)
        all_results["sigma_results"].append(compute_conditioning_ablation(
            bundle, prepared, sigma, region_masks, target_indices, area_weights,
            extreme_masks))

    results_file = output_path / "conditioning_ablation.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info("Results saved to %s", results_file)
    write_run_meta(output_path, "ablation", args)
    _print_summary(all_results)
    return all_results


if __name__ == "__main__":
    main()
