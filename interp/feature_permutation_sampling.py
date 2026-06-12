"""Feature permutation importance with FULL Heun sampling trajectory.

Companion to interp/feature_permutation.py which measures per-sigma importance
of a single denoising step. This script measures end-to-end importance: for
each input variable, permute it across batch, run the full diffusion sampling
trajectory (sigma_max -> sigma_min via Heun), and compare the final sample to
the un-permuted baseline.

The two scripts answer different questions:
- feature_permutation.py: which inputs matter for the denoiser at each noise regime?
- feature_permutation_sampling.py: which inputs matter for the final output?

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp.feature_permutation_sampling \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir /path/to/results/ \
        --n-repeats 3 \
        --num-steps 20
"""

from __future__ import annotations

import argparse
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

LOGGER = logging.getLogger(__name__)


def sample_full(bundle, x_interp, x_hres, num_steps: int, seed: int) -> torch.Tensor:
    """Run the full Heun sampling trajectory on (x_interp, x_hres).

    Uses the model's own sample() entry point with a fixed seed so baseline
    and permuted runs share the same y_init noise.
    """
    inner = bundle.inner_model
    with torch.no_grad():
        out = inner.sample(
            x_interp,
            x_hres,
            model_comm_group=None,
            grid_shard_shapes=None,
            noise_scheduler_params={"num_steps": num_steps},
            sampler_params=None,
            seed=seed,
        )
    return out


def compute_sampling_importance(
    bundle,
    prepared: dict,
    num_steps: int = 20,
    n_repeats: int = 3,
    seed: int = 12345,
    var_names_lres: dict | None = None,
    var_names_hres: dict | None = None,
    _checkpoint_path: str | None = None,
) -> dict:
    """End-to-end permutation importance via full Heun sampling, scored per surface target."""
    x_interp = prepared["x_interp"]
    x_hres = prepared["x_hres"]
    y_residual = prepared["y_residual"]

    batch_size = x_interp.shape[0]
    if batch_size < 2:
        raise ValueError(
            f"Need batch_size >= 2 for permutation, got {batch_size}. "
            "Increase --n-batches or use a larger validation set."
        )

    target_indices = get_surface_target_indices(bundle)
    LOGGER.info("Surface targets being tracked: %s", target_indices)

    # Baseline once (un-paired, just for reporting)
    LOGGER.info("Computing baseline sample (num_steps=%d, seed=%d)...", num_steps, seed)
    baseline_out = sample_full(bundle, x_interp, x_hres, num_steps=num_steps, seed=seed)
    y_target = y_residual[..., : baseline_out.shape[-1]].squeeze(1)

    baseline_target_mse = {
        name: float(((baseline_out[..., idx] - y_target[..., idx]) ** 2).mean().item())
        for name, idx in target_indices.items()
    }
    LOGGER.info("Baseline per-target MSE: %s", {k: f"{v:.4e}" for k, v in baseline_target_mse.items()})

    results = {
        "num_steps": num_steps,
        "n_repeats": n_repeats,
        "seed_base": seed,
        "surface_targets": list(target_indices.keys()),
        "baseline_mse_per_target": baseline_target_mse,
        "lres_importance": {},
        "hres_importance": {},
    }

    n_lres_vars = x_interp.shape[-1]
    n_hres_vars = x_hres.shape[-1]
    LOGGER.info("Permuting %d lres + %d hres variables, %d repeats each", n_lres_vars, n_hres_vars, n_repeats)

    def _paired_per_target(perm_out, base_paired):
        return {
            name: float(((perm_out[..., idx] - base_paired[..., idx]) ** 2).mean().item())
            for name, idx in target_indices.items()
        }

    def _aggregate_paired(perm_out, base_paired):
        return float(((perm_out - base_paired) ** 2).mean().item())

    # Permute each lres variable
    for var_idx in range(n_lres_vars):
        per_target = {name: [] for name in target_indices}
        agg = []
        for rep in range(n_repeats):
            x_perm = x_interp.clone()
            perm = torch.randperm(batch_size, device=x_interp.device)
            x_perm[..., var_idx] = x_interp[perm][..., var_idx]
            perm_out = sample_full(bundle, x_perm, x_hres, num_steps=num_steps, seed=seed + rep)
            base_paired = sample_full(bundle, x_interp, x_hres, num_steps=num_steps, seed=seed + rep)
            per_tgt = _paired_per_target(perm_out, base_paired)
            for k, v in per_tgt.items():
                per_target[k].append(v)
            agg.append(_aggregate_paired(perm_out, base_paired))
        var_name = var_names_lres.get(var_idx, f"lres_{var_idx}") if var_names_lres else f"lres_{var_idx}"
        results["lres_importance"][str(var_idx)] = {
            "name": var_name,
            "paired_mse_aggregate": float(np.mean(agg)),
            "paired_mse_per_target": {name: float(np.mean(vs)) for name, vs in per_target.items()},
            "paired_mse_per_target_std": {name: float(np.std(vs)) for name, vs in per_target.items()},
        }
        LOGGER.info(
            "  lres[%d]=%s  agg_paired=%.4e  per_target=%s",
            var_idx, var_name,
            results["lres_importance"][str(var_idx)]["paired_mse_aggregate"],
            {k: f"{v:.2e}" for k, v in results["lres_importance"][str(var_idx)]["paired_mse_per_target"].items()},
        )
        # Intermediate save every 10 vars so a late crash doesn\'t lose hours of work
        if (var_idx + 1) % 10 == 0 and _checkpoint_path is not None:
            with open(_checkpoint_path, "w") as _f:
                import json as _json
                _json.dump(results, _f, indent=2)
            LOGGER.info("  [checkpoint] partial results saved to %s after lres[%d]", _checkpoint_path, var_idx)

    # Save again after lres phase (we just spent ~90 min on it)
    if _checkpoint_path is not None:
        with open(_checkpoint_path, "w") as _f:
            import json as _json
            _json.dump(results, _f, indent=2)
        LOGGER.info("  [checkpoint] post-lres-phase save to %s", _checkpoint_path)

    # Permute each hres variable
    for var_idx in range(n_hres_vars):
        per_target = {name: [] for name in target_indices}
        agg = []
        for rep in range(n_repeats):
            x_perm = x_hres.clone()
            perm = torch.randperm(batch_size, device=x_hres.device)
            x_perm[..., var_idx] = x_hres[perm][..., var_idx]
            perm_out = sample_full(bundle, x_interp, x_perm, num_steps=num_steps, seed=seed + rep)
            base_paired = sample_full(bundle, x_interp, x_hres, num_steps=num_steps, seed=seed + rep)
            per_tgt = _paired_per_target(perm_out, base_paired)
            for k, v in per_tgt.items():
                per_target[k].append(v)
            agg.append(_aggregate_paired(perm_out, base_paired))
        var_name = var_names_hres.get(var_idx, f"hres_{var_idx}") if var_names_hres else f"hres_{var_idx}"
        results["hres_importance"][str(var_idx)] = {
            "name": var_name,
            "paired_mse_aggregate": float(np.mean(agg)),
            "paired_mse_per_target": {name: float(np.mean(vs)) for name, vs in per_target.items()},
            "paired_mse_per_target_std": {name: float(np.std(vs)) for name, vs in per_target.items()},
        }
        LOGGER.info(
            "  hres[%d]=%s  agg_paired=%.4e  per_target=%s",
            var_idx, var_name,
            results["hres_importance"][str(var_idx)]["paired_mse_aggregate"],
            {k: f"{v:.2e}" for k, v in results["hres_importance"][str(var_idx)]["paired_mse_per_target"].items()},
        )

    return results


def run_full_sampling_study(
    checkpoint_path: str,
    output_dir: str,
    num_steps: int = 20,
    n_repeats: int = 3,
    n_batches: int = 2,
    device: str = "cuda",
    precision: str = "fp32",
    seed: int = 12345,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s", checkpoint_path)
    bundle = load_model(checkpoint_path, device=device, precision=precision)
    var_names = get_variable_names(bundle)

    LOGGER.info("Getting validation data...")
    bundle.datamodule.setup(stage="predict")
    dl = bundle.datamodule.val_dataloader()
    batches = []
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        batches.append(batch)
    LOGGER.info("Collected %d batches", len(batches))

    x_lres_all = torch.cat([b[0] for b in batches], dim=0)
    x_hres_all = torch.cat([b[1] for b in batches], dim=0)
    y_all = torch.cat([b[2] for b in batches], dim=0)
    LOGGER.info("Stacked batch shapes: lres=%s, hres=%s, y=%s",
                x_lres_all.shape, x_hres_all.shape, y_all.shape)

    prepared = prepare_batch(bundle, x_lres_all, x_hres_all, y_all)

    result = compute_sampling_importance(
        bundle,
        prepared,
        num_steps=num_steps,
        n_repeats=n_repeats,
        seed=seed,
        var_names_lres=var_names.get("input_lres"),
        var_names_hres=var_names.get("input_hres"),
        _checkpoint_path=str(output_path / "permutation_importance_full_sampling.partial.json"),
    )

    all_results = {
        "checkpoint": checkpoint_path,
        "n_batches": len(batches),
        "batch_size": x_lres_all.shape[0],
        "result": result,
    }

    results_file = output_path / "permutation_importance_full_sampling.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOGGER.info("Results saved to %s", results_file)

    _print_summary(result)
    return all_results


def _print_summary(res: dict):
    print("\n" + "=" * 70)
    print("FULL-SAMPLING FEATURE PERMUTATION — per-surface-target paired MSE")
    print(f"(num_steps={res['num_steps']}, n_repeats={res['n_repeats']})")
    print("=" * 70)
    targets = res["surface_targets"]
    print("\nBaseline per-target MSE (sampler output vs clean target):")
    for t in targets:
        print(f"    {t:>5}: {res['baseline_mse_per_target'][t]:.4e}")
    for target in targets:
        print(f"\n=== Target: {target} ===")
        ranked_l = sorted(
            res["lres_importance"].items(),
            key=lambda x: x[1]["paired_mse_per_target"][target],
            reverse=True,
        )
        print("  Top 15 lres inputs (by paired MSE for this target):")
        for i, (var_idx, info) in enumerate(ranked_l[:15]):
            print(f"    {i+1:2d}. {info['name']:30s}  paired_mse={info['paired_mse_per_target'][target]:.4e}")
        ranked_h = sorted(
            res["hres_importance"].items(),
            key=lambda x: x[1]["paired_mse_per_target"][target],
            reverse=True,
        )
        if ranked_h:
            print("  Top 5 hres inputs (by paired MSE for this target):")
            for i, (var_idx, info) in enumerate(ranked_h[:5]):
                print(f"    {i+1:2d}. {info['name']:30s}  paired_mse={info['paired_mse_per_target'][target]:.4e}")


def main():
    parser = argparse.ArgumentParser(
        description="Full-sampling feature permutation importance for AIFSDD",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Heun sampling steps (default 20; canonical inference uses 40)")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--n-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_full_sampling_study(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        n_repeats=args.n_repeats,
        n_batches=args.n_batches,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
