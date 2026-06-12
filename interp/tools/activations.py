"""Activation norms + CKA layer similarity across sigma levels (one pass).

Merges the former activation_profiling.py and cka_analysis.py: both needed the
same per-sigma denoiser forward with hooks on the 16 processor blocks, so they
now share it. Per sigma:

  - norms : l2/mean/std/max_abs of encoder, decoder, whole-processor outputs
            AND every individual processor block (richer than the old tool,
            which could only see the whole processor on this architecture).
  - cka   : pairwise linear CKA between all processor block activations
            (layer clustering, sigma-dependent representation transitions).

Writes BOTH legacy JSON files (activation_profiling.json + cka_analysis.json)
so existing renderers and old result sets stay comparable.

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp activations \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir /path/to/results/ \
        --event franklin_o96_o320_m4
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
from interp.core.hooks import (
    captured_block_activations,
    captured_module_stats,
    tensor_stats,
)
from interp.core.model import denoise_at_sigma, load_model, prepare_batch
from interp.core.runmeta import write_run_meta

LOGGER = logging.getLogger(__name__)


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two activation matrices (n_samples, n_features).

    Stays in fp32 on whatever device the inputs live. Centering + Frobenius
    normalization keep the HSIC norms in [0, 1] so fp32 squaring is safe; the
    Gram products are (d, d) with d=hidden_dim, so they're tiny.
    """
    X = X.float()
    Y = Y.float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    nX, nY = X.norm(), Y.norm()
    if nX < 1e-12 or nY < 1e-12:  # numerically-dead activations -> define CKA=0
        return 0.0
    X = X / nX
    Y = Y / nY
    hsic_xy = ((X.T @ Y) ** 2).sum()
    hsic_xx = ((X.T @ X) ** 2).sum()
    hsic_yy = ((Y.T @ Y) ** 2).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float((hsic_xy / denom).item())


def cka_matrix(block_acts: dict[str, torch.Tensor]):
    """Pairwise CKA over the captured block activations."""
    layer_names = sorted(block_acts.keys())
    flat = {n: block_acts[n].reshape(block_acts[n].shape[0], -1) for n in layer_names}
    n = len(layer_names)
    m = np.zeros((n, n))
    for i, ni in enumerate(layer_names):
        for j, nj in enumerate(layer_names):
            if i <= j:
                v = linear_CKA(flat[ni], flat[nj])
                m[i, j] = m[j, i] = v
    return layer_names, m


def _print_summary(profiling: dict, cka: dict):
    print("\n" + "=" * 90)
    print("ACTIVATION NORMS (whole modules)")
    print("=" * 90)
    sigmas = profiling["sigmas"]
    modules = ["encoder", "processor", "decoder"]
    header = f"{'Module':<28s}" + "".join(f"  σ={s:<6.1f}" for s in sigmas)
    print(header)
    for mod in modules:
        row = f"{mod:<28s}"
        for s in sigmas:
            norm = profiling["profiles"].get(str(s), {}).get(mod, {}).get("l2_norm",
                                                                          float("nan"))
            row += f"  {norm:8.1f}"
        print(row)
    print("\nCKA: mean off-diagonal per sigma")
    for sigma_str, data in cka["cka_matrices"].items():
        m = np.array(data["matrix"])
        mask = ~np.eye(m.shape[0], dtype=bool)
        print(f"  sigma={float(sigma_str):8.2f}: {m[mask].mean():.4f}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Activation norms + CKA for AIFSDD")
    add_model_args(p)
    add_event_args(p)
    add_sigma_args(p)
    args = p.parse_args(argv)
    setup_logging()

    bundle_dir, dates, members, steps, _ = resolve_event_args(args)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s", args.checkpoint)
    bundle = load_model(args.checkpoint, device=args.device, precision=args.precision)
    inner = bundle.inner_model

    batch = collect_event_bundles(bundle, bundle_dir, dates, members, steps)
    prepared = prepare_batch(bundle, batch.x_lres, batch.x_hres, batch.y)
    noise = torch.randn_like(prepared["y_residual"])

    profiles = {}
    cka_matrices = {}
    for sigma in args.sigmas:
        LOGGER.info("Activations at sigma=%.3f", sigma)
        with captured_block_activations(inner) as blocks, \
                captured_module_stats(inner) as modstats:
            _ = denoise_at_sigma(bundle, prepared["x_interp"], prepared["x_hres"],
                                 prepared["y_residual"], sigma, noise=noise)
            layer_stats = dict(modstats)
            for name, act in blocks.items():
                layer_stats[f"processor.{name}"] = tensor_stats(act)
            if blocks:
                names, m = cka_matrix(blocks)
                cka_matrices[str(sigma)] = {"layer_names": names, "matrix": m.tolist()}
            else:
                LOGGER.warning("No block activations captured at sigma=%.2f", sigma)
        profiles[str(sigma)] = layer_stats

    profiling = {"checkpoint": args.checkpoint, "sigmas": args.sigmas,
                 "profiles": profiles}
    cka = {"checkpoint": args.checkpoint, "sigmas": args.sigmas,
           "cka_matrices": cka_matrices}

    # Two files (legacy names) so old runs and old renderers stay comparable.
    with open(output_path / "activation_profiling.json", "w") as f:
        json.dump(profiling, f, indent=2)
    with open(output_path / "cka_analysis.json", "w") as f:
        json.dump(cka, f, indent=2)
    LOGGER.info("Results saved to %s/{activation_profiling,cka_analysis}.json", output_path)
    write_run_meta(output_path, "activations", args)
    _print_summary(profiling, cka)
    return profiling, cka


if __name__ == "__main__":
    main()
