"""Activation magnitude profiling across sigma levels.

Hooks into encoder, processor layers, and decoder to record activation
norms at each sigma level. Reveals which layers are most active at which
denoising stage.

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp.activation_profiling \
        --checkpoint /path/to/checkpoint.ckpt \
        --output-dir /path/to/results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_DT_ROOT = Path(__file__).resolve().parent.parent
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from interp.model_utils import load_model, prepare_batch, denoise_at_sigma, collect_event_bundles

LOGGER = logging.getLogger(__name__)


class ActivationCollector:
    """Register forward hooks to capture activation statistics."""

    def __init__(self):
        self.stats = defaultdict(dict)
        self._hooks = []

    def register(self, model):
        """Register hooks on encoder, processor layers, and decoder."""
        inner = model

        # Encoder
        if hasattr(inner, "encoder"):
            h = inner.encoder.register_forward_hook(self._make_hook("encoder"))
            self._hooks.append(h)

        # Processor layers
        if hasattr(inner, "processor"):
            proc = inner.processor
            if hasattr(proc, "blocks"):
                for i, block in enumerate(proc.blocks):
                    h = block.register_forward_hook(self._make_hook(f"processor.block_{i:02d}"))
                    self._hooks.append(h)
            elif hasattr(proc, "layers"):
                for i, layer in enumerate(proc.layers):
                    h = layer.register_forward_hook(self._make_hook(f"processor.layer_{i:02d}"))
                    self._hooks.append(h)
            elif hasattr(proc, "chunk"):
                for i, chunk in enumerate(proc.chunk):
                    h = chunk.register_forward_hook(self._make_hook(f"processor.chunk_{i:02d}"))
                    self._hooks.append(h)
            else:
                h = proc.register_forward_hook(self._make_hook("processor"))
                self._hooks.append(h)

        # Decoder
        if hasattr(inner, "decoder"):
            h = inner.decoder.register_forward_hook(self._make_hook("decoder"))
            self._hooks.append(h)

        LOGGER.info("Registered %d activation hooks", len(self._hooks))

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                tensors = [t for t in output if isinstance(t, torch.Tensor)]
            elif isinstance(output, torch.Tensor):
                tensors = [output]
            else:
                return

            for t in tensors:
                ft = t.float()
                self.stats[self._current_sigma][name] = {
                    "l2_norm": ft.norm().item(),
                    "mean": ft.mean().item(),
                    "std": ft.std().item(),
                    "max_abs": ft.abs().max().item(),
                    "shape": list(t.shape),
                }
                break  # Only first tensor

        return hook_fn

    def set_sigma(self, sigma: float):
        self._current_sigma = sigma

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def run_activation_profiling(
    checkpoint_path: str,
    output_dir: str,
    sigmas: list[float] | None = None,
    n_batches: int = 4,
    device: str = "cuda",
    precision: str = "bf16",
    bundle_dir: str = None,
    dates: list[str] | None = None,
    members: list[str] | None = None,
    steps: list[str] | None = None,
):
    """Profile activation magnitudes across sigma levels."""
    if sigmas is None:
        sigmas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s", checkpoint_path)
    bundle = load_model(checkpoint_path, device=device, precision=precision)

    LOGGER.info("Loading event bundles...")
    x_lres_all, x_hres_all, y_all = collect_event_bundles(
        bundle, bundle_dir, dates, members, steps)

    prepared = prepare_batch(bundle, x_lres_all, x_hres_all, y_all)

    # Register hooks
    collector = ActivationCollector()
    collector.register(bundle.inner_model)

    # Fixed noise
    noise = torch.randn_like(prepared["y_residual"])

    for sigma in sigmas:
        LOGGER.info("Profiling activations at sigma=%.3f", sigma)
        collector.set_sigma(sigma)
        _ = denoise_at_sigma(
            bundle,
            prepared["x_interp"],
            prepared["x_hres"],
            prepared["y_residual"],
            sigma,
            noise=noise,
        )

    collector.remove_hooks()

    # Convert to serializable format
    results = {
        "checkpoint": checkpoint_path,
        "sigmas": sigmas,
        "profiles": {},
    }
    for sigma, layer_stats in sorted(collector.stats.items()):
        results["profiles"][str(sigma)] = layer_stats

    results_file = output_path / "activation_profiling.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Results saved to %s", results_file)

    _print_summary(results)
    return results


def _print_summary(results: dict):
    print("\n" + "=" * 90)
    print("ACTIVATION MAGNITUDE PROFILING")
    print("=" * 90)

    # Collect all layer names
    all_layers = set()
    for sigma_str, layer_stats in results["profiles"].items():
        all_layers.update(layer_stats.keys())
    all_layers = sorted(all_layers)

    # Header
    sigmas = results["sigmas"]
    header = f"{'Layer':<28s}"
    for s in sigmas:
        header += f"  σ={s:<6.1f}"
    print(header)
    print("-" * 90)

    # L2 norms per layer per sigma
    for layer in all_layers:
        row = f"{layer:<28s}"
        for s in sigmas:
            stats = results["profiles"].get(str(s), {}).get(layer, {})
            norm = stats.get("l2_norm", float("nan"))
            row += f"  {norm:8.1f}"
        print(row)

    print("\nNote: Dramatic norm changes across sigma indicate sigma-specialized layers")


def main():
    parser = argparse.ArgumentParser(description="Activation profiling for AIFSDD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sigmas", nargs="+", type=float,
                        default=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--bundle-dir", required=True,
                        help="Dir of real event .nc bundles (replaces val_dataloader).")
    parser.add_argument("--dates", nargs="+", default=["20230826"])
    parser.add_argument("--members", nargs="+", default=["01"])
    parser.add_argument("--steps", nargs="+", default=["048"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_activation_profiling(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        sigmas=args.sigmas,
        n_batches=args.n_batches,
        device=args.device,
        precision=args.precision,
        bundle_dir=args.bundle_dir,
        dates=args.dates,
        members=args.members,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
