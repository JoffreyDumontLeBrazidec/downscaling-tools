"""CKA (Centered Kernel Alignment) layer similarity analysis.

Computes pairwise CKA between all processor layer activations at different
sigma levels. Reveals layer clustering and sigma-dependent phase transitions
in representation structure.

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp.cka_analysis \
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


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two activation matrices.

    Stays in fp32 on whatever device the inputs live (typically GPU).
    Centering + Frobenius normalization keep the HSIC norms in [0, 1] so
    fp32 squaring is safe. The Gram products X^T X / X^T Y are (d, d) where
    d=hidden_dim=512, so they're tiny — the bulk of work is on the small
    matrices, not the (n_samples, d) inputs.

    Parameters
    ----------
    X, Y : torch.Tensor of shape (n_samples, n_features), any device, fp32 ok.

    Returns
    -------
    float : CKA similarity in [0, 1]
    """
    X = X.float()
    Y = Y.float()

    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Frobenius-normalize: makes ||X^T X||_F^2 <= ||X||_F^4 = 1, overflow-safe.
    nX = X.norm()
    nY = Y.norm()
    if nX < 1e-12 or nY < 1e-12:
        return 0.0
    X = X / nX
    Y = Y / nY

    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = (XtY ** 2).sum()
    hsic_xx = (XtX ** 2).sum()
    hsic_yy = (YtY ** 2).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float((hsic_xy / denom).item())


class LayerActivationCollector:
    """Collect activations from processor layers for CKA analysis."""

    def __init__(self):
        self.activations = {}
        self._hooks = []

    def register(self, model):
        """Register hooks on every individual processor block.

        Anemoi's processor splits num_layers into num_chunks ProcessorChunks
        (stored in `proc.proc`); each chunk holds the actual transformer
        layers in `chunk.blocks`. Drill through both to get one hook per
        layer (e.g. 2 chunks * 8 blocks = 16 layers).
        """
        proc = model.processor
        items = []
        if hasattr(proc, "proc") and len(proc.proc) > 0:
            global_idx = 0
            for chunk in proc.proc:
                blocks = getattr(chunk, "blocks", None)
                if blocks is None:
                    LOGGER.warning("Chunk %s has no .blocks attribute", type(chunk).__name__)
                    continue
                for block in blocks:
                    items.append((global_idx, block))
                    global_idx += 1
        elif hasattr(proc, "blocks"):
            items = list(enumerate(proc.blocks))
        elif hasattr(proc, "layers"):
            items = list(enumerate(proc.layers))
        else:
            LOGGER.warning("Cannot find processor sub-layers for CKA (type=%s, attrs=%s)",
                           type(proc).__name__, [a for a in dir(proc) if not a.startswith('_')][:20])
            return

        for i, layer in items:
            name = f"layer_{i:02d}"
            h = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

        LOGGER.info("Registered %d processor layer hooks for CKA", len(self._hooks))

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                t = output[0]
            else:
                t = output
            if isinstance(t, torch.Tensor):
                # Keep on GPU + fp32; CKA computation lives on-device for speed.
                self.activations[name] = t.detach().float()
        return hook_fn

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def run_cka_analysis(
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
    """Compute pairwise CKA between processor layers at each sigma."""
    if sigmas is None:
        sigmas = [0.1, 1.0, 5.0, 20.0, 80.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s", checkpoint_path)
    bundle = load_model(checkpoint_path, device=device, precision=precision)

    LOGGER.info("Loading event bundles...")
    x_lres_all, x_hres_all, y_all = collect_event_bundles(
        bundle, bundle_dir, dates, members, steps)

    prepared = prepare_batch(bundle, x_lres_all, x_hres_all, y_all)

    collector = LayerActivationCollector()
    collector.register(bundle.inner_model)

    noise = torch.randn_like(prepared["y_residual"])

    results = {
        "checkpoint": checkpoint_path,
        "sigmas": sigmas,
        "cka_matrices": {},
    }

    for sigma in sigmas:
        LOGGER.info("Computing CKA at sigma=%.2f", sigma)
        collector.clear()

        _ = denoise_at_sigma(
            bundle,
            prepared["x_interp"],
            prepared["x_hres"],
            prepared["y_residual"],
            sigma,
            noise=noise,
        )

        layer_names = sorted(collector.activations.keys())
        n_layers = len(layer_names)

        if n_layers == 0:
            LOGGER.warning("No activations collected at sigma=%.2f", sigma)
            continue

        # Flatten activations to (n_samples, n_features) for CKA
        flat_acts = {}
        for name in layer_names:
            act = collector.activations[name]
            flat_acts[name] = act.reshape(act.shape[0], -1)

        # Pairwise CKA matrix
        cka_matrix = np.zeros((n_layers, n_layers))
        for i, name_i in enumerate(layer_names):
            for j, name_j in enumerate(layer_names):
                if i <= j:
                    cka_val = linear_CKA(flat_acts[name_i], flat_acts[name_j])
                    cka_matrix[i, j] = cka_val
                    cka_matrix[j, i] = cka_val

        results["cka_matrices"][str(sigma)] = {
            "layer_names": layer_names,
            "matrix": cka_matrix.tolist(),
        }

    collector.remove_hooks()

    results_file = output_path / "cka_analysis.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Results saved to %s", results_file)

    _print_summary(results)
    return results


def _print_summary(results: dict):
    print("\n" + "=" * 70)
    print("CKA LAYER SIMILARITY ANALYSIS")
    print("=" * 70)

    for sigma_str, data in results["cka_matrices"].items():
        sigma = float(sigma_str)
        layer_names = data["layer_names"]
        matrix = np.array(data["matrix"])
        n = len(layer_names)

        print(f"\nSigma = {sigma:.2f} ({n} layers)")
        print("-" * 50)

        # Print adjacent-layer CKA (layer i vs layer i+1)
        print("  Adjacent-layer CKA:")
        for i in range(n - 1):
            print(f"    {layer_names[i]} <-> {layer_names[i+1]}: {matrix[i, i+1]:.4f}")

        # Print first-layer vs all CKA
        print(f"  Layer 0 vs others:")
        for i in range(1, n):
            print(f"    {layer_names[0]} <-> {layer_names[i]}: {matrix[0, i]:.4f}")

        # Mean off-diagonal CKA
        mask = ~np.eye(n, dtype=bool)
        mean_cka = matrix[mask].mean()
        print(f"  Mean off-diagonal CKA: {mean_cka:.4f}")

    print("\nInterpretation:")
    print("  CKA ~ 1.0 → layers learn very similar representations")
    print("  CKA ~ 0.0 → layers learn different representations")
    print("  Clusters of high CKA → processing stages (e.g., early=synoptic, late=local)")
    print("  CKA changes across sigma → sigma-dependent specialization")


def main():
    parser = argparse.ArgumentParser(description="CKA layer similarity for AIFSDD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sigmas", nargs="+", type=float,
                        default=[0.1, 1.0, 5.0, 20.0, 80.0])
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

    run_cka_analysis(
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
