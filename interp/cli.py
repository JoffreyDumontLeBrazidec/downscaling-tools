"""Single entry point + shared argparse builders for the interp tools.

Usage
-----
    cd ~/dev/downscaling-tools
    python -m interp <tool> [tool args]

Tools: permutation, ablation, activations, patching, ig, report, compare.
Every tool shares the same model / event-bundle / sigma flags (see
add_model_args / add_event_args / add_sigma_args).
"""

from __future__ import annotations

import argparse
import logging
import sys

# One canonical sigma ladder for all tools (log-spaced through the EDM range).
# Pass --sigmas to densify for smooth profiles.
CANONICAL_SIGMAS = [0.1, 1.0, 5.0, 20.0, 80.0]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def add_model_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("model")
    g.add_argument("--checkpoint", required=True, help="Path to training checkpoint")
    g.add_argument("--output-dir", required=True, help="Output directory")
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    g.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    return p


def add_event_args(p: argparse.ArgumentParser):
    """Event-bundle source flags. --event names a registry entry in
    interp.core.data.EVENTS; explicit flags override its fields."""
    g = p.add_argument_group("event bundles (real .nc bundles, never val_dataloader)")
    g.add_argument("--event", default=None,
                   help="Named event from interp.core.data.EVENTS "
                        "(e.g. franklin_o96_o320, franklin_o96_o320_m4)")
    g.add_argument("--bundle-dir", default=None,
                   help="Dir of real event .nc bundles (overrides --event's dir)")
    g.add_argument("--dates", nargs="+", default=None)
    g.add_argument("--members", nargs="+", default=None)
    g.add_argument("--steps", nargs="+", default=None,
                   help="Zero-padded step hours; each (date,member,step) is one batch element.")
    return p


def add_sigma_args(p: argparse.ArgumentParser, default=None):
    p.add_argument("--sigmas", nargs="+", type=float,
                   default=list(default if default is not None else CANONICAL_SIGMAS))
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tools = {
        "permutation": ("interp.tools.permutation", "feature permutation importance "
                        "(per-sigma denoiser or full-sampling mode)"),
        "ablation": ("interp.tools.ablation", "conditioning ablation (zero lres/hres/noisy target)"),
        "activations": ("interp.tools.activations", "activation norms + CKA layer similarity"),
        "patching": ("interp.tools.patching", "causal activation patching (residual/grid_region/stage)"),
        "ig": ("interp.tools.ig", "integrated gradients (mean/box/eye/tail/spectral functionals)"),
        "report": ("interp.viz.report", "render the per-checkpoint PDF report from JSONs"),
        "compare": ("interp.compare", "summary-scalar comparison across run dirs"),
    }
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        for name, (_, desc) in tools.items():
            print(f"  {name:12s} {desc}")
        return 0
    name = argv[0]
    if name not in tools:
        raise SystemExit(f"unknown tool {name!r}; options: {', '.join(tools)}")
    import importlib
    mod = importlib.import_module(tools[name][0])
    return mod.main(argv[1:])


if __name__ == "__main__":
    main()
