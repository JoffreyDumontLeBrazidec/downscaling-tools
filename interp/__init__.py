"""Mechanistic interpretability toolkit for AIFSDD (Anemoi diffusion downscaler).

Entry point:  python -m interp <tool>   (see interp.cli)

Tools (interp/tools/):
  - permutation : which input variables matter (per-sigma or end-to-end)?
  - ablation    : how much does each conditioning pathway carry at each sigma?
  - activations : layer activation norms + CKA layer similarity vs sigma
  - patching    : which block/stage causally commits each surface field?
  - ig          : integrated gradients onto the inputs, for mean/box/eye/
                  tail/spectral output functionals (storm- and extreme-centered)

Shared primitives live in interp/core/ (model, data/events, geometry, regions,
hooks, runmeta); rendering in interp/viz/; cross-run comparison in compare.py.
"""
