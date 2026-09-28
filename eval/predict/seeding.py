"""Deterministic, shard-independent seeding for downscaling inference.

Two defects motivated this module, both measured and read from source on
2026-08-31 (note: epics/unified-inference/in-progress/
20260831_inference_seeding_blocks_parity_gates.md).

1. NO inference path pinned a seed. `ANEMOI_BASE_SEED` was exported by the
   training and sigma-evaluator templates only, never by a prediction path, so
   `anemoi.inference.runners.parallel._seed_procs` fell back to
   `torch.initial_seed()` -- a fresh value on every process launch. Two runs of
   the same configuration therefore produced different draws. Measured on the
   j9xm (32 chunks) / j9yd (1 chunk) pair: zero of 6,599,680 grid points match,
   and the difference is 54-78% of the member-to-member spread. That makes any
   pointwise before-and-after gate impossible.

2. `_seed_procs` seeds only the NON-ZERO ranks. `torch.manual_seed` sits inside
   its `else` branch, so rank 0 computes the seed, broadcasts it, and never
   applies it -- while every other rank receives the SAME seed. That matters
   because the initial diffusion noise is drawn AFTER the grid is sharded:
   `_before_sampling` shards the upsampled input along the grid dimension, then
   `predict_step` reads `grid_size = x_in_lres_upsampled.shape[-2]` (the LOCAL
   shard) and draws `torch.randn` of that shape. Shards are balanced to within
   one point, and the global O1280 grid (6,599,680) divides EXACTLY by four, so
   a standard four-rank run drew bit-identical noise on ranks 1, 2 and 3 --
   three quarters of the globe starting from the same noise field.

`seed_inference` fixes both at once: every rank is seeded, and each rank derives
a DISTINCT seed from a single base, so per-shard noise is independent while the
whole run stays reproducible from `ANEMOI_BASE_SEED`.

Reproducibility contract. Seeding happens ONCE per process, in
`setup_distributed`, before the loop over date/step pairs; members are drawn
inside that loop and advance the generator, so each member still gets its own
noise. Two runs therefore reproduce each other only when ALL of the following
match:

  * the base seed (`ANEMOI_BASE_SEED`);
  * the rank count, because shard boundaries set the per-shard noise shape;
  * the ordering AND count of dates, steps and members, because the generator is
    consumed in sequence -- predicting a subset does NOT reproduce those same
    members inside a fuller run.
  * the GPU class, added 2026-09-04 -- see below.

Any before-and-after gate must hold all four fixed and vary only the knob under
test. Code that wants genuinely independent draws (for example the ladder's
candidate-B seed-draw loop) must vary `ANEMOI_BASE_SEED` explicitly; re-running
the same command no longer suffices.

The GPU-class condition (measured 2026-09-04). Seeding alone does NOT make the
draw portable between machines. From one base seed, `torch.randn` on CUDA yields
bit-identical values on Atos AC (A100, torch 2.10.0), Atos AG (GH200, torch
2.9.0) and JUPITER (GH200, torch 2.9.0) for small tensors -- shapes (8,) and
(1024,) all hash to 8f848bed.../ee2edd5d... -- but DIVERGES at the shape that
actually matters, the per-shard draw of (1_649_920, 10): AG and JUPITER both give
da635fe0adcc049e while AC gives f325fa5ca94a83e0. Large-tensor generation maps
elements to Philox counters through the kernel launch configuration, which
follows the device, so the streams are different orderings of the same
distribution. Both remain proper standard normals (std 0.99975 vs 0.99998).

Consequence: a run is reproducible within a GPU class and not across one. A
JUPITER prediction and an AG prediction of the same configuration agree to
floating-point noise; the same prediction on AC differs by roughly the ensemble
spread (measured 0.30 of the 10v standard deviation over 10 members at O1280),
which is a DIFFERENT DRAW, not a wrong answer. Scores and distributions stay
valid across hosts; only pointwise comparison breaks. Compare JUPITER against
AG, never against AC.

Caveat: AG and JUPITER share both the GPU generation and the torch version,
while AC differs in both, so this evidence cannot separate the two causes. The
practical rule is the same either way -- hold the host fixed across a pointwise
gate.
"""

from __future__ import annotations

import logging
import os

import torch

LOG = logging.getLogger(__name__)

# Matches the value already used by the training and sigma-evaluator templates,
# so a run that pins nothing behaves like the rest of the project rather than
# drawing a fresh seed every launch.
DEFAULT_BASE_SEED = 756

# Upstream `_seed_procs` widens small seeds by this factor; mirrored so that a
# given ANEMOI_BASE_SEED keeps the meaning it has everywhere else in the stack.
_SEED_THRESHOLD = 1000


def resolve_base_seed(env: dict[str, str] | None = None) -> int:
    """Return the base seed, from ANEMOI_BASE_SEED or the project default."""
    environ = os.environ if env is None else env
    raw = environ.get("ANEMOI_BASE_SEED")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_BASE_SEED
    seed = int(str(raw).strip())
    if seed < _SEED_THRESHOLD:
        seed *= _SEED_THRESHOLD
    return seed


def derive_rank_seed(base_seed: int, global_rank: int) -> int:
    """Derive this rank's seed. Distinct per rank, deterministic in the base."""
    return int(base_seed) + int(global_rank)


def seed_inference(global_rank: int = 0, *, base_seed: int | None = None) -> int:
    """Seed this process for inference and return the seed actually applied.

    Call once per process, after the rank is known and before the model runs.
    """
    base = resolve_base_seed() if base_seed is None else int(base_seed)
    seed = derive_rank_seed(base, global_rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    LOG.info(
        "inference seeded: base=%d rank=%d seed=%d (set ANEMOI_BASE_SEED to change)",
        base,
        global_rank,
        seed,
    )
    return seed
