"""How the initial diffusion noise relates across the lead times of one member (TD-228).

Today the generator is seeded once per process and consumed in sequence over (date, step,
member), so lead 24 of member 3 draws a noise field independent of lead 12 of member 3
(`seeding.py`, "Reproducibility contract"). This module adds two other conventions, chosen
with ``--noise-across-steps`` of ``eval.predict.main``:

- ``independent`` (default): unchanged, the generator runs on.
- ``shared``: one noise field per (date, member), reused at every step. Implemented by
  re-seeding the generator from (base seed, rank, date, member) right before each member's
  prediction, so the first draw of every step of that member is the same field (same shard
  shape on the same rank). Nothing is patched. This is the fixed-noise convention of Andrae
  et al. (arXiv 2410.05431) and the frozen seed of spateGAN-ERA5 (arXiv 2411.16098).
- ``ou``: an Ornstein-Uhlenbeck chain over the steps of one (date, member):
  ``noise_k = rho * noise_{k-1} + sqrt(1 - rho^2) * fresh_k``. Every marginal stays standard
  normal, only the lead-to-lead correlation changes (Andrae et al., Algorithm 2). The first
  step draws exactly the ``shared`` field (same seed), so an OU run and a shared run agree on
  their first lead. Implemented by wrapping ``torch.randn`` during the member's prediction:
  the lane's model draws its initial field with one ``torch.randn`` call
  (``diffusiondownscaler_encoder_processor_decoder.py``, ``y_init``), and churn is off on the
  baseline sampler, so exactly one call is expected per member and step; the count is
  recorded (``draw_counts``) and a count other than one is logged as a warning, because then
  the chain also touched a draw it was not meant for.

The steps of one date must be predicted in increasing order for ``ou`` (the chain runs along
the steps); ``main.py`` sorts them. Members are chained independently.
"""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Iterator

LOG = logging.getLogger(__name__)

MODES = ("independent", "shared", "ou")


def member_seed(rank_seed: int, date: str, member: int, step: int | None = None) -> int:
    """Deterministic seed for one (date, member[, step]) on this rank; fits in 2**31."""
    s = (int(rank_seed) * 1_000_003 + int(date) * 7_919 + int(member) * 104_729) % (2**31 - 1)
    if step is not None:
        s = (s + int(step) * 15_485_863) % (2**31 - 1)
    return int(s)


class NoiseAcrossSteps:
    """Context manager factory: ``with policy.member_context(date, step, member): predict``."""

    def __init__(self, mode: str, rho: float | None, rank_seed: int) -> None:
        if mode not in MODES:
            raise ValueError(f"noise_across_steps must be one of {MODES}, got {mode!r}")
        if mode == "ou":
            if rho is None or not (0.0 <= float(rho) <= 1.0):
                raise ValueError("--noise-rho in [0, 1] is required with --noise-across-steps ou")
        self.mode = mode
        self.rho = None if rho is None else float(rho)
        self.rank_seed = int(rank_seed)
        self._prev: dict[tuple, object] = {}      # (date, member, shape, dtype, device) -> tensor
        self._first_step: dict[tuple[str, int], int] = {}
        self.draw_counts: dict[tuple[str, int, int], int] = {}   # (date, step, member) -> randn calls

    def requires_sorted_steps(self) -> bool:
        return self.mode == "ou"

    def describe(self) -> dict:
        return {"mode": self.mode, "rho": self.rho, "rank_seed": self.rank_seed}

    def _reseed(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @contextlib.contextmanager
    def member_context(self, date: str, step: int, member: int) -> Iterator[None]:
        if self.mode == "independent":
            yield
            return
        if self.mode == "shared":
            seed = member_seed(self.rank_seed, date, member)
            LOG.info("noise shared across steps: date=%s step=%d member=%d seed=%d", date, step, member, seed)
            self._reseed(seed)
            yield
            return

        # ou
        import torch

        key_dm = (date, int(member))
        first = self._first_step.setdefault(key_dm, int(step))
        if int(step) < first:
            raise RuntimeError(f"ou noise chain: step {step} came after a later step for {key_dm}; sort the steps")
        seed = member_seed(self.rank_seed, date, member, None if int(step) == first else int(step))
        self._reseed(seed)
        rho = float(self.rho)
        count_key = (date, int(step), int(member))
        self.draw_counts[count_key] = 0
        original_randn = torch.randn
        policy = self

        def chained_randn(*args, **kwargs):
            fresh = original_randn(*args, **kwargs)
            policy.draw_counts[count_key] += 1
            key = (date, int(member), tuple(fresh.shape), str(fresh.dtype), str(fresh.device))
            prev = policy._prev.get(key)
            if prev is None or int(step) == first:
                out = fresh
            else:
                out = rho * prev + math.sqrt(max(0.0, 1.0 - rho * rho)) * fresh
            policy._prev[key] = out.detach().clone()
            return out

        torch.randn = chained_randn
        try:
            yield
        finally:
            torch.randn = original_randn
        n = self.draw_counts[count_key]
        if n != 1:
            LOG.warning("ou noise chain: %d torch.randn calls for date=%s step=%d member=%d (expected 1: "
                        "the initial field only; churn must be off)", n, date, step, member)
        else:
            LOG.info("ou noise chain rho=%.3f: date=%s step=%d member=%d (first step %d)", rho, date, step,
                     member, first)
