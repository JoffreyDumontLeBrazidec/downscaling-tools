"""GL512 patches to the prepml downscaling_unified runner (gl512-20261109-prepml-wrap only, 2026-11-09).

Two patches, both applied to season-scale-tools/downscaling_unified_runner.py at import, without
editing that shared file (it serves every other wrap).

1. Sampler routing. The runner routes development_hacks.extra_args by the signature of
   interface.predict_step. The 18a5415a interface takes only **kwargs, so the runner passes the keys one
   by one, and the model's predict_step keeps only num_steps/sigma_max/sigma_min/rho/schedule_type for
   the scheduler: num_steps_high, num_steps_low, high/low_schedule_type and sigma_transition fall into
   remaining kwargs and are ignored, and ExperimentalSamplerScheduler then splits 30 steps 15/15 instead
   of 10/20. Here the scheduler and sampler keys are passed as the noise_scheduler_params /
   sampler_params dicts the model's predict_step accepts (as the Jupiter GL512 read kit does), and each
   draw logs its denoiser call count (59 for the 30-step Heun sampler).

2. Standing rule 31 seeding.

Without this module the runner seeds ONCE per process (UnifiedDownscalingRunner._seed_procs: base
ANEMOI_BASE_SEED or 756, plus the rank) and every lead of the task then draws from wherever the
generator happens to be. A prepml model task is one (date, member) process, so every member of every
date starts from the same seed: member k on date d gets the same noise sequence as member j on date e.

This module wraps UnifiedDownscalingRunner.predict_step so that, just before each draw:
  * torch (CPU and CUDA) is seeded with derive_draw_seed(base, init yyyymmdd, lead_h, member), the
    formula of downscaling-tools eval/predict/seeding.py (copied verbatim: that repo is not on the
    prepml PYTHONPATH). base = ANEMOI_BASE_SEED (x1000 when < 1000) or 756, as resolve_base_seed();
    member = the prepml member number (output encoding `number`) minus 1, i.e. 0-based as documented
    by derive_draw_seed;
  * the initial noise is ONE global field: every rank holds the same seed, the 5-D torch.randn whose
    grid axis has this rank's shard size draws the full O1280 field and returns this rank's slice
    (balanced partition, the same split as shard_tensor). This is the GlobalNoise of the Jupiter GL512
    read kit (agent-work/20261030-gl512-100k-wide/kit/gl512_predict.py), so the draw does not depend on
    the rank count. Exactly one such draw per predict_step is required on multi-rank runs (fail loud).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

LOG = logging.getLogger(__name__)

DEFAULT_BASE_SEED = 756
_SEED_THRESHOLD = 1000
SEED_FORMULA = "numpy SeedSequence([base, yyyymmdd, lead_h, member]).generate_state(1, uint32)"


def resolve_base_seed() -> int:
    raw = os.environ.get("ANEMOI_BASE_SEED")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_BASE_SEED
    seed = int(str(raw).strip())
    if seed < _SEED_THRESHOLD:
        seed *= _SEED_THRESHOLD
    return seed


def derive_draw_seed(base: int, date: int, lead: int, member: int) -> int:
    seq = np.random.SeedSequence([int(base), int(date), int(lead), int(member)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _find_number(obj):
    """Return output.<writer>.encoding.number from the anemoi-inference run config, or None."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        enc = obj.get("encoding")
        if isinstance(enc, dict) and enc.get("number") is not None:
            return enc["number"]
        for value in obj.values():
            found = _find_number(value)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_number(value)
            if found is not None:
                return found
    return None


def _member_index(runner) -> int:
    number = _find_number(getattr(runner.config, "output", None))
    if number is None:
        raise SystemExit("gl512_runner_patches: no output encoding `number` in the run config; "
                         "cannot derive the rule-31 member seed")
    return int(number) - 1


class _GlobalNoise:
    """While active, a 5-D torch.randn whose grid axis (-2) has this rank's shard size draws the FULL
    field (n_grid points) and returns this rank's contiguous slice."""

    def __init__(self, n_grid: int, world: int, rank: int):
        from anemoi.models.distributed.balanced_partition import get_balanced_partition_sizes

        self.n_grid, self.world, self.rank = int(n_grid), int(world), int(rank)
        self.sizes = list(get_balanced_partition_sizes(self.n_grid, self.world))
        self.off = int(sum(self.sizes[: self.rank]))
        self.full, self.other = [], 0

    def _randn(self, *size, **kw):
        shp = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
        if self.world > 1 and len(shp) == 5 and int(shp[-2]) == self.sizes[self.rank]:
            z = self._orig(shp[:-2] + (self.n_grid,) + shp[-1:], **kw)
            self.full.append(float(z.double().sum().item()))
            return z.narrow(-2, self.off, int(shp[-2])).contiguous()
        self.other += 1
        return self._orig(*size, **kw)

    def __enter__(self):
        self.full, self.other = [], 0
        self._orig = torch.randn
        torch.randn = self._randn
        return self

    def __exit__(self, *exc):
        torch.randn = self._orig
        return False


def _route_extra_args_as_dicts(params, kw, extra_args):
    """Replacement for downscaling_unified_runner._route_extra_args (patch 1)."""
    import downscaling_unified_runner as dur

    ns = {k: v for k, v in extra_args.items() if k in dur._NOISE_SCHEDULER_KEYS}
    sp = {k: v for k, v in extra_args.items() if k in dur._SAMPLER_KEYS}
    rest = {k: v for k, v in extra_args.items() if k not in dur._NOISE_SCHEDULER_KEYS | dur._SAMPLER_KEYS}
    if ns:
        kw["noise_scheduler_params"] = ns
    if sp:
        kw["sampler_params"] = sp
    kw.update(rest)
    return kw


def _count_denoiser_calls(model):
    """Wrap the denoiser of the loaded interface once with a call counter; return the counter list."""
    inner = getattr(model, "model", model)
    calls = getattr(inner, "_gl512_calls", None)
    if calls is None:
        calls = [0]
        fwd = inner.fwd_with_preconditioning

        def _counted(*a, **k):
            calls[0] += 1
            return fwd(*a, **k)

        inner.fwd_with_preconditioning = _counted
        inner._gl512_calls = calls
    return calls


def _patch() -> None:
    import downscaling_unified_runner as dur

    cls = dur.UnifiedDownscalingRunner
    if getattr(cls, "_gl512_draw_seeding", False):
        return
    dur._route_extra_args = _route_extra_args_as_dicts
    orig_predict = cls.predict_step
    orig_prepare_hres = cls._prepare_high_res_input_tensor

    def _prepare_high_res_input_tensor(self, *args, **kwargs):
        tensor = orig_prepare_hres(self, *args, **kwargs)
        self._gl512_n_grid = int(tensor.shape[-2])
        return tensor

    def predict_step(self, model, input_tensor_torch, *args, **kwargs):
        date = args[0] if len(args) >= 1 else kwargs["date"]
        step = args[1] if len(args) >= 2 else kwargs["step"]
        init = date - step
        lead_h = int(step.total_seconds() // 3600)
        member = _member_index(self)
        base = resolve_base_seed()
        seed = derive_draw_seed(base, int(init.strftime("%Y%m%d")), lead_h, member)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        world = int(getattr(self, "world_size", 1) or 1) if self.model_comm_group is not None else 1
        rank = int(getattr(self, "global_rank", 0) or 0)
        LOG.info("GL512 draw seeded: base=%d init=%s lead=%d member=%d (0-based) seed=%d world=%d formula=%s",
                 base, init.strftime("%Y%m%d"), lead_h, member, seed, world, SEED_FORMULA)
        calls = _count_denoiser_calls(model)
        calls[0] = 0
        if world == 1:
            out = orig_predict(self, model, input_tensor_torch, *args, **kwargs)
            LOG.info("GL512 draw done: lead=%d denoiser_calls=%d", lead_h, calls[0])
            return out
        gn = _GlobalNoise.__new__(_GlobalNoise)
        gn.world, gn.rank = world, rank

        # n_grid is known only once the high-res input is prepared inside orig_predict; resolve lazily.
        def _lazy_randn(*size, **kw):
            if not hasattr(gn, "sizes"):
                _GlobalNoise.__init__(gn, self._gl512_n_grid, world, rank)
                gn._orig = orig_randn
            return gn._randn(*size, **kw)

        orig_randn = torch.randn
        torch.randn = _lazy_randn
        try:
            out = orig_predict(self, model, input_tensor_torch, *args, **kwargs)
        finally:
            torch.randn = orig_randn
        full = getattr(gn, "full", [])
        LOG.warning("GL512 noise rank=%d lead=%d seed=%d full_draws=%d full_sum=%s other_randn=%d denoiser_calls=%d",
                    rank, lead_h, seed, len(full), full, getattr(gn, "other", 0), calls[0])
        if len(full) != 1:
            raise SystemExit(f"gl512_runner_patches: rank {rank}: {len(full)} full-field noise draws in one "
                             f"predict_step (want 1)")
        return out

    cls._prepare_high_res_input_tensor = _prepare_high_res_input_tensor
    cls.predict_step = predict_step
    cls._gl512_draw_seeding = True
    LOG.info("gl512_runner_patches: sampler dicts routed; predict_step wrapped (rule 31, global noise)")


_patch()
