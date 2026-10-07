"""Applied-schedule probe, few-step sampler screen (2026-09-30; stage-2 copy 2026-10-01: every scheduler
attribute read with getattr, so a scheduler without num_steps/sigma_max/sigma_min (e.g. the custom one) cannot
make the probe raise inside a production job).

Copy of the RW50k sampler campaign's probe
  /home/ecm5702/agent-work/20260929-RW50k-sampler/scripts/probe_sampler.py
extended in two ways, both print-only: (1) it wraps get_schedule of EVERY registered noise
scheduler (the campaign copy wrapped only the piecewise one, so a karras or exponential arm printed
no schedule), and (2) it counts the denoiser calls of each EDMHeunSampler.sample by wrapping the
denoising_fn it receives. The wrappers call the originals and return their results unchanged.

Runs eval.predict.main exactly as `python -m eval.predict.main` would (same argv, cwd first on
sys.path). Normally reached through scripts/probe_cli.py, which substitutes this file for
`-m eval.predict.main` in the command eval.cli predict builds.
"""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
from anemoi.models.samplers import diffusion_samplers as ds  # noqa: E402

RANK = os.environ.get("SLURM_PROCID", "?")
_n = {"sched": 0, "sample": 0}
_MAXPRINT = 2


def _wrap_sched(cls):
    orig = cls.get_schedule

    def _sched(self, *a, **k):
        s = orig(self, *a, **k)
        _n["sched"] += 1
        if _n["sched"] <= _MAXPRINT:
            vals = [float(v) for v in s.flatten()]
            st = getattr(self, "sigma_transition", None)
            above = sum(v > st for v in vals) if st is not None else "na"
            sig = ",".join("%.6g" % v for v in vals)
            print(
                f"PROBE_SCHEDULE rank={RANK} call={_n['sched']} class={type(self).__name__} "
                f"num_steps={getattr(self, 'num_steps', 'na')} sigma_max={getattr(self, 'sigma_max', 'na')} "
                f"sigma_min={getattr(self, 'sigma_min', 'na')} "
                f"num_steps_high={getattr(self, 'num_steps_high', 'na')} "
                f"num_steps_low={getattr(self, 'num_steps_low', 'na')} sigma_transition={st} "
                f"high={getattr(self, 'high_schedule_type', 'na')} low={getattr(self, 'low_schedule_type', 'na')} "
                f"rho={getattr(self, 'rho', 'na')} rho_high={getattr(self, 'rho_high', 'na')} "
                f"rho_low={getattr(self, 'rho_low', 'na')} "
                f"len={len(vals)} n_above_transition={above} sigmas=[{sig}]",
                flush=True,
            )
        return s

    cls.get_schedule = _sched


for _cls in sorted(set(ds.NOISE_SCHEDULERS.values()), key=lambda c: c.__name__):
    _wrap_sched(_cls)

_orig_sample = ds.EDMHeunSampler.sample


def _sample(self, x, y, sigmas, denoising_fn, *a, **k):
    _n["sample"] += 1
    calls = [0]

    def _counted(*aa, **kk):
        calls[0] += 1
        return denoising_fn(*aa, **kk)

    out = _orig_sample(self, x, y, sigmas, _counted, *a, **k)
    if _n["sample"] <= _MAXPRINT:
        eff = {key: k.get(key, getattr(self, key)) for key in ("S_churn", "S_min", "S_max", "S_noise")}
        print(
            f"PROBE_SAMPLER rank={RANK} call={_n['sample']} class={type(self).__name__} applied={eff} "
            f"n_sigmas={len(sigmas)} denoiser_calls={calls[0]}",
            flush=True,
        )
    return out


ds.EDMHeunSampler.sample = _sample
import eval  # noqa: E402
import manual_inference  # noqa: E402

print(
    f"PROBE_PATHS rank={RANK} eval={eval.__file__} manual_inference={manual_inference.__file__} "
    f"diffusion_samplers={ds.__file__}",
    flush=True,
)
if __name__ == "__main__":
    sys.argv[0] = "eval.predict.main"
    runpy.run_module("eval.predict.main", run_name="__main__", alter_sys=True)
