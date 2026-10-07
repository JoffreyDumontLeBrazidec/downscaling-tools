#!/usr/bin/env python3
"""Write the arm lanes of campaign T1d stage A (sampler screen on the 1.2M parent, 2026-10-07).

Each arm is a lane `tc_o320_o1280_<arm>` on the base `tc_o320_o1280_p12m_ctrl` (1.2M parent, run 551dfd1e,
step 400,000; Franklin-Idalia box) that changes sampler keys only, through `predict.sampler_overrides`
(merged key by key by eval/config/loader.py). Every arm is Heun with churn off (S_churn 0, S_noise 1.0;
at S_churn 0 the Heun loop draws no noise, so S_noise is inert and the arms share the initial noise
draw per member), sigma_min 0.03.

  p12m_pw30_c0      piecewise 30 (10 exp + 20 Karras rho 7), sigma_max 1e5, transition 10   59 calls
  p12m_c0_pw16_s1k  piecewise 16 (5 exp + 11 Karras rho 7), sigma_max 1e3 (S_max 1e3)        31 calls
  p12m_st2          explicit list: 1e5, 1e4, then 1e3 and the nodes of p12m_c0_pw16_s1k     35 calls
  p12m_st4          explicit list: 1e5, 3e4, 1e4, 3e3, then 1e3 and the same nodes          39 calls

The explicit lists take the nodes below 1e3 of p12m_c0_pw16_s1k bit for bit: `applied_sigmas` re-implements
the fork's ExperimentalSamplerScheduler (anemoi-core 27391c1, float64; torch.linspace's symmetric fill
included), and `--verify-fork <diffusion_samplers.py>` runs the fork's own schedulers (needs torch) and
checks every arm's applied schedule against it, bit for bit.

Usage:  python3 make_lanes_t1d.py --out <eval/config/lanes dir> [--force] [--verify-fork <path>] [--json <file>]
It never overwrites an existing lane file unless --force is given. It prints the arm table and the lists.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fewstep_sampler_20260930"))
import make_lanes as ml  # noqa: E402  (pw, custom, calls, yaml_scalar, PW30)

BASE = "tc_o320_o1280_p12m_ctrl"
TRANSITION = 10.0  # the boundary of the "high" and "low" segment in the ln-step table


# ---- faithful float64 re-implementation of the fork's schedulers ---------------------------------
def _linspace_torch(start: float, end: float, n: int) -> np.ndarray:
    """torch.linspace on CPU in float64: the first half counts up from start, the second half down
    from end (numpy's linspace counts up all the way and differs by an ulp on some points)."""
    if n == 1:
        return np.array([start])
    step = (end - start) / (n - 1)
    i = np.arange(n, dtype=np.float64)
    half = n // 2
    return np.where(i < half, start + step * i, end - step * (n - 1 - i))


def _exponential_segment(a: float, b: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([a])
    return np.exp(_linspace_torch(math.log(a), math.log(b), n))


def _karras_segment(a: float, b: float, n: int, rho: float) -> np.ndarray:
    if n <= 1:
        return np.array([a])
    i = np.arange(n, dtype=np.float64)
    return (a ** (1.0 / rho) + i / (n - 1.0) * (b ** (1.0 / rho) - a ** (1.0 / rho))) ** rho


def _segment(kind, a, b, n, rho):
    return _exponential_segment(a, b, n) if kind == "exponential" else _karras_segment(a, b, n, rho)


def applied_sigmas(block: dict) -> list[float]:
    """The positive levels the runtime applies (the terminal zero is not listed)."""
    st = block["schedule_type"]
    if st in ("custom", "explicit"):
        return [float(v) for v in block["sigmas"]]
    if st in ("experimental_piecewise", "experimental_sampler"):
        rho = float(block.get("rho", 7.0))
        hi = _segment(block["high_schedule_type"], float(block["sigma_max"]), float(block["sigma_transition"]),
                      int(block["num_steps_high"]) + 1, float(block.get("rho_high", rho)))
        lo = _segment(block["low_schedule_type"], float(block["sigma_transition"]), float(block["sigma_min"]),
                      int(block["num_steps_low"]), float(block.get("rho_low", rho)))
        return [float(v) for v in np.concatenate([hi, lo[1:]])]
    raise SystemExit(f"applied_sigmas: schedule_type {st} not handled here")


# ---- the arms ------------------------------------------------------------------------------------
def c0(n_high, n_low, **kw):
    """Heun piecewise, churn off, S_noise 1.0 (inert at churn 0)."""
    return ml.pw(n_high, n_low, S_churn=0.0, S_noise=1.0, **kw)


PW30_C0 = c0(10, 20)
C0_PW16_S1K = c0(5, 11, sigma_max=1000.0)  # pw() sets S_max = sigma_max = 1e3
_ARM2 = applied_sigmas(C0_PW16_S1K)
BELOW_1K = [s for s in _ARM2 if s < 999.0]  # the 15 nodes strictly below the top (the top is 999.9999999999998)


def sparse_top(top):
    return ml.custom(list(top) + [1000.0] + BELOW_1K)


ARMS = {
    "p12m_pw30_c0": PW30_C0,
    "p12m_c0_pw16_s1k": C0_PW16_S1K,
    "p12m_st2": sparse_top([100000.0, 10000.0]),
    "p12m_st4": sparse_top([100000.0, 30000.0, 10000.0, 3000.0]),
}
NOTES = {
    "p12m_pw30_c0": "quality reference, 30 steps churn off; runs at base seeds 756 and 757000",
    "p12m_c0_pw16_s1k": "cost reference, the T1 few-step sampler (16 steps, sigma_max 1e3)",
    "p12m_st2": "sparse top: 1e5 and 1e4 above the cost reference's 1e3 start",
    "p12m_st4": "sparse top: 1e5, 3e4, 1e4, 3e3 above the cost reference's 1e3 start",
}


def lane_name(arm: str) -> str:
    return f"tc_o320_o1280_{arm}"


def lane_text(arm: str, block: dict) -> str:
    lines = [
        f"# campaign T1d stage A (2026-10-07), arm {arm}: {NOTES[arm]}.",
        f"# Written by scripts/t1d_sampler_20261007/make_lanes_t1d.py; {ml.calls(block)} denoiser calls per member.",
        "# Only sampler keys change; sampler_overrides merges key by key onto the base lane's block.",
        "# The checkpoint (1.2M parent, 551dfd1e step 400k) comes from the base lane's predict.checkpoint.",
        f"base: {BASE}",
        "",
        "predict:",
        "  sampler_overrides:",
    ]
    lines += [f"    {k}: {ml.yaml_scalar(v)}" for k, v in block.items()]
    return "\n".join(lines) + "\n"


def ln_steps(sig: list[float]):
    """Largest ln(sigma_i / sigma_i+1) over the steps above the transition (both ends >= 10) and below it
    (upper end <= 10, down to sigma_min; the final step to zero is excluded)."""
    hi = [math.log(a / b) for a, b in zip(sig[:-1], sig[1:]) if b >= TRANSITION * (1 - 1e-12)]
    lo = [math.log(a / b) for a, b in zip(sig[:-1], sig[1:]) if a <= TRANSITION * (1 + 1e-12)]
    return (max(hi) if hi else float("nan")), (max(lo) if lo else float("nan"))


def verify_fork(path: str) -> int:
    """Run the fork's own schedulers on every arm and compare bit for bit."""
    import importlib.util
    import types

    for name in ("anemoi", "anemoi.models", "anemoi.models.distributed", "anemoi.models.distributed.shapes"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["anemoi.models.distributed.shapes"].DatasetShardSizes = dict
    spec = importlib.util.spec_from_file_location("fork_ds", path)
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    bad = 0
    drop = {"schedule_type", "sampler", "S_churn", "S_min", "S_max", "S_noise"}
    for arm, block in ARMS.items():
        cls = ds.NOISE_SCHEDULERS[block["schedule_type"]]
        sched = cls(**{k: v for k, v in block.items() if k not in drop}).get_schedule()
        fork = [float(v) for v in sched]
        mine = applied_sigmas(block) + [0.0]
        same = fork == mine
        bad += not same
        print(f"VERIFY {arm}: fork {cls.__name__} {len(fork) - 1} levels + 0; bitwise equal: {same}")
        if not same:
            print("   fork:", fork, "\n   mine:", mine)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the lanes directory (eval/config/lanes of the worktree)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-fork", default=None, help="path to the fork's diffusion_samplers.py (needs torch)")
    ap.add_argument("--json", default=None, help="also write the arm table as JSON here")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / f"{BASE}.yaml").exists():
        print(f"WARNING: base lane {BASE}.yaml is not in {out}; the arms will not resolve there", file=sys.stderr)
    paths = {arm: out / f"{lane_name(arm)}.yaml" for arm in ARMS}
    clash = [str(p) for p in paths.values() if p.exists()]
    if clash and not args.force:
        raise SystemExit("refusing to overwrite (use --force): " + ", ".join(clash))
    table = []
    print(f"{'arm':18s} {'schedule':24s} {'steps':>5s} {'calls':>5s} {'max ln-step hi':>14s} {'max ln-step lo':>14s}")
    for arm, block in ARMS.items():
        paths[arm].write_text(lane_text(arm, block))
        sig = applied_sigmas(block)
        hi, lo = ln_steps(sig)
        sched = block["schedule_type"] if block["schedule_type"] == "custom" else (
            f"pw {block['num_steps_high']}+{block['num_steps_low']} s{block['sigma_max']:.0e}")
        print(f"{arm:18s} {sched:24s} {len(sig):5d} {ml.calls(block):5d} {hi:14.3f} {lo:14.3f}")
        table.append({"arm": arm, "lane": lane_name(arm), "schedule": sched, "steps": len(sig),
                      "calls": ml.calls(block), "max_ln_step_high": hi, "max_ln_step_low": lo,
                      "sigmas": sig, "block": block})
    print("\nApplied positive levels (the runtime appends the terminal 0); log10 step and ln step to the next level:")
    for t in table:
        sig = t["sigmas"]
        print(f"\n{t['arm']} ({t['steps']} levels, {t['calls']} calls):")
        for i, s in enumerate(sig):
            nxt = sig[i + 1] if i + 1 < len(sig) else None
            step = f"  dlog10 {math.log10(s / nxt):6.3f}  dln {math.log(s / nxt):6.3f}" if nxt else "  -> 0"
            print(f"  {i:2d}  {s!r:>22s}{step}")
    print("\nHeun high-segment rule (T1): ln-step <= 1.15 safe, >= 1.3 fails. Arms above 1.15 in the high segment:",
          ", ".join(f"{t['arm']} ({t['max_ln_step_high']:.3f})" for t in table if t["max_ln_step_high"] > 1.15) or "none")
    for arm, p in paths.items():
        print("wrote", p)
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
    if args.verify_fork:
        sys.exit(1 if verify_fork(args.verify_fork) else 0)


if __name__ == "__main__":
    main()
