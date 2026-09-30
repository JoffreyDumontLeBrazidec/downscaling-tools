#!/usr/bin/env python3
"""Write the lane files of the few-step sampler screen (stage 1, 2026-09-30).

Each arm is a lane that inherits everything from a base lane and changes only sampler
keys through ``predict.sampler_overrides`` (merged key by key by eval/config/loader.py).
Two bases: ``_tc_hres_ULR_ctrl`` for RW50k (the RW50k sampler campaign's cyclone-screen
lane, untracked, AC hres-lead runtime) and ``tc_o320_o1280_ft400k_gmass_ctrl`` for the
donor 12dcefea at 400k (tracked, pristine runtime).

Usage:  python3 make_lanes.py --out <dir of eval/config/lanes> [--stage 1a|1b|all]
It never overwrites an existing file unless --force is given. It prints the arm table.

Denoiser calls per member: Heun 2N-1, DPM-Solver++ 2M N.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PW30 = {  # the lane standard, restated so every arm's block is complete and self-describing
    "schedule_type": "experimental_piecewise", "num_steps": 30, "sigma_max": 100000.0,
    "sigma_transition": 10.0, "sigma_min": 0.03, "high_schedule_type": "exponential",
    "low_schedule_type": "karras", "num_steps_high": 10, "num_steps_low": 20, "rho": 7.0,
    "sampler": "heun", "S_churn": 2.5, "S_min": 0.75, "S_max": 100000.0, "S_noise": 1.05,
}


def pw(n_high, n_low, **kw):
    d = dict(PW30, num_steps=n_high + n_low, num_steps_high=n_high, num_steps_low=n_low)
    d.update(kw)
    if "sigma_max" in kw and "S_max" not in kw:
        d["S_max"] = kw["sigma_max"]
    return d


def karras(n, **kw):
    d = dict(PW30, schedule_type="karras", num_steps=n)
    # the piecewise keys stay in the block (the scheduler ignores them); make them consistent anyway
    d.update(num_steps_high=max(1, n // 3), num_steps_low=n - max(1, n // 3))
    d.update(kw)
    return d


def exponential(n, **kw):
    d = dict(PW30, schedule_type="exponential", num_steps=n)
    d.update(num_steps_high=max(1, n // 3), num_steps_low=n - max(1, n // 3))
    d.update(kw)
    return d


def dpm(n_high, n_low, **kw):
    # DPM-Solver++ 2M is deterministic: churn keys are inert but recorded as 0 to say so.
    return pw(n_high, n_low, sampler="dpmpp_2m", S_churn=0.0, S_noise=1.0, **kw)


def calls(block):
    n = block["num_steps"]
    return n if block["sampler"] == "dpmpp_2m" else 2 * n - 1


# ---- stage 1a: no code change needed (runs on the pinned campaign code) -------------------
STAGE_1A = {
    # the step ladder at the lane's own split ratio (1 high : 2 low), transition 10, sigma_max 1e5
    "pw24": pw(8, 16), "pw20": pw(7, 13), "pw16": pw(5, 11), "pw12": pw(4, 8), "pw8": pw(3, 5), "pw6": pw(2, 4),
    # where do the steps go? split and transition at 16 steps
    "pw16_h8l8": pw(8, 8), "pw16_h3l13": pw(3, 13),
    "pw16_t3": pw(5, 11, sigma_transition=3.0), "pw16_t30": pw(5, 11, sigma_transition=30.0),
    # the noise ceiling at few steps (matrix7: few-step grain grows with sigma_max)
    "pw16_s10k": pw(5, 11, sigma_max=10000.0), "pw12_s10k": pw(4, 8, sigma_max=10000.0),
    "pw12_s1k": pw(4, 8, sigma_max=1000.0),
    # the spacing of the low segment
    "pw16_rl3": pw(5, 11, rho_low=3.0), "pw16_rl12": pw(5, 11, rho_low=12.0),
    # churn off at few steps (the RW50k campaign measures it at 30)
    "pw20_c0": pw(7, 13, S_churn=0.0), "pw16_c0": pw(5, 11, S_churn=0.0), "pw12_c0": pw(4, 8, S_churn=0.0),
    # one-segment schedules
    "kar16": karras(16), "kar12": karras(12), "exp16": exponential(16), "exp12": exponential(12),
}
# ---- stage 1b: needs the patched runtime (anemoi-core claude/project-thread-ncf9gh 27391c1) ----
STAGE_1B = {
    "dpm30": dpm(10, 20), "dpm20": dpm(7, 13), "dpm16": dpm(5, 11), "dpm12": dpm(4, 8), "dpm8": dpm(3, 5),
    "dpm16_s10k": dpm(5, 11, sigma_max=10000.0),
    # hand-designed lists come after the 1a read; one placeholder exercises the code path
    "cu12_a": dict(PW30, schedule_type="custom", num_steps=12, sigma_max=100000.0, sigma_min=0.03,
                   sigmas=[100000.0, 300.0, 30.0, 12.0, 6.0, 3.0, 1.5, 0.75, 0.35, 0.15, 0.07, 0.03],
                   num_steps_high=3, num_steps_low=9),
}


def exp_list(n, sigma_max=100000.0, sigma_min=0.03):
    """N sigmas spaced exponentially from sigma_max to sigma_min; the custom scheduler appends the
    terminal zero. The fork's ExponentialScheduler lacks that zero (stage-1a exp16/exp12 were
    cancelled for it, note section 9), so the one-segment arms run as explicit lists instead."""
    import math
    r = math.log(sigma_min / sigma_max) / (n - 1)
    return [round(sigma_max * math.exp(r * i), 6) for i in range(n)]


def custom(sigmas, **kw):
    # num_steps_high/low and sigma_transition are inert for the custom scheduler; zeroed to say so.
    d = dict(PW30, schedule_type="custom", num_steps=len(sigmas), sigma_max=sigmas[0],
             sigma_min=sigmas[-1], sigmas=list(sigmas), num_steps_high=0, num_steps_low=0)
    d.update(kw)
    return d


# one-segment exponential arms of E5, re-registered as explicit lists (stage 1b, patched runtime)
STAGE_1B["exp16_x"] = custom(exp_list(16))
STAGE_1B["exp12_x"] = custom(exp_list(12))


def c0(n_high, n_low, **kw):
    """Heun piecewise with churn OFF (S_churn 0; S_noise inert, kept at 1.05 like the campaign's RW50k_c0).
    Stage-1a read of 2026-09-30 (lead-matched): b0's S_noise 1.05 gives 1.27x the truth's fine variance on
    10u and c0_30 gives 1.00 with 3 % lower nMSE at equal fair CRPS; churn also makes the draws depend on
    the step count, so churn-off arms are the only ones paired across the whole ensemble."""
    return pw(n_high, n_low, S_churn=0.0, **kw)


# stage 1b-Heun: the churn-off ladder on the existing runtime (owner's reference decision of 2026-09-30
# pending; this is the recommended option). pw20_c0, pw16_c0 and pw12_c0 already exist from stage 1a.
STAGE_1B_C0 = {
    "c0_pw24": c0(8, 16),
    "c0_pw16_s10k": c0(5, 11, sigma_max=10000.0),
    "c0_pw16_s1k": c0(5, 11, sigma_max=1000.0),
    "c0_pw12_s1k": c0(4, 8, sigma_max=1000.0),
    "c0_pw12_s100": c0(4, 8, sigma_max=100.0),
    "c0_pw16_h3l13": c0(3, 13),
    # added 13:10 UTC after the bundle read: pw16_h8l8 (8+8) beat pw16 (5+11) on every metric with churn on
    "c0_pw16_h8l8": c0(8, 8),
    "c0_pw12_s1k_h6l6": c0(6, 6, sigma_max=1000.0),
    # replicates of the campaign's c0_30 at other base seeds, for the noise band of the churn-off reference
    # (submit with ANEMOI_BASE_SEED=757 / 758 exported, i.e. bases 757000 / 758000, like pw30_s757/s758)
    "c0_pw30_s757": c0(10, 20),
    "c0_pw30_s758": c0(10, 20),
}
# noise-multiplier-1.0 ladder, kept for reference: n100_30 equals c0_30 on texture, nMSE and CRPS on the
# box (0.999 / 0.3210 / 0.8503 vs 0.998 / 0.3209 / 0.8499), so the cheaper deterministic churn-off arms
# carry stage 1b; run these only if the owner prefers churn on.
STAGE_1B_N100 = {
    "n100_pw24": pw(8, 16, S_noise=1.0), "n100_pw20": pw(7, 13, S_noise=1.0),
    "n100_pw16": pw(5, 11, S_noise=1.0), "n100_pw12": pw(4, 8, S_noise=1.0),
    "n100_pw30_s757": pw(10, 20, S_noise=1.0),
}


DONOR_1A = {"pw20": pw(7, 13), "pw12": pw(4, 8), "pw8": pw(3, 5), "pw16_s10k": pw(5, 11, sigma_max=10000.0)}

BASES = {
    "fs": ("_tc_hres_ULR_ctrl", "RW50k 922d1697 step 50,000, hres-lead runtime on AC"),
    "fd": ("tc_o320_o1280_ft400k_gmass_ctrl", "donor 12dcefea step 400,000, pristine runtime on AC"),
}


def yaml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(yaml_scalar(x) for x in v) + "]"
    return repr(float(v)) if isinstance(v, float) else str(v)


def lane_text(prefix, arm, block, base, note):
    lines = [
        f"# few-step sampler screen, stage 1 (2026-09-30), arm {prefix}_{arm}: {note}.",
        f"# Written by scripts/fewstep_sampler_20260930/make_lanes.py; {calls(block)} denoiser calls per member.",
        "# Only sampler keys change; sampler_overrides merges key by key onto the base lane's block.",
        f"base: {base}",
        "",
        "predict:",
        "  sampler_overrides:",
    ]
    for k, v in block.items():
        lines.append(f"    {k}: {yaml_scalar(v)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the lanes directory (eval/config/lanes of the worktree)")
    ap.add_argument("--stage", default="1a", choices=["1a", "1b", "1b-heun", "1b-n100", "all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = []
    if args.stage in ("1a", "all"):
        plan += [("fs", a, b) for a, b in STAGE_1A.items()] + [("fd", a, b) for a, b in DONOR_1A.items()]
    if args.stage in ("1b", "all"):
        plan += [("fs", a, b) for a, b in STAGE_1B.items()]
    if args.stage in ("1b", "1b-heun", "all"):
        plan += [("fs", a, b) for a, b in STAGE_1B_C0.items()]
    if args.stage in ("1b-n100", "all"):
        plan += [("fs", a, b) for a, b in STAGE_1B_N100.items()]
    print(f"{'lane':34s} {'sampler':9s} {'steps':>5s} {'calls':>5s}  block")
    for prefix, arm, block in plan:
        base, note = BASES[prefix]
        name = f"_tc_hres_ULR_{prefix}_{arm}" if prefix == "fs" else f"tc_o320_o1280_fd_{arm}"
        path = out / f"{name}.yaml"
        if path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {path} (use --force)")
        path.write_text(lane_text(prefix, arm, block, base, note))
        short = {k: v for k, v in block.items() if PW30.get(k) != v}
        print(f"{name:34s} {block['sampler']:9s} {block['num_steps']:5d} {calls(block):5d}  {json.dumps(short)}")


if __name__ == "__main__":
    main()
