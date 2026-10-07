"""GITS-style dynamic-programming schedules from the dense cost matrices (T1d, CPU).

A schedule with K sampler steps has K positive levels, from the first dense level
(sigma 1e5) to the last (0.03), then the terminal zero that the fork's custom scheduler
appends: K - 1 intervals between positive levels, and 2K - 1 denoiser calls with Heun
(two per step, one on the last step 0.03 -> 0). The last step is common to every
schedule and is not in the cost. The DP picks the K - 2 interior levels on the dense grid
that minimise the summed one-step cost C[n_m, n_{m+1}] (GITS 2405.11326; optimal stepsize
distillation 2503.21774 uses the same recursion).

Budgets K = 8, 10, 12, 16 (the plan), and 20, 29, 30 for comparison (c0_30 has 30 steps,
29 intervals). Fits: on the 20230826 draws, on the 20230828 draws, and on all; each fitted
schedule is evaluated in-sample and on the other date (hold-out). The existing schedules
c0_30 and c0_pw16_s1k (sigma_max 1e3; charged with its start-truncation term) and a
log-uniform K-step baseline are mapped to the nearest dense levels and costed the same way.

Outputs (in --out-dir): dp_schedules.json (every schedule: sigmas, ln-steps, calls,
`noise_scheduler` block for schedule_type custom, costs on every set), c0_30_running_cost.json,
dp_summary.md, and schedules/<name>.json (just the noise-scheduler block, ready for
--noise-scheduler-json or a lane's sampler block).

  python -m scripts.t1d_sampler_20261007.dp.dp_schedule --cost-dir /path/diag/cost --out-dir /path/diag/dp
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (custom_scheduler_json, dense_levels, nearest_level_indices,  # type: ignore
                        reference_levels, save_json)
else:
    from .common import custom_scheduler_json, dense_levels, nearest_level_indices, reference_levels, save_json

LOG = logging.getLogger("t1d.dp_schedule")
BUDGETS = (8, 10, 12, 16, 20, 29, 30)
COSTS = ("C_L2_heun_ref", "C_band_heun_ref", "C_L2_euler", "C_band_euler")
PRIMARY = ("C_L2_heun_ref", "C_band_heun_ref")


def dp_path(C: np.ndarray, K: int, start: int = 0, end: int | None = None) -> tuple[list[int], float]:
    """Minimum-cost path start -> end through exactly K levels (K - 1 intervals) on the
    strictly upper-triangular cost C (NaN or inf = forbidden). O(K N^2)."""
    N = C.shape[0]
    end = N - 1 if end is None else end
    if K < 2 or K > end - start + 1:
        raise ValueError(f"K={K} impossible between levels {start} and {end}")
    W = np.where(np.isfinite(C), C, np.inf)
    best = np.full(N, np.inf)
    best[start] = 0.0
    back = np.full((K, N), -1, dtype=np.int64)
    for m in range(1, K):
        tot = best[:, None] + W                         # (from i, to j)
        arg = np.argmin(tot, axis=0)
        new = tot[arg, np.arange(N)]
        back[m] = arg
        best = new
    path = [end]
    for m in range(K - 1, 0, -1):
        path.append(int(back[m, path[-1]]))
    path = path[::-1]
    if path[0] != start or not np.isfinite(best[end]):
        raise RuntimeError("DP found no finite path")
    return path, float(best[end])


def path_cost(C, path) -> float:
    return float(sum(C[a, b] for a, b in zip(path[:-1], path[1:])))


def interval_costs(C, path) -> list[float]:
    return [float(C[a, b]) for a, b in zip(path[:-1], path[1:])]


def load_sets(cost_dir: Path) -> dict:
    sets = {}
    for p in sorted(cost_dir.glob("cost_mean_*.npz")):
        z = np.load(p)
        sets[p.stem[len("cost_mean_"):]] = {k: z[k] for k in z.files}
    if "all" not in sets:
        raise SystemExit(f"no cost_mean_all.npz in {cost_dir}")
    return sets


def snap_levels(sig: np.ndarray) -> np.ndarray:
    """Replace the recorded levels (the fp32 sampler's values) by the exact log-uniform
    dense levels 1e5 ... 0.03 when they match to 1e-5 in ln(sigma)."""
    ref = dense_levels(len(sig))
    if np.max(np.abs(np.log(ref / sig))) < 1e-5:
        return ref
    LOG.warning("recorded levels are not the log-uniform 1e5..0.03 grid; using them as recorded")
    return sig


def describe(sig, path):
    s = np.asarray([sig[i] for i in path])
    return {"levels": [int(i) for i in path], "sigmas": [float(x) for x in s],
            "ln_steps": [float(x) for x in np.log(s[:-1] / s[1:])],
            "num_steps": len(path), "calls": 2 * len(path) - 1,
            "noise_scheduler": custom_scheduler_json(s)}


def costs_on_sets(sets, path, extra_trunc=None) -> dict:
    out = {}
    for sname, S in sets.items():
        row = {}
        for c in COSTS:
            if c not in S:
                continue
            v = path_cost(S[c], path)
            if extra_trunc is not None:
                tkey = "trunc_C_L2" if "L2" in c else "trunc_C_band"
                v += float(S[tkey][extra_trunc])
            row[c] = v
        out[sname] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sets = load_sets(Path(args.cost_dir))
    sig = snap_levels(np.asarray(sets["all"]["sigma"], dtype=np.float64))
    N = len(sig)
    fit_sets = [s for s in sets if s != "all"] + ["all"]
    out = Path(args.out_dir)
    result = {"dense_levels": sig.tolist(), "n_levels": N, "sets": {k: int(v.get("n_draws", 0)) for k, v in sets.items()},
              "budget_convention": "K = sampler steps = positive levels incl. 1e5 and 0.03; K-1 intervals; 2K-1 calls",
              "schedules": {}}

    # --- reference schedules mapped to the dense grid ---
    refs = {}
    for name in ("c0_30", "c0_pw16_s1k"):
        lv = reference_levels(name)
        idx = nearest_level_indices(sig, lv)
        map_err = float(np.max(np.abs(np.log(sig[idx] / lv))))
        dup = int(len(idx) - len(np.unique(idx)))
        if dup:
            LOG.warning("%s: %d levels map onto an already used dense level (grid too coarse); "
                        "costed on the unique levels", name, dup)
            idx = np.unique(idx)
        d = describe(sig, list(idx))
        d["nominal_sigmas"] = lv.tolist()
        d["mapping_max_ln_error"] = map_err
        d["duplicate_levels"] = dup
        trunc = int(idx[0]) if idx[0] != 0 else None
        d["start_truncation_charged"] = trunc is not None
        d["costs"] = costs_on_sets(sets, list(idx), extra_trunc=trunc)
        d["noise_scheduler"] = None                     # these run as their own piecewise blocks
        refs[name] = list(idx)
        result["schedules"][name] = d
    for K in args.budgets:
        idx = list(np.unique(nearest_level_indices(sig, np.exp(np.linspace(math.log(sig[0]), math.log(sig[-1]), K)))))
        d = describe(sig, idx)
        d["costs"] = costs_on_sets(sets, idx)
        result["schedules"][f"logu_{K}"] = d

    # --- c0_30 running sum ---
    run = {}
    for sname, S in sets.items():
        rs = {}
        for c in PRIMARY:
            ic = interval_costs(S[c], refs["c0_30"])
            hi = [sig[a] for a in refs["c0_30"][:-1]]
            below = [x for x, h in zip(ic, hi) if h <= 10.0 * (1 + 1e-9)]
            rs[c] = {"interval_upper_sigma": [float(h) for h in hi], "interval_cost": ic,
                     "cumulative": np.cumsum(ic).tolist(), "total": float(np.sum(ic)),
                     "n_intervals_below_10": len(below), "cost_below_10": float(np.sum(below)),
                     "share_below_10": float(np.sum(below) / max(np.sum(ic), 1e-300))}
        run[sname] = rs
    save_json(out / "c0_30_running_cost.json", run)

    # --- DP fits ---
    for fit in fit_sets:
        for c in COSTS:
            if c not in sets[fit]:
                continue
            for K in args.budgets:
                if K > N:
                    continue
                path, val = dp_path(sets[fit][c], K)
                name = f"dp_{c}_K{K}_fit{fit}"
                d = describe(sig, path)
                d.update(fit_set=fit, cost=c, K=K, fit_cost=val, costs=costs_on_sets(sets, path))
                held = [s for s in sets if s not in ("all", fit)] if fit != "all" else []
                d["holdout_sets"] = held
                result["schedules"][name] = d
                if c in PRIMARY:
                    (out / "schedules").mkdir(parents=True, exist_ok=True)
                    with open(out / "schedules" / f"{name}.json", "w") as f:
                        json.dump(d["noise_scheduler"], f)
    # per-variable, per-band breakdown (heun_ref, all draws) of the primary schedules
    S = sets["all"]
    vb = sorted({k[len("n_heun_ref_"):] for k in S if k.startswith("n_heun_ref_")})
    for name, d in result["schedules"].items():
        if name in ("c0_30", "c0_pw16_s1k") or (d.get("cost") in PRIMARY and d.get("fit_set") == "all"):
            d["per_var_band_all"] = {k: path_cost(S["n_heun_ref_" + k], d["levels"]) for k in vb}
    # small export for the docs bundle: the fixed-cost matrices only, float32
    (out / "bundle").mkdir(parents=True, exist_ok=True)
    for sname, Sx in sets.items():
        keep = {k: np.asarray(v, dtype=np.float32) for k, v in Sx.items()
                if k.startswith("C_") or k.startswith("trunc_C_")}
        keep["sigma"] = sig
        np.savez_compressed(out / "bundle" / f"cost_C_{sname}.npz", **keep)
    save_json(out / "dp_schedules.json", result)
    write_summary(out / "dp_summary.md", result, run, sets)
    LOG.info("wrote %s", out)


def write_summary(path, result, run, sets):
    sch = result["schedules"]
    names = [s for s in sets if s != "all"]
    L = ["# T1d DP schedules: summary", "",
         f"Dense levels: {result['n_levels']}; sets: {result['sets']}.", "",
         "Costs are sums of the dimensionless one-step costs along the schedule (heun_ref unless named).",
         "Hold-out = the date the schedule was NOT fitted on.", ""]
    for c in PRIMARY:
        L += [f"## {c}", "", "| schedule | K | calls | fit | in-sample | hold-out | all | c0_30 on same set | max ln-step |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for n, d in sch.items():
            if d.get("cost") != c:
                continue
            fit = d["fit_set"]
            ins = d["costs"][fit][c]
            ho = d["costs"][d["holdout_sets"][0]][c] if d["holdout_sets"] else float("nan")
            ref_set = d["holdout_sets"][0] if d["holdout_sets"] else "all"
            L.append(f"| {n} | {d['K']} | {d['calls']} | {fit} | {ins:.4g} | {ho:.4g} | {d['costs']['all'][c]:.4g} | "
                     f"{sch['c0_30']['costs'][ref_set][c]:.4g} | {max(d['ln_steps']):.2f} |")
        L += ["", "| reference schedule | K | calls | " + " | ".join(names + ["all"]) + " |",
              "|---|---:|---:|" + "---:|" * (len(names) + 1)]
        for n in ("c0_30", "c0_pw16_s1k") + tuple(k for k in sch if k.startswith("logu_")):
            d = sch[n]
            row = " | ".join(f"{d['costs'][s][c]:.4g}" for s in names + ["all"])
            L.append(f"| {n} | {d['num_steps']} | {d['calls']} | {row} |")
        L.append("")
        r = run["all"][c]
        L.append(f"c0_30 running sum ({c}, all draws): total {r['total']:.4g}; {r['n_intervals_below_10']} intervals "
                 f"below sigma 10 carry {100 * r['share_below_10']:.1f} % of it.")
        L.append("")
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
