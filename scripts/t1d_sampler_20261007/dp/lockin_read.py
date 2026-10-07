"""Per-variable anomaly lock-in sigma50 from the trajectory tool's --lockin output (T1d).

Definition (findings-20260805 point 2; p1/analyze_lockin.py): the anomaly pattern
correlation of each call's x-hat-0 with the draw's own FINAL sample, both taken relative
to the FIRST call's x-hat-0 (sigma ~ sigma_max, about the conditional mean)
(`corr_final_anom` in trajectory.json). Seed-mean curve, averaged per unique sigma,
read from high to low sigma; sigma50 = the highest sigma from which the curve stays at or
above 0.5 for every lower sigma. Also reported: the same rule per draw, and the
log-interpolated 0.5 crossing.

Sources: trajectory.json files (default), or --from-states with trajectory_states_s*.npz
(the same correlations from D in normalised units: the anomaly removes the input and the
per-channel affine scaling does not change a correlation; exact for 10u, 10v, 2t, msl).

  python -m scripts.t1d_sampler_20261007.dp.lockin_read \
      --inputs '/path/diag/*/trajectory.json' --out /path/diag/lockin/lockin_sigma50.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import warnings
from pathlib import Path

import numpy as np

KEY = "corr_final_anom"


def sigma50(sigmas, curve, thr=0.5):
    """sigmas, curve per call. Returns (rule sigma50, interpolated crossing, uniq, per_sigma)."""
    sigmas = np.asarray(sigmas, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    uniq = np.unique(sigmas)[::-1]
    ps = np.array([np.nanmean(curve[sigmas == s]) for s in uniq])
    ok = ps >= thr
    lock = None
    for i in range(len(uniq)):
        if ok[i:].all():
            lock = float(uniq[i])
            break
    cross = None
    if lock is not None:
        i = int(np.flatnonzero(uniq == lock)[0])
        if i > 0 and np.isfinite(ps[i - 1]) and ps[i] != ps[i - 1]:
            t = (thr - ps[i - 1]) / (ps[i] - ps[i - 1])
            cross = float(math.exp(math.log(uniq[i - 1]) + t * (math.log(uniq[i]) - math.log(uniq[i - 1]))))
        else:
            cross = lock
    return lock, cross, uniq, ps


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def curves_from_json(path):
    d = json.load(open(path))
    out = []
    for t in d.get("trajectories", []):
        lk = t.get("lockin")
        if not lk:
            continue
        out.append({"draw": f"{Path(path).parent.name}_s{t['seed']}", "sigmas": lk["sigmas"],
                    "vars": {v: lk["vars"][v][KEY] for v in lk["vars"]}})
    return out


def curves_from_states(path):
    z = np.load(path)
    names = [str(v) for v in z["vars"]]
    D, fin = z["D"].astype(np.float64), z["final"].astype(np.float64)
    ref = D[0]
    cur = {n: [_corr(D[k, v] - ref[v], fin[v] - ref[v]) for k in range(D.shape[0])] for v, n in enumerate(names)}
    seed = int(z["meta_seed"]) if "meta_seed" in z.files else -1
    return [{"draw": f"{Path(path).parent.name}_s{seed}", "sigmas": z["sigma"].tolist(), "vars": cur}]


def main(argv=None):
    warnings.filterwarnings("ignore", message="Mean of empty slice")   # call 0: zero anomaly
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--from-states", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    draws = []
    for p in paths:
        draws.extend(curves_from_states(p) if args.from_states else curves_from_json(p))
    if not draws:
        raise SystemExit("no lock-in curves found (was the run made with --lockin?)")
    sig = np.asarray(draws[0]["sigmas"], dtype=np.float64)
    for d in draws:
        if len(d["sigmas"]) != len(sig) or np.max(np.abs(np.log(np.asarray(d["sigmas"]) / sig))) > 1e-5:
            raise SystemExit(f"{d['draw']}: call sigmas differ from the first draw's")
    names = list(draws[0]["vars"])
    res = {"definition": "anomaly lock-in sigma50 (findings-20260805 pt 2): seed-mean corr_final_anom per "
                         "unique sigma; highest sigma from which it stays >= 0.5", "n_draws": len(draws),
           "draws": [d["draw"] for d in draws], "vars": {}}
    for v in names:
        mean_curve = np.nanmean([np.asarray(d["vars"][v], dtype=np.float64) for d in draws], axis=0)
        lock, cross, uniq, ps = sigma50(sig, mean_curve)
        per = {}
        for d in draws:
            l1, c1, _, _ = sigma50(d["sigmas"], d["vars"][v])
            per[d["draw"]] = {"sigma50": l1, "crossing": c1}
        vals = [x["sigma50"] for x in per.values() if x["sigma50"] is not None]
        res["vars"][v] = {"sigma50": lock, "crossing_interp": cross,
                          "per_draw": per,
                          "per_draw_median": float(np.median(vals)) if vals else None,
                          "per_draw_range": [float(min(vals)), float(max(vals))] if vals else None,
                          "curve_sigma": uniq.tolist(), "curve_mean": ps.tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    for v in names:
        r = res["vars"][v]
        print(f"{v:>4}: sigma50 {r['sigma50']}  (interp {r['crossing_interp']}; per-draw median "
              f"{r['per_draw_median']}, range {r['per_draw_range']})")


if __name__ == "__main__":
    main()
