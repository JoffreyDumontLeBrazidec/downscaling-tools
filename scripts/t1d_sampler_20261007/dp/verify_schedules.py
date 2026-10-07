"""T1d batch 2: verification draws. Does the summed one-step cost predict the real error of a schedule?

Each candidate schedule is RUN with the trajectory tool on the same 16 bundles and seeds as the dense reference,
one draw per (schedule, bundle, seed), and its final state is compared with the dense run's final state (same
seed, so the same initial unit noise: the tool calls torch.manual_seed(seed) right before the model's sample(),
whose first random draw is y_init = randn(shape) * sigmas[0]; the shape is the cut-grid shape for every schedule,
so the unit noise is identical whatever sigma_max is. `analyze` CHECKS this on the saved call-0 inputs).
The error is split per variable and band with the same splitter and normaliser as cost_matrix.py (band variance of
the dense run's final state), giving C_L2_run and C_band_run per draw; the table reports their means over the
draws, the predicted summed path cost (dp_schedules.json, set "all"), and the ratio run / predicted.

  make     candidate schedule file from dp_schedules.json: the DP schedules fitted on all draws for both fixed
           costs and K = 8, 10, 12, 16 (identical ones merged), c0_30, c0_pw16_s1k and log-uniform 16, each as a
           `schedule_type: custom` block on its NOMINAL levels (c0_30 and pw16 exactly as the fork's piecewise
           scheduler builds them)
  run      (GPU) one bundle, all seeds, every schedule of the file; the model is loaded once and reused
  analyze  (CPU) the table

  python -m scripts.t1d_sampler_20261007.dp.verify_schedules make --dp-json <root>/dp/dp_schedules.json --out <root>/verify/schedules_verify.json
  python -m scripts.t1d_sampler_20261007.dp.verify_schedules run --schedules ... --checkpoint ... --bundle-dir ... \
      --date 20230826 --step 024 --seeds 1000 1001 1002 1003 --window ... --scope ... --sampler-params ... --out-root <root>/verify
  python -m scripts.t1d_sampler_20261007.dp.verify_schedules analyze --verify-root <root>/verify --dense-root <root> \
      --dp-json <root>/dp/dp_schedules.json --out <root>/verify/verify_table
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (COST_VARS, W_COARSE, W_FINE, custom_scheduler_json, reference_levels,  # type: ignore
                        save_json, splitter_for)
else:
    from .common import (COST_VARS, W_COARSE, W_FINE, custom_scheduler_json, reference_levels, save_json,
                         splitter_for)

LOG = logging.getLogger("t1d.verify")
PRIMARY = ("C_L2_heun_ref", "C_band_heun_ref")
DEFAULT_K = (8, 10, 12, 16)


# ------------------------------------------------------------------ make
def make(dp_json, out, budgets=DEFAULT_K, fit="all"):
    res = json.load(open(dp_json))
    sch = res["schedules"]
    cands, seen = {}, {}
    for K in budgets:
        for c in PRIMARY:
            name = f"dp_{c}_K{K}_fit{fit}"
            if name not in sch:
                continue
            key = tuple(sch[name]["levels"])
            if key in seen:                                  # same levels from both costs: run once
                cands[seen[key]]["also"].append(name)
                continue
            seen[key] = name
            cands[name] = {"noise_scheduler": sch[name]["noise_scheduler"], "dp_name": name, "also": [],
                           "calls": sch[name]["calls"]}
    for ref in ("c0_30", "c0_pw16_s1k"):
        lv = reference_levels(ref)
        cands[ref] = {"noise_scheduler": custom_scheduler_json(lv), "dp_name": ref, "also": [], "calls": 2 * len(lv) - 1}
    lv = np.exp(np.linspace(math.log(1e5), math.log(0.03), 16))
    cands["logu_16"] = {"noise_scheduler": custom_scheduler_json(lv), "dp_name": "logu_16", "also": [], "calls": 31}
    total = sum(v["calls"] for v in cands.values())
    meta = {"candidates": cands, "calls_per_draw_all": total,
            "gpu_h_16_draws_at_1.1s": round(16 * total * 1.1 / 3600, 2),
            "gpu_h_per_schedule": {k: round(16 * v["calls"] * 1.1 / 3600, 3) for k, v in cands.items()}}
    save_json(out, meta)
    print(f"{len(cands)} candidate schedules, {total} calls per draw, about {meta['gpu_h_16_draws_at_1.1s']} GPU-h "
          f"for 16 draws (A100 box, 1.1 s/call) plus one model load per bundle job -> {out}")
    return meta


# ------------------------------------------------------------------ run (GPU)
def run(a):
    import interp.tools.trajectory as T
    cache = {}
    orig = T.load_model

    def cached_load(ckpt, **kw):                           # load once, reuse for every schedule
        key = (str(ckpt), tuple(sorted(kw.items())))
        if key not in cache:
            cache[key] = orig(ckpt, **kw)
        return cache[key]

    T.load_model = cached_load
    cands = json.load(open(a.schedules))["candidates"]
    labels = a.labels or list(cands)
    for label in labels:
        blk = cands[label]["noise_scheduler"]
        out = Path(a.out_root) / label / f"d{a.date}_l{a.step}"
        argv = ["--checkpoint", a.checkpoint, "--output-dir", str(out),
                "--bundle-dir", a.bundle_dir, "--dates", a.date, "--members", a.member, "--steps", a.step,
                "--mode", "trajectory", "--save-trajectory-states", "--num-steps", str(len(blk["sigmas"])),
                "--seeds", *[str(s) for s in a.seeds], "--eye-radius-km", str(a.radius_km),
                "--auto-window", a.window, "--precision", "fp32", "--ceiling-sigmas", "10",
                "--noise-scheduler-json", json.dumps(blk), "--sampler-params-json", a.sampler_params,
                "--local-scope-json", a.scope]
        LOG.info("schedule %s: %d levels, %d calls per draw -> %s", label, len(blk["sigmas"]), 2 * len(blk["sigmas"]) - 1, out)
        T.main(argv)


# ------------------------------------------------------------------ analyze (CPU)
def _load_light(p):
    z = np.load(p)
    d = {"final": z["final"].astype(np.float64), "u0": z["x_in"][0].astype(np.float64) / float(z["sigma"][0]),
         "vars": [str(v) for v in z["vars"]], "lat": z["lat"].astype(np.float64), "lon": z["lon"].astype(np.float64),
         "meta": {k[5:]: z[k].item() for k in z.files if k.startswith("meta_")}, "n_calls": int(z["x_in"].shape[0])}
    z.close()
    return d


def analyze(a):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dp = json.load(open(a.dp_json))["schedules"]
    cands = json.load(open(Path(a.verify_root) / "schedules_verify.json"))["candidates"]
    rows, per_draw = [], {}
    splitters = {}
    for label, c in cands.items():
        files = sorted((Path(a.verify_root) / label).glob("d2023*_l*/trajectory_states_s*.npz"))
        l2s, bands, noise_ok, n_expected = [], [], [], []
        for f in files:
            dense_p = Path(a.dense_root) / f.parent.name / f.name
            if not dense_p.exists():
                LOG.warning("%s: no dense counterpart %s", f, dense_p)
                continue
            v, r = _load_light(f), _load_light(dense_p)
            if not (np.allclose(v["lat"], r["lat"]) and np.allclose(v["lon"], r["lon"])):
                raise SystemExit(f"{f}: box cells differ from the dense run's")
            noise_ok.append(float(np.max(np.abs(v["u0"] - r["u0"]))))
            n_expected.append(v["n_calls"] == c["calls"])
            key = f.parent.name
            if key not in splitters:
                splitters[key] = splitter_for(SimpleNamespace(lat=r["lat"], lon=r["lon"], meta=r["meta"]))
            spl = splitters[key]
            l2, band, per = [], [], {}
            for name in COST_VARS:
                i, j = v["vars"].index(name), r["vars"].index(name)
                V = spl.variance(r["final"][j])
                E = spl.energies(spl.spectrum(v["final"][i] - r["final"][j]))
                l2.append((E["fine100"] + E["coarse100"]) / (V["fine100"] + V["coarse100"]))
                band.append(W_FINE * E["fine100"] / V["fine100"] + W_COARSE * E["coarse100"] / V["coarse100"])
                per.update({f"{name}_{b}": float(E[b] / V[b]) for b in E})
            l2s.append(float(np.mean(l2)))
            bands.append(float(np.mean(band)))
            per_draw.setdefault(label, {})[f"{key}/{f.stem}"] = {"C_L2_run": l2s[-1], "C_band_run": bands[-1],
                                                                 "max_unit_noise_diff": noise_ok[-1], **per}
        d = dp.get(c["dp_name"], {})
        pred = d.get("costs", {}).get("all", {})
        row = {"schedule": label, "same_as": c["also"], "calls": c["calls"], "n_draws": len(l2s),
               "C_L2_run": float(np.mean(l2s)) if l2s else float("nan"),
               "C_band_run": float(np.mean(bands)) if bands else float("nan"),
               "C_L2_run_range": [min(l2s), max(l2s)] if l2s else None,
               "C_band_run_range": [min(bands), max(bands)] if bands else None,
               "pred_L2": pred.get("C_L2_heun_ref", float("nan")), "pred_band": pred.get("C_band_heun_ref", float("nan")),
               "noise_identical": bool(noise_ok) and max(noise_ok) < 1e-4, "max_unit_noise_diff": max(noise_ok) if noise_ok else None,
               "calls_as_expected": all(n_expected) if n_expected else None}
        row["ratio_L2"] = row["C_L2_run"] / row["pred_L2"] if row["pred_L2"] else float("nan")
        row["ratio_band"] = row["C_band_run"] / row["pred_band"] if row["pred_band"] else float("nan")
        rows.append(row)
    ok = [r for r in rows if r["n_draws"] and np.isfinite(r["pred_band"]) and r["pred_band"] > 0 and r["C_band_run"] > 0]
    corr = {}
    if len(ok) >= 3:
        for k, p in (("C_L2_run", "pred_L2"), ("C_band_run", "pred_band")):
            corr[k] = float(np.corrcoef(np.log([r[p] for r in ok]), np.log([r[k] for r in ok]))[0, 1])
    save_json(str(a.out) + ".json", {"rows": rows, "loglog_corr_run_vs_pred": corr, "per_draw": per_draw,
                                     "note": "run error = final state vs the dense 240-level reference, same seed; "
                                             "predicted = summed heun_ref path cost, set all (pw16 incl. start truncation)"})
    L = ["| schedule | calls | draws | C_L2 run | C_band run | predicted L2 | predicted band | ratio L2 | ratio band | same noise |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: r["calls"]):
        L.append(f"| {r['schedule']}{' (= ' + ', '.join(r['same_as']) + ')' if r['same_as'] else ''} | {r['calls']} | "
                 f"{r['n_draws']} | {r['C_L2_run']:.4g} | {r['C_band_run']:.4g} | {r['pred_L2']:.4g} | {r['pred_band']:.4g} | "
                 f"{r['ratio_L2']:.3g} | {r['ratio_band']:.3g} | {'yes' if r['noise_identical'] else 'NO'} |")
    if corr:
        L.append("")
        L.append(f"log-log correlation, run vs predicted: C_L2 {corr.get('C_L2_run', float('nan')):.3f}, "
                 f"C_band {corr.get('C_band_run', float('nan')):.3f}")
    Path(str(a.out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if any(r["n_draws"] and not r["noise_identical"] for r in rows):
        raise SystemExit("initial unit noise differs between a schedule and the dense run: the pairing assumption fails")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--dp-json", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_K))
    m.add_argument("--fit", default="all")
    r = sub.add_parser("run")
    r.add_argument("--schedules", required=True)
    r.add_argument("--labels", nargs="+", default=None)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--bundle-dir", required=True)
    r.add_argument("--date", required=True)
    r.add_argument("--step", required=True)
    r.add_argument("--member", default="01")
    r.add_argument("--seeds", nargs="+", type=int, required=True)
    r.add_argument("--window", required=True)
    r.add_argument("--scope", required=True)
    r.add_argument("--sampler-params", required=True)
    r.add_argument("--radius-km", type=float, default=500.0)
    r.add_argument("--out-root", required=True)
    n = sub.add_parser("analyze")
    n.add_argument("--verify-root", required=True)
    n.add_argument("--dense-root", required=True)
    n.add_argument("--dp-json", required=True)
    n.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(a.dp_json, a.out, a.budgets, a.fit)
    elif a.cmd == "run":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        run(a)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
