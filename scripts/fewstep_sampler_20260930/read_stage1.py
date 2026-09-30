#!/usr/bin/env python3
"""Paired read of the few-step sampler screen (stage 1) against the 30-step baseline.

Inputs are `eval.cli evaluate` output folders (each holds evaluators/<name>/metrics.json and,
for `probabilistic`, evaluators/probabilistic/scores_by_lead.csv). The baseline is the RW50k
campaign's RW50k_b0 scored on leads 24 and 120; the seed replicates (base seeds 757 and 758)
set the noise band per metric: max |seed - baseline| x 1.5, never narrower than the floors of
the pre-registration (fair CRPS 1 %, nMSE 1 %, spread 5 %, texture ratio 0.03, grain index 0.05,
retention 0.02). An arm is OUTSIDE on an evaluator when at least half of its judged metrics are
outside the band or any one is outside by more than twice the band. See section 6 of
epics/fast-generative-downscaling/in-progress/20260930_T1_fewstep_sampler_screen_o320_o1280.md.

Usage:
  python3 read_stage1.py --baseline <dir> --seed <dir> --seed <dir> \
      --arm pw20=<dir> --arm pw12=<dir> ... [--calls pw20=39 ...] --out <folder>

Writes <out>/stage1_read.json and <out>/stage1_read.md. Nothing here is a verdict: the
shortlist rule of the note is applied mechanically and every row it selects still goes to the
result-skeptic before it is quoted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# Metrics read per evaluator: (regex on the metric name, kind). "ratio" metrics are compared as a
# difference in the ratio; "relative" metrics as a percent difference to the baseline.
HEADLINE = [
    ("texture", re.compile(r"^tex_(10u|10v|2t)_(all|sea|land|global)_fine_var_ratio$"), "ratio"),
    ("texture", re.compile(r"^tex_(10u|10v|2t)_all_fine_(lag1_zonal|nn_corr)_grain$"), "ratio"),
    ("wind_extremes", re.compile(r"^wx_.*_retention\d+_delta$"), "ratio"),
    ("wind_extremes", re.compile(r"^wx_.*_peak_model$"), "relative"),
    ("probabilistic", re.compile(r"^probabilistic_(2t|10ff|msl|10u|10v)_.*_fcrps_mean$"), "relative"),
    ("probabilistic", re.compile(r"^probabilistic_(2t|10ff|msl|10u|10v)_.*_spread_mean$"), "relative"),
    ("probabilistic", re.compile(r"^probabilistic_(2t|10ff|msl|10u|10v)_.*_rmse_ens_mean_mean$"), "relative"),
    ("surface", re.compile(r"^surface_weighted_nmse$"), "relative"),
    ("surface", re.compile(r"^surface_.*_nmse$"), "relative"),
    ("shape", re.compile(r"^shape_elongated_fraction_.*"), "ratio"),
    ("tc", re.compile(r"^tc_(idalia|franklin)_(mslp_min|wind_max|mslp_p01|wind_p9999)$"), "relative"),
]
FLOORS = {"fcrps": 0.01, "rmse_ens_mean": 0.01, "nmse": 0.01, "spread": 0.05,
          "fine_var_ratio": 0.03, "_grain": 0.05, "retention": 0.02}
JUDGED = ("texture", "wind_extremes", "probabilistic", "surface")  # tc and shape are recorded, not judged


def load_metrics(run_dir: Path) -> dict[str, float]:
    out = {}
    for mj in sorted(run_dir.glob("evaluators/*/metrics.json")):
        for rec in json.loads(mj.read_text()):
            v = rec.get("value")
            if isinstance(v, (int, float)) and math.isfinite(v):
                out[rec["metric"]] = float(v)
    return out


def load_cases(run_dir: Path) -> dict[tuple, float]:
    """Per (date, step, weather_state, domain, metric) values of the probabilistic evaluator."""
    p = run_dir / "evaluators" / "probabilistic" / "scores_by_lead.csv"
    cases = {}
    if p.exists():
        with p.open() as fh:
            for row in csv.DictReader(fh):
                try:
                    cases[(row["date"], row["step"], row["weather_state"], row["domain"], row["metric"])] = float(row["value"])
                except (KeyError, ValueError):
                    continue
    return cases


def floor_for(metric: str) -> float:
    for key, val in FLOORS.items():
        if key in metric:
            return val
    return 0.0


def diff(kind: str, a: float, b: float) -> float | None:
    if kind == "ratio":
        return a - b
    return (a - b) / b if b else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--seed", action="append", default=[], help="a seed-replicate run dir of the baseline")
    ap.add_argument("--arm", action="append", default=[], help="name=eval_stage1 dir")
    ap.add_argument("--calls", action="append", default=[], help="name=denoiser calls per member")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load_metrics(Path(args.baseline))
    base_cases = load_cases(Path(args.baseline))
    seeds = [load_metrics(Path(s)) for s in args.seed]
    calls = {k: int(v) for k, v in (c.split("=", 1) for c in args.calls)}
    arms = {k: Path(v) for k, v in (a.split("=", 1) for a in args.arm)}

    # noise band per metric (section 6a of the note: the replicates are unpaired 100-draw
    # runs of the baseline, so their deviation is the null for an unpaired arm difference)
    band = {}
    band_detail = {}
    for metric, bval in base.items():
        kind = next((k for _, rx, k in HEADLINE if rx.match(metric)), None)
        if kind is None:
            continue
        devs = [abs(diff(kind, s[metric], bval) or 0.0) for s in seeds if metric in s]
        band[metric] = max([1.5 * d for d in devs] + [floor_for(metric)])
        band_detail[metric] = {
            "baseline": bval, "seed_deviations": devs, "floor": floor_for(metric),
            "band": band[metric], "limited_by": "floor" if band[metric] == floor_for(metric) else "seeds",
        }

    result = {"baseline_metrics": len(base), "seed_replicates": len(seeds),
              "noise_band": band_detail, "arms": {}}
    for name, run_dir in arms.items():
        m = load_metrics(run_dir)
        cases = load_cases(run_dir)
        rows = []
        seen = set()
        flagged = defaultdict(list)
        judged_count = defaultdict(int)
        far = defaultdict(list)
        for evaluator, rx, kind in HEADLINE:
            for metric in sorted(base):
                if not rx.match(metric) or metric not in m or metric in seen:
                    continue
                seen.add(metric)
                d = diff(kind, m[metric], base[metric])
                b = band.get(metric, 0.0)
                inside = d is not None and abs(d) <= b
                rows.append({"evaluator": evaluator, "metric": metric, "arm": m[metric], "baseline": base[metric],
                             "diff": d, "band": b, "inside": inside})
                if evaluator in JUDGED:
                    judged_count[evaluator] += 1
                    if not inside:
                        flagged[evaluator].append(metric)
                        if d is not None and abs(d) > 2 * b:
                            far[evaluator].append(metric)
        outside = {ev: flagged[ev] for ev in judged_count
                   if far[ev] or len(flagged[ev]) * 2 >= judged_count[ev]}
        # per-case sign counts for the probabilistic metrics (10 cases: 5 dates x 2 leads)
        signs = {}
        for key, bval in base_cases.items():
            if key not in cases or key[4] not in ("fcrps", "spread", "rmse_ens_mean"):
                continue
            k2 = key[2:]  # (weather_state, domain, metric)
            signs.setdefault(k2, {"worse": 0, "better": 0, "n": 0})
            signs[k2]["n"] += 1
            worse = cases[key] > bval if key[4] != "spread" else cases[key] < bval
            signs[k2]["worse" if worse else "better"] += 1
        result["arms"][name] = {
            "calls": calls.get(name), "metrics_read": len(rows), "rows": rows,
            "flagged_metrics": dict(flagged), "judged_metrics": dict(judged_count),
            "outside_noise": outside, "inside_on_all_judged": not outside,
            "case_signs": {"/".join(k): v for k, v in signs.items()},
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage1_read.json").write_text(json.dumps(result, indent=2))
    lines = ["| arm | calls | inside the noise on all judged metrics | metrics outside (evaluator: count) |", "|---|---:|---|---|"]
    for name, r in sorted(result["arms"].items(), key=lambda kv: (kv[1]["calls"] or 999, kv[0])):
        outs = ", ".join(f"{k}: {len(v)}" for k, v in r["outside_noise"].items()) or "none"
        lines.append(f"| {name} | {r['calls'] or '?'} | {'yes' if r['inside_on_all_judged'] else 'no'} | {outs} |")
    lines.append("")
    lines.append("Measured noise band per headline metric (relative for ratios, absolute for correlations and deltas):")
    lines.append("")
    lines.append("| metric | baseline | seed deviations | floor | band | limited by |")
    lines.append("|---|---:|---|---:|---:|---|")
    for metric, d in sorted(band_detail.items()):
        devs = ", ".join(f"{x:.4f}" for x in d["seed_deviations"]) or "none"
        lines.append(f"| {metric} | {d['baseline']:.4g} | {devs} | {d['floor']:.3g} | {d['band']:.4f} | {d['limited_by']} |")
    lines.append("")
    seed_limited = [m for m, d in band_detail.items() if d["limited_by"] == "seeds"]
    lines.append(f"{len(seed_limited)} of {len(band_detail)} metrics have a band wider than the floor; on those the "
                 "100-draw screen resolves less than the pre-registered threshold and the verdict is 'not resolved at stage 1'.")
    lines.append("")
    lines.append(f"Noise band from {len(seeds)} seed replicate(s), 1.5 x the largest seed deviation, floors fair CRPS 1 %, nMSE 1 %, spread 5 %. "
                 "Spread convention: eval.cli probabilistic, domain mean of the pointwise member standard deviation (ddof 1). "
                 "Cyclone extremes and shape are recorded, not judged, at stage 1.")
    (out / "stage1_read.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
