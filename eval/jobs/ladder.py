"""Ladder eval — cheap per-checkpoint training-progress metrics vs a frozen baseline.

Owner-approved design 2026-07-24 (docs task:
epics/training-diagnostics/metric-skill-gap/in-progress/20260724_ladder_eval_m1.md).

One profile = one lane + one PINNED budget (dates/steps/members [+ seed draws]) + pinned
sampler + evaluator knobs. Every rung AND the baseline are scored with the identical recipe,
so rows are comparable. TC numbers at ladder budgets are TREND INDICATORS, never verdicts
(single-run replica noise ±7.0/4.8 hPa at deepest-of-10 — I1, 2026-07-10).

Subcommands:
  score    --profile P --checkpoint CK --step N [--baseline-label L] [--dry-run]
  sweep    --profile P --run-root DIR [--dry-run]        # score every unscored step_* ckpt
  collect  --profile P --step N --eval-dir DIR [...]     # (called inside the sbatch) -> row
  loss     --profile P --mlflow-dir DIR --run-name NAME  # train/val series into ladder.json
  plot     --profile P [--out PNG]
Storage:  scratch_root/eval/ladder/<card_id>/step_<N>/   (run artifacts)
          perm_root/eval-ladders/<card_id>/{ladder.json, ladder.png}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO / "eval" / "config" / "ladder"
CKPT_STEP_RE = re.compile(r"step[_=]?(\d+)")


# --------------------------------------------------------------------------- profiles
def load_profile(name: str) -> dict:
    # accept either a registered profile name or a direct path to a profile YAML
    p = Path(name) if name.endswith(".yaml") else PROFILE_DIR / f"{name}.yaml"
    if not p.exists():
        raise SystemExit(f"ladder: unknown profile '{name}' (expected {p})")
    name = p.stem
    prof = yaml.safe_load(p.read_text())
    prof["_name"] = name
    for key in ("card_id", "lane", "host", "bundle_dir", "budget"):
        if key not in prof:
            raise SystemExit(f"ladder: profile '{name}' missing key '{key}'")
    return prof


def ladder_paths(prof: dict) -> dict:
    perm = Path(prof.get("perm_root", "/home/ecm5702/perm")) / "eval-ladders" / prof["card_id"]
    scratch = Path(prof.get("scratch_root", "/home/ecm5702/scratch")) / "eval" / "ladder" / prof["card_id"]
    return {"perm": perm, "scratch": scratch, "json": perm / "ladder.json", "png": perm / "ladder.png"}


def load_ladder(prof: dict) -> dict:
    jp = ladder_paths(prof)["json"]
    if jp.exists():
        return json.loads(jp.read_text())
    return {
        "schema_version": "1.0",
        "card_id": prof["card_id"],
        "lane": prof["lane"],
        "profile": prof["_name"],
        "profile_pins": {k: prof.get(k) for k in ("budget", "seed_draws", "spectra", "storm_box")},
        "note": "TC values at ladder budget = trend indicator, NOT a verdict (replica noise ±7/4.8 hPa).",
        "baselines": {},
        "rows": [],
        "loss": {},
    }


def save_ladder(prof: dict, ladder: dict) -> Path:
    jp = ladder_paths(prof)["json"]
    jp.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=jp.parent, delete=False, suffix=".tmp") as f:
        json.dump(ladder, f, indent=1, sort_keys=True)
        tmp = f.name
    os.replace(tmp, jp)
    return jp


# --------------------------------------------------------------------------- score/sweep
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=ladder_{card}_{step}
#SBATCH --qos={qos}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={walltime}
#SBATCH --output={logdir}/ladder_{card}_step{step}_%j.out
set -euo pipefail
module load ecmwf-toolbox || true
source {venv}/bin/activate
cd {repo}
export METVIEW_PYTHON_START_TIMEOUT=300
{predict_block}
python -m eval.cli evaluate --lane {lane} --host {host} \\
  --predictions-dir {evaldir}/predictions --output-dir {evaldir} \\
  --only {evaluators} --run-label ladder_{card}_step{step}{step_flag}
python -m eval._backends.storm_maps.render --predictions-dir {evaldir}/predictions \\
  --output-dir {evaldir}/evaluators/storm_maps {storm_args} || echo "storm_maps failed (non-fatal)"
python -m eval.jobs.ladder collect --profile {profile} --step {step} --eval-dir {evaldir} {collect_extra}
"""

PREDICT_ONESHOT = """python -m eval.cli predict --lane {lane} --host {host} --mode manual \\
  --checkpoint {ckpt} --bundle-dir {bundles} --output-dir {evaldir} \\
  --dates {dates} --steps {steps} --members {members}
"""

# candidate-B mode: M independent draws of ONE bundle (model noise is unseeded ->
# repeated predict = independent samples of the conditional PDF).
PREDICT_SEED_DRAWS = """for D in $(seq 1 {draws}); do
  python -m eval.cli predict --lane {lane} --host {host} --mode manual \\
    --checkpoint {ckpt} --bundle-dir {bundles} --output-dir {evaldir}/draw_$D \\
    --dates {dates} --steps {steps} --members {members}
done
python -m eval.jobs.ladder gatherdraws --eval-dir {evaldir} --draws {draws}
"""


def cmd_score(args: argparse.Namespace) -> None:
    prof = load_profile(args.profile)
    b = prof["budget"]
    paths = ladder_paths(prof)
    step = args.step if args.step is not None else infer_step(args.checkpoint)
    evaldir = paths["scratch"] / (f"baseline_{args.baseline_label}" if args.baseline_label else f"step_{step:07d}")
    evaldir.mkdir(parents=True, exist_ok=True)
    (paths["scratch"] / "logs").mkdir(parents=True, exist_ok=True)

    seed_draws = int(prof.get("seed_draws", 0) or 0)
    common = dict(lane=prof["lane"], host=prof["host"], ckpt=args.checkpoint,
                  bundles=prof["bundle_dir"], evaldir=evaldir,
                  dates=b["dates"], steps=b["steps"], members=b["members"])
    predict_block = PREDICT_ONESHOT.format(**common)
    collect_extra = f"--checkpoint {args.checkpoint}"
    if args.baseline_label:
        collect_extra += f" --baseline-label {args.baseline_label}"
    if seed_draws:
        sb = prof["seed_budget"]
        predict_block += PREDICT_SEED_DRAWS.format(
            lane=prof["lane"], host=prof["host"], ckpt=args.checkpoint,
            bundles=prof["bundle_dir"], evaldir=evaldir, draws=seed_draws,
            dates=sb["dates"], steps=sb["steps"], members=sb["members"])
        collect_extra += f" --seed-draws {seed_draws}"

    slurm = prof.get("slurm", {})
    script = SBATCH_TEMPLATE.format(
        card=prof["card_id"], step=step, qos=slurm.get("qos", "ng"),
        gpus=slurm.get("gpus", 1), cpus=slurm.get("cpus", 16), mem=slurm.get("mem", "128G"),
        walltime=slurm.get("walltime", "04:00:00"), logdir=paths["scratch"] / "logs",
        venv=prof.get("venv", "/home/ecm5702/dev/.ds-260612"), repo=REPO,
        predict_block=predict_block, lane=prof["lane"], host=prof["host"], evaldir=evaldir,
        evaluators=prof.get("evaluators", "tc,probabilistic,spectra"),
        step_flag=f" --steps {b['steps']}",
        storm_args=prof.get("storm_maps_args", ""),
        profile=prof["_name"], collect_extra=collect_extra)
    sb_path = evaldir / "ladder_score.sbatch"
    sb_path.write_text(script)
    if args.dry_run:
        print(f"[dry-run] wrote {sb_path}; not submitted")
        return
    out = subprocess.run(["sbatch", str(sb_path)], capture_output=True, text=True, check=True)
    print(out.stdout.strip(), f"-> {evaldir}")


def infer_step(ckpt: str) -> int:
    m = CKPT_STEP_RE.search(Path(ckpt).name)
    if not m:
        raise SystemExit(f"ladder: cannot infer step from checkpoint name {ckpt}; pass --step")
    return int(m.group(1))


def cmd_sweep(args: argparse.Namespace) -> None:
    prof = load_profile(args.profile)
    ladder = load_ladder(prof)
    done = {r["step"] for r in ladder["rows"]}
    cadence = int(prof.get("cadence", 10000))
    cks = {}
    for p in sorted(Path(args.run_root).rglob("inference-anemoi-by_step-*.ckpt")):
        m = CKPT_STEP_RE.search(p.name)
        if m:
            cks[int(m.group(1))] = p
    todo = [s for s in sorted(cks) if s not in done and s % cadence == 0]
    print(f"ladder sweep: {len(cks)} ckpts found, {len(done)} scored, {len(todo)} to score: {todo}")
    for s in todo:
        ns = argparse.Namespace(profile=args.profile, checkpoint=str(cks[s]), step=s,
                                baseline_label=None, dry_run=args.dry_run)
        cmd_score(ns)


# --------------------------------------------------------------------------- collect
def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def collect_metrics(evaldir: Path, steps_csv: str) -> dict:
    out: dict = {}
    prob = _read_json(evaldir / "evaluators" / "probabilistic" / "probabilistic_summary.json")
    for k, v in (prob.get("headline_metrics") or {}).items():
        # keep crps/fcrps/spread/rmse for tropics + n.hem only (ladder scope)
        if any(t in k for t in ("crps", "spread", "rmse")) and any(d in k for d in ("tropics", "n.hem")):
            out[k] = v
    spec = _read_json(evaldir / "evaluators" / "spectra" / "metrics.json")
    for k, v in spec.items():
        if k.startswith("spectra_") and ("relative_l2" in k or "score" in k):
            out[k] = v
    tc = _read_json(evaldir / "evaluators" / "tc" / "stats.json")
    for ev, evd in (tc.get("events") or {}).items():
        rows = ((evd.get("extreme_tail") or {}).get("rows")) or []
        for row in rows:
            src = row.get("source") or row.get("label") or "model"
            for m in ("mslp_min", "mslp_p001", "wind_max", "wind_p9999"):
                if m in row and row[m] is not None:
                    out[f"tc_{ev}_{src}_{m}"] = row[m]
    sm = _read_json(evaldir / "evaluators" / "storm_maps" / "storm_maps_spectra.json")
    for k, v in sm.items():
        if "fine_band" in k or "storm_box_min" in k or "slope_fine" in k:
            out[f"storm_{k}"] = v
    draws = _read_json(evaldir / "seed_draws.json")
    for k, v in draws.items():
        out[f"seed_{k}"] = v
    return out


def cmd_collect(args: argparse.Namespace) -> None:
    prof = load_profile(args.profile)
    evaldir = Path(args.eval_dir)
    metrics = collect_metrics(evaldir, prof["budget"]["steps"])
    if not metrics:
        raise SystemExit(f"ladder collect: no metrics found under {evaldir}")
    ladder = load_ladder(prof)
    # Forward-parity audit stamp (P2 finding 2026-07-24: scoring under a different anemoi-core
    # forward than the checkpoint trained with silently shifts spread ~2x — record, don't guess).
    try:
        core_sha = subprocess.run(
            ["git", "-C", "/home/ecm5702/dev/pristine/anemoi-core", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        core_sha = "unknown"
    row = {
        "step": args.step,
        "checkpoint": args.checkpoint,
        "eval_dir": str(evaldir),
        "scored_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_core_sha": core_sha,
        "metrics": metrics,
    }
    if args.baseline_label:
        ladder["baselines"][args.baseline_label] = row
    else:
        ladder["rows"] = [r for r in ladder["rows"] if r["step"] != args.step] + [row]
        ladder["rows"].sort(key=lambda r: r["step"])
    jp = save_ladder(prof, ladder)
    print(f"ladder collect: step {args.step} -> {jp} ({len(metrics)} metrics)")


def cmd_gatherdraws(args: argparse.Namespace) -> None:
    """Candidate-B: reduce per-draw NCs to eye/wind distribution stats."""
    import numpy as np
    import xarray as xr
    evaldir = Path(args.eval_dir)
    eyes, winds = [], []
    for d in range(1, args.draws + 1):
        for nc in sorted((evaldir / f"draw_{d}" / "predictions").glob("predictions_*.nc")):
            ds = xr.open_dataset(nc)
            msl = ds["y_pred"].sel(variable="msl") if "variable" in ds.dims else ds["msl"]
            eyes.append(float(np.min(np.asarray(msl))))
            try:
                u = np.asarray(ds["y_pred"].sel(variable="10u"))
                v = np.asarray(ds["y_pred"].sel(variable="10v"))
                winds.append(float(np.max(np.hypot(u, v))))
            except Exception:
                pass
            ds.close()
    stats = {}
    if eyes:
        e = np.asarray(eyes)
        stats.update(eye_median=float(np.median(e)), eye_p25=float(np.percentile(e, 25)),
                     eye_min=float(e.min()), eye_std=float(e.std(ddof=1)), n_draws=len(eyes))
    if winds:
        w = np.asarray(winds)
        stats.update(wind_median=float(np.median(w)), wind_p75=float(np.percentile(w, 75)),
                     wind_max=float(w.max()), wind_std=float(w.std(ddof=1)))
    (evaldir / "seed_draws.json").write_text(json.dumps(stats, indent=1))
    print(f"gatherdraws: {stats}")


# --------------------------------------------------------------------------- loss
def cmd_loss(args: argparse.Namespace) -> None:
    """Concatenate an MLflow FILE-STORE run family (parent + resume-leg child runs sharing
    mlflow.runName) into per-metric step series. File format: '<ts_ms> <value> <step>'."""
    prof = load_profile(args.profile)
    ladder = load_ladder(prof)
    wanted = args.metrics.split(",")
    root = Path(args.mlflow_dir)
    series: dict[str, list] = {}
    n_runs = 0
    for run_dir in root.iterdir():
        name_tag = run_dir / "tags" / "mlflow.runName"
        if not name_tag.exists() or args.run_name not in name_tag.read_text():
            continue
        n_runs += 1
        for metric in wanted:
            mf = run_dir / "metrics" / metric
            if not mf.exists():
                continue
            for line in mf.read_text().splitlines():
                parts = line.split()
                if len(parts) == 3:
                    series.setdefault(metric, []).append((int(parts[2]), float(parts[1])))
    if not n_runs:
        raise SystemExit(f"ladder loss: no runs matching '{args.run_name}' under {root}")
    for metric, pts in series.items():
        dedup = sorted(dict(pts).items())  # last value wins per step, sorted by step
        ladder["loss"][metric] = {"step": [s for s, _ in dedup], "value": [v for _, v in dedup]}
    save_ladder(prof, ladder)
    summary = ", ".join(f"{k}: {len(v['step'])} pts" for k, v in ladder["loss"].items())
    print(f"ladder loss ({n_runs} run legs merged): {summary}")


# --------------------------------------------------------------------------- plot
def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = load_profile(args.profile)
    ladder = load_ladder(prof)
    rows = ladder["rows"]
    if not rows:
        raise SystemExit("ladder plot: no rows yet")
    steps = [r["step"] for r in rows]

    def pick(sub: str, contains: tuple = ()) -> dict[str, list]:
        keys = sorted({k for r in rows for k in r["metrics"]
                       if k.startswith(sub) and all(c in k for c in contains)})
        return {k: [r["metrics"].get(k) for r in rows] for k in keys}

    panels = [
        ("CRPS (proxy)", pick("probabilistic_", ("crps",)), None),
        ("Spread (proxy)", pick("probabilistic_", ("spread",)), None),
        ("TC eye / wind (TREND indicator, ±replica noise)", pick("tc_"), 7.0),
        ("Fine-band / spectra", {**pick("storm_"), **pick("spectra_")}, None),
        ("Seed-draw distribution (candidate B)", pick("seed_"), None),
        ("Train/val loss (diagnostic only — loss ≠ skill)", None, None),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    for ax, (title, data, band) in zip(axes.flat, panels):
        ax.set_title(title, fontsize=10)
        if title.startswith("Train"):
            for metric, s in ladder.get("loss", {}).items():
                ax.plot(s["step"], s["value"], label=metric, lw=1)
        elif data:
            for k, vals in list(data.items())[:8]:
                ax.plot(steps, vals, marker="o", ms=3, lw=1, label=k.replace("probabilistic_", "")[:40])
                if band:
                    v = [x for x in vals if x is not None]
                    if v:
                        ax.fill_between(steps, [x - band if x else None for x in vals],
                                        [x + band if x else None for x in vals], alpha=0.08)
        # baseline reference lines
        for label, brow in ladder.get("baselines", {}).items():
            for k in (data or {}):
                bv = brow["metrics"].get(k)
                if bv is not None:
                    ax.axhline(bv, ls="--", lw=0.8, alpha=0.5)
        ax.legend(fontsize=6, loc="best")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Ladder — {ladder['card_id']} (profile {prof['_name']}; baselines dashed)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out) if args.out else ladder_paths(prof)["png"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"ladder plot -> {out}")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(prog="ladder", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--profile", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--step", type=int)
    s.add_argument("--baseline-label")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("sweep")
    s.add_argument("--profile", required=True)
    s.add_argument("--run-root", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("collect")
    s.add_argument("--profile", required=True)
    s.add_argument("--step", type=int, required=True)
    s.add_argument("--eval-dir", required=True)
    s.add_argument("--checkpoint", default="")
    s.add_argument("--baseline-label")
    s.add_argument("--seed-draws", type=int, default=0)
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("gatherdraws")
    s.add_argument("--eval-dir", required=True)
    s.add_argument("--draws", type=int, required=True)
    s.set_defaults(fn=cmd_gatherdraws)

    s = sub.add_parser("loss")
    s.add_argument("--profile", required=True)
    s.add_argument("--mlflow-dir", required=True)
    s.add_argument("--run-name", required=True)
    s.add_argument("--metrics", default="train_multi_dataset_loss_step,val_multi_dataset_loss_epoch")
    s.set_defaults(fn=cmd_loss)

    s = sub.add_parser("plot")
    s.add_argument("--profile", required=True)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_plot)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
