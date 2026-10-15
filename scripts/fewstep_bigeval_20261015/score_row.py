#!/usr/bin/env python3
"""Score one row of the few-step big eval (o96->o320) on one case set, evaluator by evaluator.

    python scripts/fewstep_bigeval_20261015/score_row.py --row edm_pw30c0 --set A \
        --pred-root <PRED_ROOT> --eval-root <EVAL_ROOT> [--dry-run]

score_row.sbatch calls this on Atos. For every evaluator of the set it runs

    python -m eval.cli evaluate --lane o96_o320 --predictions-dir <pred> \
        --output-dir <EVAL_ROOT>/<set>/<row>[/<view>] --only <evaluator> --opt ... --run-label <row>

once, with the options of PLAN below, and skips an evaluator whose results directory already
holds the ``.complete`` marker that eval.cli writes when it finishes. A failed evaluator does
not stop the others; the exit status is 1 when any failed.

Predictions: ``<PRED_ROOT>/<pred set dir>/<arm>/step_NNNNNN/<sampler dir>/predictions`` with
``<pred set dir>`` = ``set<SET>`` by default (``--pred-set-dir ''`` when PRED_ROOT already is
the set's root), or the row's ``predictions: {<SET>: dir}`` in rows.yaml.

Shared inputs built once under ``<EVAL_ROOT>/_cache`` (the EDM job runs first, submit_all.sh):
the O96<->O320 linear matrix pair of texture (eval.shared.inter_mat), texture's kNN cache and
shape's box geometry and mask caches.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LANE = "o96_o320"

# Atos inputs (hpcperm), overridable with --static-root.
STATIC = {
    "residual_stats": "residuals_statistics/o320_dict_0_72.npy",
    "forcings_npz": "o320-forcings.npz",
}
STATIC_ROOT = "/home/ecm5702/hpcperm/data"

EUROPE_VIEW = [33, 70, -15, 40]      # lane's Europe view, 33-70 N, 15 W-40 E
SET_A_STEP = 60

PROB_STATES = ["10u", "10v", "10ff", "2t", "msl", "tp", "t_850", "z_500"]
PROB_DOMAINS = ["global", "n.hem", "tropics", "s.hem", "europe"]
SPEC_VARS = ["10u", "10v", "ws", "2t", "msl", "tp"]
GLOBAL_SPEC_STATES = ["10u", "10v", "2t", "msl", "tp"]
TEXTURE_STATES = ["10u", "10v", "2t", "msl", "t_850"]
ECMWF_SPEC_STATES = ["10u", "10v", "2t", "msl", "sp", "t_850", "z_500"]


@dataclass
class Task:
    evaluator: str
    view: str                     # "" for <EVAL_ROOT>/<set>/<row>, else a sub-directory
    opts: dict = field(default_factory=dict)
    needs: tuple = ()             # shared inputs to prepare first: "inter_mat"


def _flow(v) -> str:
    """YAML flow text of an option value (what eval.cli's --opt reads back)."""
    return yaml.safe_dump(v, default_flow_style=True, width=1 << 20).strip().removesuffix("\n...").strip()


def plan(set_name: str, *, eval_root: Path, static_root: str, dates: list[str],
         with_ecmwf_spectra: bool) -> list[Task]:
    """Evaluators and options of one set (the pre-registered read, section 4)."""
    cache = eval_root / "_cache"
    if set_name == "A":
        st = [SET_A_STEP]
        tasks = [
            Task("probabilistic", "", {"weather_states": PROB_STATES, "domains": PROB_DOMAINS,
                                       "steps": st}),
            Task("box_spectra", "", {"box": EUROPE_VIEW, "steps": st, "variables": SPEC_VARS,
                                     "res_deg": 0.125}),
            Task("box_spectra", "natl", {"box": "north_atlantic", "steps": st,
                                         "variables": SPEC_VARS, "res_deg": 0.125}),
            Task("spectra_coherence", "", {"nside": "auto", "steps": st,
                                           "weather_states": GLOBAL_SPEC_STATES, "per_date": True}),
            Task("shape", "", {"box": "north_atlantic", "steps": st, "kernel_scale": 4,
                               "regrid_deg": 0.36, "window_deg": 8,
                               "cache_dir": str(cache / "shape"),
                               "stats_file": f"{static_root}/{STATIC['residual_stats']}"}),
            Task("texture", "", {"steps": st, "weather_states": TEXTURE_STATES,
                                 "input_source": "x_interp",
                                 "paths": {
                                     "up_matrix": str(cache / "inter_mat" / "up_lres_to_hres_linear.mat.npz"),
                                     "down_matrix": str(cache / "inter_mat" / "down_hres_to_lres_linear.mat.npz"),
                                     "residual_stats": f"{static_root}/{STATIC['residual_stats']}",
                                     "forcings_npz": f"{static_root}/{STATIC['forcings_npz']}",
                                     "knn_cache": str(cache / "texture" / "o320_knn32.npz")}},
                 needs=("inter_mat",)),
            Task("wind_extremes", "", {"steps": st, "input_source": "x_interp",
                                       "radii_km": [50, 75, 100, 150, 200], "adjacency_km": 40}),
            Task("precip_dist", "", {"steps": st, "tail_stats": True, "wet_threshold_mm": 0.1,
                                     "tail_stats_per_date": True}),
        ]
        if with_ecmwf_spectra:
            tasks.append(Task("spectra_ecmwf_v2", "", {
                "steps": st, "dates": [str(d) for d in dates], "members": list(range(1, 11)),
                "weather_states": ECMWF_SPEC_STATES}))
        return tasks
    if set_name == "B":
        return [
            Task("tc", "", {}),     # the lane's tc block: idalia + franklin vs OPER analysis
            Task("tc_structure", "", {"events": ["idalia", "franklin"],
                                      "params": {"cell_area_km2": 1211.2, "dr_km": 20}}),
        ]
    raise SystemExit(f"--set must be A or B, got {set_name!r}")


def load_row(rows_file: Path, label: str) -> dict:
    data = yaml.safe_load(rows_file.read_text())
    items = data["rows"] if isinstance(data, dict) else data
    for it in items:
        if it.get("label") == label:
            return it
    raise SystemExit(f"row {label!r} not in {rows_file}; labels: {[i.get('label') for i in items]}")


def sampler_dir(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(spec))


def predictions_dir(row: dict, set_name: str, pred_root: Path, pred_set_dir: str) -> Path:
    given = (row.get("predictions") or {}).get(set_name)
    if given:
        return Path(given)
    base = pred_root / pred_set_dir if pred_set_dir else pred_root
    return base / row["arm"] / f"step_{int(row['step']):06d}" / sampler_dir(row["sampler"]) / "predictions"


_FILE_RE = re.compile(r"predictions_(\d{8})_step(\d+)\.nc$")


def prediction_files(pred: Path) -> list[tuple[str, int, Path]]:
    out = []
    for p in sorted(pred.glob("predictions_*_step*.nc")):
        m = _FILE_RE.search(p.name)
        if m:
            out.append((m.group(1), int(m.group(2)), p))
    return out


def evaluate_cmd(python: list[str], pred: Path, out: Path, task: Task, label: str) -> list[str]:
    cmd = python + ["-m", "eval.cli", "evaluate", "--lane", LANE, "--predictions-dir", str(pred),
                    "--output-dir", str(out), "--only", task.evaluator, "--run-label", label]
    for k, v in task.opts.items():
        cmd += ["--opt", f"{task.evaluator}.{k}={_flow(v)}"]
    return cmd


def ensure_inter_mat(python: list[str], probe_file: Path, out_dir: Path, *, dry_run: bool) -> list[str]:
    """Build the O96<->O320 pair once (file lock: two jobs never build it together)."""
    cmd = python + ["-m", "eval.shared.inter_mat", "--predictions-file", str(probe_file),
                    "--out-dir", str(out_dir)]
    done = all((out_dir / n).exists() for n in ("up_lres_to_hres_linear.mat.npz",
                                                 "down_hres_to_lres_linear.mat.npz"))
    if dry_run or done:
        return cmd
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        subprocess.run(cmd, check=True, cwd=str(REPO))
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--row", required=True)
    p.add_argument("--set", required=True, choices=["A", "B"])
    p.add_argument("--pred-root", required=True, type=Path)
    p.add_argument("--eval-root", required=True, type=Path)
    p.add_argument("--rows", type=Path, default=HERE / "rows.yaml")
    p.add_argument("--pred-set-dir", default=None,
                   help="sub-directory of PRED_ROOT for the set (default set<SET>; '' for none)")
    p.add_argument("--static-root", default=STATIC_ROOT)
    p.add_argument("--python", default=sys.executable,
                   help="interpreter command for eval.cli (split on spaces)")
    p.add_argument("--only", default=None, help="comma list: run only these evaluators of the set")
    p.add_argument("--skip", default="", help="comma list of evaluators to leave out")
    p.add_argument("--with-ecmwf-spectra", action="store_true",
                   help="also run spectra_ecmwf_v2 (Atos AC only, gptosp; set A)")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = p.parse_args(argv)

    row = load_row(a.rows, a.row)
    if a.set not in [str(s) for s in row.get("sets", [])]:
        print(f"row {a.row} is not scored on set {a.set}; nothing to do")
        return 0
    pred_set_dir = f"set{a.set}" if a.pred_set_dir is None else a.pred_set_dir
    pred = predictions_dir(row, a.set, a.pred_root, pred_set_dir)
    files = prediction_files(pred)
    if not files and not a.dry_run:
        print(f"ERROR: no predictions_YYYYMMDD_stepNNN.nc under {pred}", file=sys.stderr)
        return 2
    if a.set == "A":
        files_a = [f for f in files if f[1] == SET_A_STEP]
        if files and len(files_a) != 36:
            print(f"WARNING: set A expects 36 files at step {SET_A_STEP:03d}, found {len(files_a)} in {pred}")
    dates = sorted({d for d, s, _ in files if a.set != "A" or s == SET_A_STEP})

    python = shlex.split(a.python)
    out_row = a.eval_root / a.set / a.row
    only = set(filter(None, (a.only or "").split(","))) or None
    skip = set(filter(None, a.skip.split(",")))
    tasks = [t for t in plan(a.set, eval_root=a.eval_root, static_root=a.static_root, dates=dates,
                             with_ecmwf_spectra=a.with_ecmwf_spectra)
             if (only is None or t.evaluator in only) and t.evaluator not in skip]

    print(f"row {a.row} set {a.set}: {len(files)} files in {pred}")
    failures: list[str] = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for t in tasks:
        out = out_row / t.view if t.view else out_row
        tag = f"{t.evaluator}{'/' + t.view if t.view else ''}"
        if (out / "evaluators" / t.evaluator / ".complete").exists():
            print(f"[skip] {tag}: .complete present in {out / 'evaluators' / t.evaluator}")
            continue
        if "inter_mat" in t.needs:
            probe = files[0][2] if files else pred / "predictions_YYYYMMDD_step060.nc"
            try:
                cmd = ensure_inter_mat(python, probe, a.eval_root / "_cache" / "inter_mat",
                                       dry_run=a.dry_run)
                if a.dry_run:
                    print("[dry-run] " + shlex.join(cmd) + "   # once, if missing")
            except subprocess.CalledProcessError as exc:
                failures.append(f"{tag}: inter_mat build failed ({exc.returncode})")
                continue
        cmd = evaluate_cmd(python, pred, out, t, a.row)
        if a.dry_run:
            print("[dry-run] " + shlex.join(cmd))
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"[run] {tag}\n  " + shlex.join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
        if rc != 0:
            failures.append(f"{tag}: exit {rc}")
    if failures:
        print("FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
