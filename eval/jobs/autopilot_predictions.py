from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
TERMINAL_ALL = TERMINAL_OK | TERMINAL_BAD
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass
class JobTrack:
    name: str
    script: Path
    dependency: str | None = None
    job_id: str | None = None
    retries: int = 0
    max_retries: int = 1
    state: str = "PENDING"


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _submit(script: Path, dependency: str | None = None) -> str:
    cmd = ["sbatch"]
    if dependency:
        cmd += [f"--dependency=afterok:{dependency}"]
    cmd += [str(script)]
    out = _run(cmd)
    m = re.search(r"(\d+)$", out)
    if not m:
        raise RuntimeError(f"Could not parse job id from sbatch output: {out}")
    return m.group(1)


def _cancel(job_id: str) -> None:
    subprocess.run(["scancel", job_id], check=False)


def _sacct_state(job_id: str) -> str:
    try:
        out = _run(["sacct", "-j", job_id, "--format=State", "-n", "-P"])
    except subprocess.CalledProcessError:
        return "PENDING"
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        s = s.split("|")[0].strip().upper()
        if s and s != "UNKNOWN":
            return s
    return "PENDING"


def _squeue_state(job_id: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", job_id, "-o", "%T|%r"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    line = out.splitlines()[0].strip()
    if not line:
        return None
    state = line.split("|", 1)[0].strip().upper()
    reason = ""
    if "|" in line:
        reason = line.split("|", 1)[1].strip().upper()
    if "DEPENDENCYNEVERSATISFIED" in reason:
        return "FAILED"
    return state or None


def _write_state(state_file: Path, run_id: str, jobs: dict[str, JobTrack]) -> None:
    payload = {
        "run_id": run_id,
        "updated_epoch": int(time.time()),
        "jobs": {
            k: {
                "job_id": v.job_id,
                "state": v.state,
                "retries": v.retries,
                "max_retries": v.max_retries,
                "script": str(v.script),
                "dependency": v.dependency,
            }
            for k, v in jobs.items()
        },
    }
    state_file.write_text(json.dumps(payload, indent=2))


def _validate_safe_name(name: str, *, label: str) -> None:
    if not SAFE_NAME_RE.fullmatch(name):
        raise SystemExit(
            f"{label} contains unsafe characters: {name!r}. "
            "Allowed pattern: [A-Za-z0-9._-]+"
        )


def _validate_eval_root(eval_root: Path) -> None:
    # Catch accidental nesting where users pass an existing run directory as eval root.
    if (eval_root / "logs").is_dir() and (eval_root / "jobs").is_dir():
        raise SystemExit(
            f"Refusing eval root that already looks like a run directory: {eval_root}. "
            "Pass the parent eval root (for example /home/ecm5702/perm/eval)."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Autopilot for predictions->eval chained pipeline")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--eval-root", default="/home/ecm5702/perm/eval")
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--input-root", default="/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia")
    ap.add_argument("--ckpt-id", default="4a5b2f1b24b84c52872bfcec1410b00f")
    ap.add_argument("--predict-qos", default="dg")
    ap.add_argument("--predict-time", default="00:30:00")
    ap.add_argument("--predict-cpus", default="32")
    ap.add_argument("--predict-mem", default="256G")
    ap.add_argument("--predict-gpus", default="1")
    ap.add_argument("--eval-qos", default="nf")
    ap.add_argument("--eval-time", default="08:00:00")
    ap.add_argument("--eval-cpus", default="8")
    ap.add_argument("--eval-mem", default="64G")
    ap.add_argument(
        "--allow-existing-run-dir",
        action="store_true",
        help="Allow reuse of an existing run directory (explicitly unsafe).",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    _validate_safe_name(args.run_id, label="run-id")
    eval_root = Path(args.eval_root).expanduser().resolve()
    _validate_eval_root(eval_root)
    run_dir = eval_root / args.run_id
    if run_dir.exists() and not args.allow_existing_run_dir:
        raise SystemExit(
            f"Run directory already exists: {run_dir}. "
            "Refusing silent reuse; use a new run-id or pass --allow-existing-run-dir explicitly."
        )

    project_root = Path(__file__).resolve().parents[2]
    launch_script = project_root / "eval/jobs/launch_predictions_eval_suite.sh"

    launch_args = [
        "--run-id", args.run_id,
        "--eval-root", args.eval_root,
        "--input-root", args.input_root,
        "--ckpt-id", args.ckpt_id,
        "--predict-qos", args.predict_qos,
        "--predict-time", args.predict_time,
        "--predict-cpus", args.predict_cpus,
        "--predict-mem", args.predict_mem,
        "--predict-gpus", args.predict_gpus,
        "--eval-qos", args.eval_qos,
        "--eval-time", args.eval_time,
        "--eval-cpus", args.eval_cpus,
        "--eval-mem", args.eval_mem,
        "--dry-run",
    ]
    if args.allow_existing_run_dir:
        launch_args.append("--allow-existing-run-dir")
    subprocess.check_call([str(launch_script)] + launch_args)

    gen_dir = run_dir / "jobs"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_file = logs_dir / "autopilot_predictions_state.json"

    jobs = {
        "predict25": JobTrack(
            name="predict25",
            script=gen_dir / f"predict25_{args.run_id}.sbatch",
            max_retries=args.max_retries,
        ),
        "eval25": JobTrack(
            name="eval25",
            script=gen_dir / f"eval25_{args.run_id}.sbatch",
            dependency="predict25",
            max_retries=args.max_retries,
        ),
    }

    resumed = False
    if args.resume and state_file.exists():
        try:
            prev = json.loads(state_file.read_text())
            for k, j in jobs.items():
                pj = prev.get("jobs", {}).get(k, {})
                if pj.get("job_id"):
                    j.job_id = str(pj["job_id"])
                if "retries" in pj:
                    j.retries = int(pj["retries"])
            if jobs["predict25"].job_id and jobs["eval25"].job_id:
                resumed = True
        except Exception:
            resumed = False

    if not resumed:
        jobs["predict25"].job_id = _submit(jobs["predict25"].script)
        jobs["eval25"].job_id = _submit(jobs["eval25"].script, dependency=jobs["predict25"].job_id)
        print(f"Autopilot started for {args.run_id}. State file: {state_file}")
    else:
        print(f"Autopilot resumed for {args.run_id}. State file: {state_file}")
    _write_state(state_file, args.run_id, jobs)

    while True:
        for j in jobs.values():
            if j.job_id:
                sq = _squeue_state(j.job_id)
                if sq:
                    j.state = sq
                else:
                    j.state = _sacct_state(j.job_id)

        if all(j.state in TERMINAL_ALL for j in jobs.values()):
            break

        # Retry failed predict job and relink eval dependency.
        p = jobs["predict25"]
        e = jobs["eval25"]
        if p.state in TERMINAL_BAD and p.retries < p.max_retries:
            old_predict = p.job_id
            old_eval = e.job_id
            p.retries += 1
            p.job_id = _submit(p.script)
            p.state = "PENDING"
            e.job_id = _submit(e.script, dependency=p.job_id)
            e.state = "PENDING"
            if old_predict and old_predict != p.job_id:
                _cancel(old_predict)
            if old_eval and old_eval != e.job_id:
                _cancel(old_eval)

        # Retry eval independently if predict already succeeded.
        if e.state in TERMINAL_BAD and e.retries < e.max_retries and p.state in TERMINAL_OK:
            old_eval = e.job_id
            e.retries += 1
            e.job_id = _submit(e.script, dependency=p.job_id)
            e.state = "PENDING"
            if old_eval and old_eval != e.job_id:
                _cancel(old_eval)

        _write_state(state_file, args.run_id, jobs)
        time.sleep(max(5, args.poll_seconds))

    _write_state(state_file, args.run_id, jobs)
    bad = {k: v.state for k, v in jobs.items() if v.state in TERMINAL_BAD}
    if bad:
        print("Autopilot finished with failures:")
        for k, v in bad.items():
            print(f"  {k}: {v}")
        raise SystemExit(1)

    print("Autopilot finished successfully.")


if __name__ == "__main__":
    main()
