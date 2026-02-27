from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


TERMINAL_OK = {"COMPLETED"}
TERMINAL_BAD = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
TERMINAL_ALL = TERMINAL_OK | TERMINAL_BAD


@dataclass
class JobTrack:
    name: str
    script: Path
    dependency: str | None = None
    job_id: str | None = None
    retries: int = 0
    max_retries: int = 2
    state: str = "PENDING"
    deps: list[str] = field(default_factory=list)


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


def _submit(script: Path, dependency: str | None = None) -> str:
    cmd = ["sbatch"]
    if dependency:
        cmd += [f"--dependency=afterok:{dependency}"]
    cmd += [str(script)]
    out = _run(cmd)
    match = re.search(r"(\d+)$", out)
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {out}")
    return match.group(1)


def _cancel(job_id: str) -> None:
    subprocess.run(["scancel", job_id], check=False)


def _sacct_state(job_id: str) -> str:
    try:
        out = _run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=State",
                "-n",
                "-P",
            ]
        )
    except subprocess.CalledProcessError:
        return "PENDING"
    for line in out.splitlines():
        state = line.strip()
        if not state:
            continue
        state = state.split("|")[0].strip().upper()
        if state and state != "UNKNOWN":
            return state
    return "PENDING"


def _squeue_reason(job_id: str) -> tuple[str | None, str | None]:
    try:
        out = _run(["squeue", "-h", "-j", job_id, "-o", "%T|%r"])
    except subprocess.CalledProcessError:
        return None, None
    if not out:
        return None, None
    line = out.splitlines()[0].strip()
    if not line:
        return None, None
    if "|" in line:
        state, reason = line.split("|", 1)
        return state.strip().upper() or None, reason.strip()
    return line.strip().upper() or None, None


def _job_state(job_id: str) -> str:
    sq_state, sq_reason = _squeue_reason(job_id)
    if sq_state:
        if sq_reason and "DEPENDENCYNEVERSATISFIED" in sq_reason.upper():
            return "FAILED"
        return sq_state
    return _sacct_state(job_id)


def _write_state(state_file: Path, jobs: dict[str, JobTrack], expver: str) -> None:
    payload = {
        "expver": expver,
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


def _all_terminal(jobs: dict[str, JobTrack]) -> bool:
    return all(j.state in TERMINAL_ALL for j in jobs.values())


def _regenerate_scripts(launch_script: Path, launch_args: list[str]) -> None:
    cmd = [str(launch_script)] + launch_args + ["--dry-run"]
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Background autopilot for full eval suite.")
    parser.add_argument("--expver", required=True)
    parser.add_argument("--eval-root", default="/home/ecm5702/perm/eval")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--eval-date", default="20230801/20230802")
    parser.add_argument("--eval-number", default="1/2")
    parser.add_argument("--eval-step", default="24/120")
    parser.add_argument("--quaver-first-date", default="20230826")
    parser.add_argument("--quaver-last-date", default="20230827")
    parser.add_argument("--quaver-nmem", default="2")
    parser.add_argument("--spectra-date", default="20230826/to/20230827/by/1")
    parser.add_argument("--spectra-step", default="144")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    launch_script = project_root / "eval/jobs/launch_full_eval_suite.sh"
    run_dir = Path(args.eval_root) / args.expver
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_file = logs_dir / "autopilot_state.json"

    launch_args = [
        "--expver",
        args.expver,
        "--eval-root",
        args.eval_root,
        "--eval-date",
        args.eval_date,
        "--eval-number",
        args.eval_number,
        "--eval-step",
        args.eval_step,
        "--quaver-first-date",
        args.quaver_first_date,
        "--quaver-last-date",
        args.quaver_last_date,
        "--quaver-nmem",
        args.quaver_nmem,
        "--spectra-date",
        args.spectra_date,
        "--spectra-step",
        args.spectra_step,
    ]
    _regenerate_scripts(launch_script, launch_args)

    gen_dir = run_dir / "jobs"
    jobs: dict[str, JobTrack] = {
        "eval_mars": JobTrack(
            name="eval_mars",
            script=gen_dir / f"eval_mars_{args.expver}.sbatch",
            max_retries=args.max_retries,
        ),
        "quaver": JobTrack(
            name="quaver",
            script=gen_dir / f"quaver_{args.expver}.sbatch",
            max_retries=args.max_retries,
        ),
        "spectra_compute": JobTrack(
            name="spectra_compute",
            script=gen_dir / f"spectra_compute_{args.expver}.sbatch",
            max_retries=args.max_retries,
        ),
        "spectra_plot": JobTrack(
            name="spectra_plot",
            script=gen_dir / f"spectra_plot_{args.expver}.sbatch",
            dependency="spectra_compute",
            max_retries=args.max_retries,
        ),
        "tc_retrieve": JobTrack(
            name="tc_retrieve",
            script=gen_dir / f"tc_retrieve_{args.expver}.sbatch",
            max_retries=args.max_retries,
        ),
        "tc_plot": JobTrack(
            name="tc_plot",
            script=gen_dir / f"tc_plot_{args.expver}.sbatch",
            dependency="tc_retrieve",
            max_retries=args.max_retries,
        ),
    }

    for key in ["eval_mars", "quaver", "spectra_compute", "tc_retrieve"]:
        jobs[key].job_id = _submit(jobs[key].script)

    jobs["spectra_plot"].job_id = _submit(
        jobs["spectra_plot"].script, dependency=jobs["spectra_compute"].job_id
    )
    jobs["tc_plot"].job_id = _submit(
        jobs["tc_plot"].script, dependency=jobs["tc_retrieve"].job_id
    )

    _write_state(state_file, jobs, args.expver)
    print(f"Autopilot started for {args.expver}. State file: {state_file}")

    while True:
        for j in jobs.values():
            if not j.job_id:
                continue
            j.state = _job_state(j.job_id)

        if _all_terminal(jobs):
            break

        for j in jobs.values():
            if j.state in TERMINAL_BAD and j.retries < j.max_retries:
                old_job_id = j.job_id
                j.retries += 1
                if j.dependency:
                    dep_id = jobs[j.dependency].job_id
                    j.job_id = _submit(j.script, dependency=dep_id)
                else:
                    j.job_id = _submit(j.script)
                j.state = "PENDING"
                if old_job_id and old_job_id != j.job_id:
                    _cancel(old_job_id)

                if j.name == "spectra_compute":
                    sp = jobs["spectra_plot"]
                    old_sp_job_id = sp.job_id
                    sp.job_id = _submit(sp.script, dependency=j.job_id)
                    sp.state = "PENDING"
                    if old_sp_job_id and old_sp_job_id != sp.job_id:
                        _cancel(old_sp_job_id)
                if j.name == "tc_retrieve":
                    tp = jobs["tc_plot"]
                    old_tp_job_id = tp.job_id
                    tp.job_id = _submit(tp.script, dependency=j.job_id)
                    tp.state = "PENDING"
                    if old_tp_job_id and old_tp_job_id != tp.job_id:
                        _cancel(old_tp_job_id)

        _write_state(state_file, jobs, args.expver)
        time.sleep(max(5, args.poll_seconds))

    _write_state(state_file, jobs, args.expver)
    bad = {k: v.state for k, v in jobs.items() if v.state in TERMINAL_BAD}
    if bad:
        print("Autopilot finished with failures:")
        for k, v in bad.items():
            print(f"  {k}: {v}")
        raise SystemExit(1)
    print("Autopilot finished successfully.")


if __name__ == "__main__":
    main()
