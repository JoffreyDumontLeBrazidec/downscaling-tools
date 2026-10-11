"""On-demand segments (chain_next): a segment queues the same job file once more, afterany itself.

Runs toy_next_job.sh under a fake Slurm (fake_slurm/: sbatch, squeue, scontrol keep a queue in files) and a small
driver that starts pending jobs in order once their afterany dependency has ended. No torch needed.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import chain_state  # noqa: E402

JOB = HERE / "toy_next_job.sh"


def _jobs(slurm):
    out = {}
    for f in sorted((slurm / "jobs").glob("*"), key=lambda p: int(p.name)):
        out[int(f.name)] = dict(line.split("=", 1) for line in f.read_text().splitlines())
    return out


def _set_state(slurm, jid, state):
    f = slurm / "jobs" / str(jid)
    f.write_text("".join("state={}\n".format(state) if l.startswith("state=") else l + "\n"
                         for l in f.read_text().splitlines()))


def _env(slurm, **extra):
    env = dict(os.environ, FAKE_SLURM_DIR=str(slurm), CHAIN_PY=sys.executable, TOY_CHAIN_TOOLS=str(HERE.parent),
               PATH="{}:{}".format(HERE / "fake_slurm", os.environ["PATH"]), USER="tester")
    env.update(extra)
    return env


def _submit(slurm, root, end, **extra):
    r = subprocess.run(["sbatch", "--parsable", str(JOB), str(root), str(end)], env=_env(slurm, **extra),
                       capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def _drain(slurm, max_jobs=20, partial=False, **extra):
    """Run pending jobs whose afterany dependency has ended, oldest first. Returns {job id: returncode}.
    partial: stop after max_jobs instead of requiring the queue to be empty."""
    ran = {}
    for _ in range(max_jobs):
        jobs = _jobs(slurm)
        ready = [j for j, d in jobs.items() if d["state"] == "PENDING" and
                 (not d["dependency"] or jobs[int(d["dependency"].split(":")[1])]["state"] in ("COMPLETED", "FAILED"))]
        if not ready:
            return ran
        jid = ready[0]
        _set_state(slurm, jid, "RUNNING")
        cmd = jobs[jid]["command"].split()
        r = subprocess.run(["bash"] + cmd, env=_env(slurm, SLURM_JOB_ID=str(jid), SLURM_NTASKS="4",
                                                     SLURM_SUBMIT_DIR=jobs[jid]["submit_dir"], **extra),
                           capture_output=True, text=True, timeout=120)
        ran[jid] = r.returncode
        _set_state(slurm, jid, "COMPLETED" if r.returncode == 0 else "FAILED")
    if partial:
        return ran
    raise AssertionError("queue did not drain in {} jobs".format(max_jobs))


@pytest.fixture
def slurm(tmp_path):
    d = tmp_path / "slurm"
    (d / "jobs").mkdir(parents=True)
    return d


def test_one_submission_runs_the_whole_chain(tmp_path, slurm):
    root = tmp_path / "chain"
    chain_state.main(["init", "--root", str(root)])
    first = _submit(slurm, root, 3000)
    ran = _drain(slurm)
    # three training segments (0->1000->2000->3000), then one that finds the run done and queues nothing
    assert list(ran.values()) == [0, 0, 0, 0] and min(ran) == first
    assert chain_state.checkpoint_step(str(root / "run1" / "last.ckpt")) == 3000
    jobs = _jobs(slurm)
    assert all(d["state"] != "PENDING" for d in jobs.values())
    for jid, d in jobs.items():
        if jid != first:
            assert d["dependency"] == "afterany:{}".format(jid - 1)
            assert d["leaked_job_id"] == ""  # the parent's SLURM_* variables were not passed to sbatch
    assert "next queued job=" in (root / "CHAIN.log").read_text()


def test_no_progress_stops_after_three_segments(tmp_path, slurm):
    root = tmp_path / "chain"
    chain_state.main(["init", "--root", str(root)])
    _submit(slurm, root, 3000)
    ran = _drain(slurm, TOY_FAIL="1")
    assert len(ran) == 3 and all(rc != 0 for rc in ran.values())
    assert "3 segments in a row started at step 0" in (root / "CHAIN.log").read_text()


def test_progress_resets_the_count(tmp_path, slurm):
    root = tmp_path / "chain"
    chain_state.main(["init", "--root", str(root)])
    _submit(slurm, root, 3000)
    _drain(slurm, max_jobs=1, partial=True)  # 0 -> 1000
    _drain(slurm, max_jobs=1, partial=True, TOY_FAIL="1")  # starts at 1000, dies
    ran = _drain(slurm)  # starts at 1000 again (2nd in a row: allowed), 1000 -> 2000 -> 3000, then done
    assert list(ran.values()) == [0, 0, 0]
    assert chain_state.checkpoint_step(str(root / "run1" / "last.ckpt")) == 3000
    assert "in a row started" not in (root / "CHAIN.log").read_text()


def test_stop_file(tmp_path, slurm):
    root = tmp_path / "chain"
    chain_state.main(["init", "--root", str(root)])
    _submit(slurm, root, 3000)
    _drain(slurm, max_jobs=1, partial=True)
    (root / "CHAIN.stop").touch()
    ran = _drain(slurm)
    assert list(ran.values()) == [0]  # the queued successor exits 0 at once and queues nothing
    assert chain_state.checkpoint_step(str(root / "run1" / "last.ckpt")) == 1000


def test_already_queued_successor_is_not_duplicated(tmp_path, slurm):
    """An older pre-submitted series (same job file) is already pending: chain_next adds nothing."""
    root = tmp_path / "chain"
    chain_state.main(["init", "--root", str(root)])
    first = _submit(slurm, root, 3000)
    _submit(slurm, root, 3000)  # pending, as if pre-submitted (no dependency in this fake)
    _drain(slurm, max_jobs=1, partial=True)
    assert len(_jobs(slurm)) == 2
    assert "next already queued job={}".format(first + 1) in (root / "CHAIN.log").read_text()


def test_outside_slurm_submits_nothing(tmp_path, slurm):
    root = tmp_path / "chain"
    chain_state.main(["init", "--root", str(root)])
    r = subprocess.run(["bash", str(JOB), str(root), "1000"], env=_env(slurm), capture_output=True, text=True)
    assert r.returncode == 0 and "not inside a Slurm job" in r.stdout
    assert not list((slurm / "jobs").glob("*"))
