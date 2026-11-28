"""Overlap mode (chain_handover.py): the successor queues with no dependency and takes over at a checkpoint.

A "writer" segment is a background process (toy_writer.py) registered RUNNING in the fake Slurm with its pid; it
rewrites last.ckpt every TOY_PERIOD seconds. The fake scancel kills that pid. The invariant checked everywhere: the
takeover returns only after the writer's process is dead and Slurm no longer lists it.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import chain_state  # noqa: E402

HANDOVER = HERE.parent / "chain_handover.py"
WRITER_PY = HERE / "toy_writer.py"
FAST = {"CHAIN_TAKEOVER_POLL": "0.1", "CHAIN_TAKEOVER_STABLE": "0.3", "CHAIN_TAKEOVER_KILL_WAIT": "5"}


def _env(slurm, **extra):
    env = dict(os.environ, FAKE_SLURM_DIR=str(slurm), USER="tester",
               PATH="{}:{}".format(HERE / "fake_slurm", os.environ["PATH"]), **FAST)
    env.update(extra)
    return env


def _job(slurm, jid, state, pid=None, command="/x/job.sbatch"):
    lines = ["state={}".format(state), "command={}".format(command), "dependency=", "begin="]
    if pid:
        lines.append("pid={}".format(pid))
    (slurm / "jobs" / str(jid)).write_text("\n".join(lines) + "\n")


def _state(slurm, jid):
    for line in (slurm / "jobs" / str(jid)).read_text().splitlines():
        if line.startswith("state="):
            return line.split("=", 1)[1]


def _takeover(slurm, root, me, *args, **extra):
    return subprocess.run([sys.executable, str(HANDOVER), "takeover", "--root", str(root)] + list(args),
                          env=_env(slurm, SLURM_JOB_ID=str(me), **extra), capture_output=True, text=True,
                          timeout=60)


def _ckpt(root, step):
    import io
    import pickle
    import zipfile
    (root / "run1").mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pickle.dump({"global_step": step}, buf, protocol=2)
    with zipfile.ZipFile(str(root / "run1" / "last.ckpt"), "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())


@pytest.fixture
def slurm(tmp_path):
    d = tmp_path / "slurm"
    (d / "jobs").mkdir(parents=True)
    return d


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "chain"
    chain_state.main(["init", "--root", str(r)])
    _ckpt(r, 1000)
    chain_state.main(["adopt", "--root", str(r), "--run-id", "run1"])
    return r


def _start_writer(root, period, start=1000):
    return subprocess.Popen([sys.executable, str(WRITER_PY), str(root / "run1" / "last.ckpt"), str(start),
                             str(period)])


def test_no_writer_file(slurm, root):
    r = _takeover(slurm, root, 201)
    assert r.returncode == 0, r.stderr
    assert (root / "CHAIN.writer").read_text().split()[0] == "201"


def test_writer_already_gone(slurm, root):
    _job(slurm, 200, "COMPLETED")
    (root / "CHAIN.writer").write_text("200 host t\n")
    r = _takeover(slurm, root, 201)
    assert r.returncode == 0, r.stderr
    assert "no longer queued" in (root / "CHAIN.log").read_text()
    assert not (slurm / "scancel.log").exists()


def test_running_writer_is_stopped_after_its_next_checkpoint(slurm, root):
    w = _start_writer(root, period=1.0)
    try:
        _job(slurm, 200, "RUNNING", pid=w.pid)
        (root / "CHAIN.writer").write_text("200 host t\n")
        t0 = time.time()
        r = _takeover(slurm, root, 201, "--timeout", "20")
        assert r.returncode == 0, r.stderr
        # the writer process is dead and Slurm no longer lists it BEFORE the takeover returned
        assert w.poll() is not None
        assert _state(slurm, 200) == "CANCELLED"
        assert time.time() - t0 < 15
        step = chain_state.checkpoint_step(str(root / "run1" / "last.ckpt"))
        assert step > 1000
        log = (root / "CHAIN.log").read_text()
        assert "scancel 200 after checkpoint" in log and "writer is now 201" in log
    finally:
        w.kill()


def test_timeout_kill_and_refuse(slurm, root):
    w = _start_writer(root, period=1000)  # never writes a new checkpoint in time
    try:
        _job(slurm, 200, "RUNNING", pid=w.pid)
        (root / "CHAIN.writer").write_text("200 host t\n")
        r = _takeover(slurm, root, 201, "--timeout", "1", "--on-timeout", "refuse")
        assert r.returncode == 3 and w.poll() is None and _state(slurm, 200) == "RUNNING"
        assert (root / "CHAIN.writer").read_text().split()[0] == "200"
        r = _takeover(slurm, root, 202, "--timeout", "1")
        assert r.returncode == 0, r.stderr
        assert w.wait(timeout=5) is not None and _state(slurm, 200) == "CANCELLED"
        assert "after timeout" in (root / "CHAIN.log").read_text()
    finally:
        w.kill()


def test_pending_writer_is_cancelled(slurm, root):
    _job(slurm, 200, "PENDING")
    (root / "CHAIN.writer").write_text("200 host t\n")
    r = _takeover(slurm, root, 201)
    assert r.returncode == 0, r.stderr
    assert _state(slurm, 200) == "CANCELLED"


def test_lock_held_by_live_job_refuses(slurm, root):
    _job(slurm, 300, "RUNNING")
    lock = root / ".chain_takeover.lock"
    lock.mkdir()
    (lock / "owner").write_text("300\n")
    r = _takeover(slurm, root, 201)
    assert r.returncode == 3 and "held by live job 300" in r.stderr
    _state_file = slurm / "jobs" / "300"
    _state_file.write_text(_state_file.read_text().replace("RUNNING", "COMPLETED"))
    r = _takeover(slurm, root, 201)
    assert r.returncode == 0 and "broke stale lock" in (root / "CHAIN.log").read_text()


def test_slurm_query_error_is_never_read_as_gone(slurm, root, tmp_path):
    w = _start_writer(root, period=0.5)
    try:
        _job(slurm, 200, "RUNNING", pid=w.pid)
        (root / "CHAIN.writer").write_text("200 host t\n")
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "squeue").write_text("#!/bin/bash\necho 'slurm_load_jobs error: Socket timed out' >&2; exit 1\n")
        (broken / "squeue").chmod(0o755)
        env_path = "{}:{}:{}".format(broken, HERE / "fake_slurm", os.environ["PATH"])
        r = subprocess.run([sys.executable, "-c",
                            "import sys,time; sys.path.insert(0,{!r}); import chain_handover as h; "
                            "h.time.sleep=lambda s: None\ntry:\n h.job_state(200)\n print('NOERR')\n"
                            "except h.QueryError:\n print('QUERYERROR')".format(str(HERE.parent))],
                           env=_env(slurm, PATH=env_path), capture_output=True, text=True, timeout=30)
        assert "QUERYERROR" in r.stdout, r.stdout + r.stderr
    finally:
        w.kill()


def test_next_submits_without_dependency_and_no_duplicate(slurm, root, tmp_path):
    job = tmp_path / "job.sbatch"
    job.write_text("#!/bin/bash\n")
    chain_state.main(["log", "--root", str(root), "x resolve action=resume run_id=run1 expect_step=1000 end"])
    cmd = [sys.executable, str(HANDOVER), "next", "--root", str(root), "--job-file", str(job),
           "--begin-delay", "3600", "--", "a1"]
    r = subprocess.run(cmd, env=_env(slurm, SLURM_JOB_ID="201", SLURM_NTASKS="4"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    jobs = sorted((slurm / "jobs").iterdir())
    assert len(jobs) == 1
    d = dict(l.split("=", 1) for l in jobs[0].read_text().splitlines())
    assert d["dependency"] == "" and d["begin"] == "now+3600" and d["leaked_job_id"] == ""
    r = subprocess.run(cmd, env=_env(slurm, SLURM_JOB_ID="201"), capture_output=True, text=True)
    assert "already queued" in r.stdout and len(list((slurm / "jobs").iterdir())) == 1


def test_overlap_chain_end_to_end(slurm, root, tmp_path):
    """Job body in overlap mode: takeover, resolve, next, then a writer that trains until cancelled."""
    job = HERE / "toy_overlap_job.sh"
    env = _env(slurm, CHAIN_PY=sys.executable, TOY_CHAIN_TOOLS=str(HERE.parent))
    # segment A starts (nobody to take over) and trains in the background
    _job(slurm, 200, "RUNNING", command="{} {} 100000".format(job, root))
    a = subprocess.Popen(["bash", str(job), str(root), "100000"], env=dict(env, SLURM_JOB_ID="200",
                         SLURM_SUBMIT_DIR=str(tmp_path)))
    f = slurm / "jobs" / "200"
    f.write_text(f.read_text() + "pid={}\n".format(a.pid))
    try:
        for _ in range(100):
            if (root / "CHAIN.writer").exists() and list((slurm / "jobs").glob("10*")):
                break
            time.sleep(0.1)
        succ = sorted(p.name for p in (slurm / "jobs").glob("10*"))
        assert succ, "A queued no successor"
        b_id = succ[0]
        assert dict(l.split("=", 1) for l in (slurm / "jobs" / b_id).read_text().splitlines())["dependency"] == ""
        # the successor B starts while A still trains
        f2 = slurm / "jobs" / b_id
        f2.write_text(f2.read_text().replace("state=PENDING", "state=RUNNING"))
        b = subprocess.Popen(["bash", str(job), str(root), "100000"], env=dict(env, SLURM_JOB_ID=b_id,
                             SLURM_SUBMIT_DIR=str(tmp_path)))
        for _ in range(200):
            if (root / "CHAIN.writer").read_text().split()[0] == b_id:
                break
            time.sleep(0.1)
        assert (root / "CHAIN.writer").read_text().split()[0] == b_id
        assert a.wait(timeout=10) is not None and _state(slurm, 200) == "CANCELLED"
        log = (root / "CHAIN.log").read_text()
        assert "scancel 200 after checkpoint" in log
        b.kill()
    finally:
        for p in (a,):
            if p.poll() is None:
                p.kill()
