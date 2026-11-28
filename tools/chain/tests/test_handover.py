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
    lock.write_text("300 t\n")
    r = _takeover(slurm, root, 201)
    assert r.returncode == 3 and "held by live job 300" in r.stderr
    assert lock.read_text().split()[0] == "300"  # a refused job never removes the holder's lock
    _state_file = slurm / "jobs" / "300"
    _state_file.write_text(_state_file.read_text().replace("RUNNING", "COMPLETED"))
    r = _takeover(slurm, root, 201)
    assert r.returncode == 0 and "broke stale lock" in (root / "CHAIN.log").read_text()
    assert not lock.exists()
    lock.mkdir()  # a lock directory left by the first version is cleared too
    (lock / "owner").write_text("300\n")
    assert _takeover(slurm, root, 202).returncode == 0 and not lock.exists()


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
                            "\ntry:\n h.job_state(200)\n print('NOERR')\n"
                            "except h.QueryError:\n print('QUERYERROR')".format(str(HERE.parent))],
                           env=_env(slurm, PATH=env_path, CHAIN_SLURM_PATIENCE="0"), capture_output=True, text=True,
                           timeout=30)
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


# ── review additions (28 Nov 2026) ─────────────────────────────────────────────────────────────────────


def test_ended_writer_still_listed_by_squeue_reads_as_gone(slurm, root):
    """Real squeue lists an ended job for MinJobAge with its terminal state: that is 'gone', not a 1.5 h wait."""
    for st in ("TIMEOUT", "CANCELLED", "NODE_FAIL", "FAILED"):
        _job(slurm, 200, st)
        (root / "CHAIN.writer").write_text("200 host t\n")
        t0 = time.time()
        r = _takeover(slurm, root, 201, "--timeout", "30")
        assert r.returncode == 0 and time.time() - t0 < 5, (st, r.stderr)
    assert not (slurm / "scancel.log").exists()


def test_slurm_outage_is_waited_out(slurm, root):
    """A short slurmctld outage must not refuse the successor (which would end the chain)."""
    w = _start_writer(root, period=0.5)
    try:
        _job(slurm, 200, "RUNNING", pid=w.pid)
        (root / "CHAIN.writer").write_text("200 host t\n")
        (slurm / "down").write_text("")
        p = subprocess.Popen([sys.executable, str(HANDOVER), "takeover", "--root", str(root), "--timeout", "30"],
                             env=_env(slurm, SLURM_JOB_ID="201", CHAIN_SLURM_PATIENCE="60"),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        time.sleep(4)
        assert p.poll() is None and w.poll() is None  # still waiting, nobody cancelled, nothing refused
        (slurm / "down").unlink()
        assert p.wait(timeout=40) == 0, p.stderr.read()
        assert _state(slurm, 200) == "CANCELLED" and (root / "CHAIN.writer").read_text().split()[0] == "201"
        # past the patience the answer is a refusal, never "gone"
        _job(slurm, 201, "RUNNING")
        (slurm / "down").write_text("")
        r = _takeover(slurm, root, 202, CHAIN_SLURM_PATIENCE="1")
        assert r.returncode == 3 and (root / "CHAIN.writer").read_text().split()[0] == "201"
    finally:
        w.kill()


def test_writer_requeued_during_the_wait_is_cancelled_at_once(slurm, root):
    w = _start_writer(root, period=1000)
    try:
        _job(slurm, 200, "RUNNING", pid=w.pid)
        (root / "CHAIN.writer").write_text("200 host t\n")
        p = subprocess.Popen([sys.executable, str(HANDOVER), "takeover", "--root", str(root), "--timeout", "600"],
                             env=_env(slurm, SLURM_JOB_ID="201"))
        time.sleep(1)
        f = slurm / "jobs" / "200"
        f.write_text(f.read_text().replace("state=RUNNING", "state=PENDING"))  # node failure, requeue
        assert p.wait(timeout=15) == 0
        assert _state(slurm, 200) == "CANCELLED" and "became PENDING" in (root / "CHAIN.log").read_text()
    finally:
        w.kill()


def test_yield_when_the_writer_is_young(slurm, root, tmp_path):
    """Churn guard: a successor that starts while the writer is young requeues itself and trains nothing."""
    job = tmp_path / "job.sbatch"
    job.write_text("#!/bin/bash\n")
    (slurm / "jobs" / "200").write_text("state=RUNNING\ncommand={} a1\nelapsed=10:00\n".format(job))
    (slurm / "jobs" / "201").write_text("state=RUNNING\ncommand={} a1\n".format(job))
    (root / "CHAIN.writer").write_text("200 host t\n")
    (root / "CHAIN.overlap").write_text("0 3600\n")  # begin delay 0, min writer age 1 h
    r = _takeover(slurm, root, 201)
    assert r.returncode == 10, r.stderr
    assert _state(slurm, 200) == "RUNNING" and (root / "CHAIN.writer").read_text().split()[0] == "200"
    assert not (root / ".chain_takeover.lock").exists()
    new = [p for p in (slurm / "jobs").iterdir() if p.name not in ("200", "201")]
    assert len(new) == 1
    d = dict(l.split("=", 1) for l in new[0].read_text().splitlines())
    assert d["begin"] == "now+3000" and d["dependency"] == "" and d["command"].strip() == "{} a1".format(job)
    # an old enough writer is taken over as usual
    (slurm / "jobs" / "200").write_text("state=RUNNING\ncommand={} a1\nelapsed=2:00:00\n".format(job))
    r = _takeover(slurm, root, 201, "--timeout", "1")
    assert r.returncode == 0 and _state(slurm, 200) == "CANCELLED"
    # the shell wrapper turns the yield into a clean exit
    (slurm / "jobs" / "200").write_text("state=RUNNING\ncommand={} a1\nelapsed=10:00\n".format(job))
    (root / "CHAIN.writer").write_text("200 host t\n")
    sh = subprocess.run(["bash", "-c", 'source "$T/chain_segment.sh"; chain_takeover "$R"; echo TRAINED'],
                        env=_env(slurm, SLURM_JOB_ID="201", T=str(HERE.parent), R=str(root), CHAIN_PY=sys.executable),
                        capture_output=True, text=True)
    assert sh.returncode == 0 and "TRAINED" not in sh.stdout and "yield" in sh.stdout


def test_resolve_refuses_a_segment_that_is_not_the_writer(slurm, root):
    """A body without chain_takeover (old pre-submitted segment) must not train beside the live writer."""
    _job(slurm, 200, "RUNNING")
    (root / "CHAIN.writer").write_text("200 host t\n")
    env = _env(slurm, SLURM_JOB_ID="150", T=str(HERE.parent), R=str(root), CHAIN_PY=sys.executable,
               KERNEL_GUARD_PY=sys.executable)
    script = 'source "$T/chain_segment.sh"; chain_kernel_guard() { :; }; chain_resolve "$R" 100000; echo TRAINED'
    sh = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert sh.returncode == 3 and "TRAINED" not in sh.stdout and "second writer" in sh.stderr
    _job(slurm, 200, "TIMEOUT")
    sh = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert sh.returncode == 0 and "TRAINED" in sh.stdout, sh.stderr


def test_fence_stops_a_segment_that_lost_the_chain(slurm, root, tmp_path):
    """Second line of defence: when CHAIN.writer names another job, this segment kills itself."""
    _job(slurm, 200, "RUNNING")
    (root / "CHAIN.writer").write_text("200 host t restart=0\n")
    body = ('"$PY" "$H" fence --root "$R" --pid $$ --every 0.2 --grace 0.1 & sleep 100 & wait')
    p = subprocess.Popen(["bash", "-c", body], start_new_session=True,
                         env=_env(slurm, SLURM_JOB_ID="200", PY=sys.executable, H=str(HANDOVER), R=str(root)))
    try:
        time.sleep(1.5)
        assert p.poll() is None  # still the writer: nothing happens
        (root / "CHAIN.writer").write_text("201 host t restart=0\n")
        assert p.wait(timeout=10) != 0
        assert "200" in (slurm / "scancel.log").read_text() and "fence:" in (root / "CHAIN.log").read_text()
    finally:
        if p.poll() is None:
            os.killpg(p.pid, 9)


def test_fence_tells_a_requeued_copy_from_the_old_one(slurm, root):
    """Same job id after a requeue: the old copy (restart 0) is fenced once the new copy (restart 1) took over."""
    _job(slurm, 200, "RUNNING")
    (root / "CHAIN.writer").write_text("200 host t restart=0\n")
    r = _takeover(slurm, root, 200, "--requeue-wait", "0.2", SLURM_RESTART_COUNT="1")
    assert r.returncode == 0 and "restart=1" in (root / "CHAIN.writer").read_text()
    body = '"$PY" "$H" fence --root "$R" --pid $$ --every 0.2 --grace 0.1 & sleep 100 & wait'
    p = subprocess.Popen(["bash", "-c", body], start_new_session=True,
                         env=_env(slurm, SLURM_JOB_ID="200", SLURM_RESTART_COUNT="0", PY=sys.executable,
                                  H=str(HANDOVER), R=str(root)))
    try:
        assert p.wait(timeout=10) != 0
    finally:
        if p.poll() is None:
            os.killpg(p.pid, 9)


def test_keeper_requeues_a_lost_successor_once(slurm, root, tmp_path):
    """Liveness: the successor refused or was cancelled; the running writer queues a new one, and only one."""
    job = tmp_path / "job.sbatch"
    job.write_text("#!/bin/bash\n")
    (slurm / "jobs" / "200").write_text("state=RUNNING\ncommand={} a1\n".format(job))
    (root / "CHAIN.writer").write_text("200 host t restart=0\n")
    chain_state.main(["log", "--root", str(root), "x resolve action=resume run_id=run1 expect_step=1000 end"])
    body = ('"$PY" "$H" fence --root "$R" --pid $$ --every 0.1 --keeper-every 0.3 --begin-delay 0 '
            '--job-file "$J" -- a1 & sleep 100 & wait')
    p = subprocess.Popen(["bash", "-c", body], start_new_session=True,
                         env=_env(slurm, SLURM_JOB_ID="200", PY=sys.executable, H=str(HANDOVER), R=str(root),
                                  J=str(job)))
    try:
        time.sleep(3)
        new = [q for q in (slurm / "jobs").iterdir() if q.name != "200"]
        assert len(new) == 1 and "keeper:" in (root / "CHAIN.log").read_text()
        # the successor started and sits in its takeover (RUNNING, same command): still no second one
        new[0].write_text(new[0].read_text().replace("state=PENDING", "state=RUNNING"))
        time.sleep(1.5)
        assert len([q for q in (slurm / "jobs").iterdir() if q.name != "200"]) == 1
        # it died (refusal): the keeper queues a replacement
        new[0].write_text(new[0].read_text().replace("state=RUNNING", "state=FAILED"))
        time.sleep(1.5)
        assert len([q for q in (slurm / "jobs").iterdir() if q.name != "200"]) == 2
    finally:
        os.killpg(p.pid, 9)


LOCK_WORKER = r"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
import chain_handover as h
lock, me, shared, n = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
os.environ["SLURM_JOB_ID"] = me
got = 0
for _ in range(n):
    try:
        h.take_lock(lock, me)
    except h.cs.Refuse:
        continue
    got += 1
    fd = os.open(shared, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # fails if another holder is inside
    time.sleep(0.002)
    os.close(fd); os.unlink(shared)
    if got % 2:
        h.release_lock(lock, me)      # half the time the holder "dies" and leaves a stale lock behind
        continue
    open(lock, "w").write("999 dead" + chr(10))  # hand the lock to a job that Slurm does not list
print("OK", got)
"""


def test_lock_never_has_two_holders_under_contention(slurm, root, tmp_path):
    """8 starters race for one lock that keeps going stale (owner = a dead job): never two inside."""
    lock, shared = str(tmp_path / "L.lock"), str(tmp_path / "inside")
    (tmp_path / "L.lock").write_text("999 dead\n")  # job 999 is unknown to Slurm = dead
    for j in range(301, 309):
        _job(slurm, j, "RUNNING")  # a racer's own lock is respected; only the locks left to job 999 are stale
    ps = [subprocess.Popen([sys.executable, "-c", LOCK_WORKER, str(HERE.parent), lock, str(j), shared, "25"],
                           env=_env(slurm), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
          for j in range(301, 309)]
    outs = [p.communicate(timeout=240) for p in ps]
    assert all(p.returncode == 0 for p in ps), [o[1][-400:] for o in outs if o[1]]
    assert sum(int(o[0].split()[1]) for o in outs) > 20
