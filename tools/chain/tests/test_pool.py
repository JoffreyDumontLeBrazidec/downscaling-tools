"""Pilot pool (chain_pool.py, pilot_job.sh): pilots go to the chain that needs a writer, never two to one chain."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import chain_state  # noqa: E402

POOL_PY = HERE.parent / "chain_pool.py"
PILOT = HERE.parent / "pilot_job.sh"
FAST = {"CHAIN_TAKEOVER_POLL": "0.1", "CHAIN_TAKEOVER_STABLE": "0.3", "CHAIN_TAKEOVER_KILL_WAIT": "5"}


def _env(slurm, **extra):
    env = dict(os.environ, FAKE_SLURM_DIR=str(slurm), USER="tester", CHAIN_PY=sys.executable,
               PATH="{}:{}".format(HERE / "fake_slurm", os.environ["PATH"]), **FAST)
    env.update(extra)
    return env


def _job(slurm, jid, state, left=None, command="/x/pilot.sbatch"):
    lines = ["state={}".format(state), "command={}".format(command), "dependency=", "begin="]
    if left:
        lines.append("left={}".format(left))
    (slurm / "jobs" / str(jid)).write_text("\n".join(lines) + "\n")


def _ckpt(root, step):
    import io, pickle, zipfile
    (root / "run1").mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pickle.dump({"global_step": step}, buf, protocol=2)
    with zipfile.ZipFile(str(root / "run1" / "last.ckpt"), "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())


def _pick(slurm, pool, me, *args):
    r = subprocess.run([sys.executable, str(POOL_PY), "pick", "--pool", str(pool)] + list(args),
                       env=_env(slurm, SLURM_JOB_ID=str(me)), capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return dict(l.split("=", 1) for l in r.stdout.splitlines())


@pytest.fixture
def slurm(tmp_path):
    d = tmp_path / "slurm"
    (d / "jobs").mkdir(parents=True)
    return d


@pytest.fixture
def pool(tmp_path):
    lines = []
    for name, step in (("a", 1000), ("b", 2000), ("c", 3000)):
        r = tmp_path / name
        chain_state.main(["init", "--root", str(r)])
        _ckpt(r, step)
        chain_state.main(["adopt", "--root", str(r), "--run-id", "run1"])
        lines.append("{}\t3000\t/bin/true".format(r))
    p = tmp_path / "pool" / "POOL.tsv"
    p.parent.mkdir()
    p.write_text("# test pool\n" + "\n".join(lines) + "\n")
    return p


def test_idle_chain_first_then_least_time_left(slurm, pool, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    # c is done (3000/3000); a has a live writer with 5 h left; b is idle -> b
    _job(slurm, 50, "RUNNING", left="05:00:00")
    (a / "CHAIN.writer").write_text("50 h t\n")
    out = _pick(slurm, pool, 301)
    assert out["POOL_ACTION"] == "train" and out["POOL_ROOT"] == str(b) and out["POOL_ACTIVE"] == "2"
    assert (b / "CHAIN.claim").read_text().split()[0] == "301"
    # pilot 301 is now live (claim) on b; next pilot: b is taken, a has the least time left -> a
    _job(slurm, 301, "RUNNING")
    out = _pick(slurm, pool, 302)
    assert out["POOL_ROOT"] == str(a)
    # with a takeover threshold below a's 5 h left, nothing is picked
    _job(slurm, 302, "RUNNING")
    (a / "CHAIN.claim").unlink()
    out = _pick(slurm, pool, 303, "--takeover-below", "3600")
    assert out["POOL_ACTION"] == "none" and out["POOL_ACTIVE"] == "2"


def test_dead_claim_does_not_block(slurm, pool, tmp_path):
    b = tmp_path / "b"
    _job(slurm, 77, "CANCELLED")
    (b / "CHAIN.claim").write_text("77 t\n")
    a = tmp_path / "a"
    _job(slurm, 50, "RUNNING", left="10:00:00")
    (a / "CHAIN.writer").write_text("50 h t\n")
    out = _pick(slurm, pool, 301)
    assert out["POOL_ROOT"] == str(b)


def test_all_done_means_no_pilot(slurm, pool, tmp_path):
    for n in ("a", "b"):
        _ckpt(tmp_path / n, 3000)
    out = _pick(slurm, pool, 301)
    assert out["POOL_ACTION"] == "none" and out["POOL_ACTIVE"] == "0"


def test_pilot_job_queues_up_to_active_pilots_and_runs_body(slurm, tmp_path):
    # two idle chains whose body records which root it got and whether it saw CHAIN_POOL_PILOT
    body = tmp_path / "body.sh"
    body.write_text("#!/bin/bash\necho \"$CHAIN_POOL_ROOT $CHAIN_POOL_PILOT\" >> {}/ran.txt\n".format(tmp_path))
    lines = []
    for name in ("a", "b"):
        r = tmp_path / name
        chain_state.main(["init", "--root", str(r)])
        lines.append("{}\t3000\t{} x".format(r, body))
    pool = tmp_path / "pool" / "POOL.tsv"
    pool.parent.mkdir()
    pool.write_text("\n".join(lines) + "\n")
    job = tmp_path / "pilot.sbatch"
    job.write_text("#!/bin/bash\n")
    for me in (201, 202, 203):
        _job(slurm, me, "RUNNING", command="{} {}".format(job, pool))
        r = subprocess.run(["bash", str(PILOT), str(pool)], env=_env(slurm, SLURM_JOB_ID=str(me),
                           CHAIN_TOOLS=str(HERE.parent), SLURM_SUBMIT_DIR=str(tmp_path)),
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
    ran = (tmp_path / "ran.txt").read_text().split("\n")
    # 201 -> a (both idle, a first), 202 -> b (a taken by live 201), 203 -> the one with least time left
    assert ran[0] == "{} 1".format(tmp_path / "a") and ran[1] == "{} 1".format(tmp_path / "b")
    pending = [f for f in (slurm / "jobs").iterdir() if "state=PENDING" in f.read_text()]
    assert len(pending) == 2  # never more pending pilots than active chains
    for f in pending:
        assert "dependency=\n" in f.read_text()


def test_chain_pool_file_silences_body_successor(slurm, tmp_path):
    r = tmp_path / "a"
    chain_state.main(["init", "--root", str(r)])
    (r / "CHAIN.pool").write_text("pool\n")
    out = subprocess.run(["bash", "-c", "source {}/chain_segment.sh; chain_next_auto {}".format(HERE.parent, r)],
                         env=_env(slurm, SLURM_JOB_ID="9"), capture_output=True, text=True)
    assert "queues nothing" in out.stdout and not list((slurm / "jobs").iterdir())


def test_chain_overlap_file_selects_overlap_with_its_delay(slurm, tmp_path):
    r = tmp_path / "a"
    chain_state.main(["init", "--root", str(r)])
    chain_state.main(["log", "--root", str(r), "x resolve action=fresh run_id=- expect_step=0 end"])
    (r / "CHAIN.overlap").write_text("43200 MN5 delay\n")
    job = tmp_path / "job.sbatch"
    job.write_text("#!/bin/bash\n")
    _job(slurm, 9, "RUNNING", command=str(job))
    out = subprocess.run(["bash", "-c", "source {}/chain_segment.sh; chain_next_auto {}".format(HERE.parent, r)],
                         env=_env(slurm, SLURM_JOB_ID="9"), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    new = [f for f in (slurm / "jobs").iterdir() if f.name != "9"]
    assert len(new) == 1
    d = dict(l.split("=", 1) for l in new[0].read_text().splitlines())
    assert d["begin"] == "now+43200" and d["dependency"] == ""
