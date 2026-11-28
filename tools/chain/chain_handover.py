#!/usr/bin/env python3
"""Overlapped chain segments: queue the next segment early, hand over at a checkpoint, keep exactly one writer.

Why (28 Nov 2026, thread "Speed up chained runs on MN5 and Leonardo"): an `afterany` successor only starts to
compete for nodes once its predecessor ends, so on MareNostrum 5 and Leonardo each segment pays the full queue
wait again. In overlap mode the successor is submitted WITHOUT a dependency (optionally with `--begin` delayed),
waits in the queue while the predecessor trains, and when it starts it takes the run over:

    takeover --root R          (job start, before chain_resolve)
      1. take the lock  R/.chain_takeover.lock  (mkdir, atomic on GPFS/Lustre); a lock left by a job that is no
         longer in the queue is broken, a lock held by a live job refuses (exit 3);
      2. read the writer  R/CHAIN.writer  (job id of the segment that last trained). None, self, or not in the queue:
         go to 5;
      3. writer PENDING (requeued after a node failure, or a held job): scancel it, so it can never start as a
         second writer; go to 5;
      4. writer RUNNING: wait (inside this job, not an agent poll) until the chain's last.ckpt holds a HIGHER
         global_step than at our start and has been unchanged for --stable seconds; or the writer leaves the queue
         by itself; or --timeout passes (then --on-timeout kill|refuse). Then `scancel <writer>` (plain SIGTERM,
         never --signal=USR1: Lightning's SLURM plugin answers USR1 by REQUEUEING the job, which would bring back
         a second writer later) and wait until Slurm no longer lists it in any running state (--kill-wait; past
         it: refuse, exit 3, nothing trained);
      5. write R/CHAIN.writer = this job, release the lock, exit 0. chain_resolve then reads last.ckpt as usual.

    next --root R --job-file F [--begin-delay S]    (after chain_resolve, like chain_next)
      queues F once more with NO dependency (sbatch --begin=now+S when S > 0), unless a job running the same
      command is already pending, CHAIN.stop exists, or the chain stalled (chain_state next-check).

The one-writer guarantee rests on Slurm's own job state, read with squeue (sacct as a fallback); a query
error is never read as "gone". Cost of one handover: the successor's GPUs idle until the predecessor's next
checkpoint (at most one save interval) plus the successor's start-up. Python 3.6, stdlib only.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_state as cs  # noqa: E402

WRITER = "CHAIN.writer"
LOCK = ".chain_takeover.lock"
ALIVE = {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "STOPPED", "RESIZING", "SIGNALING", "STAGE_OUT",
         "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED"}
PENDING = {"PENDING"}


class QueryError(Exception):
    pass


def _me():
    return os.environ.get("SLURM_JOB_ID", "")


def job_state(jid):
    """Slurm state of job jid: a state string, or '' when Slurm no longer lists it as queued or running.
    Raises QueryError when Slurm cannot be asked (never read as 'gone')."""
    for attempt in range(5):
        try:
            r = subprocess.run(["squeue", "-h", "-j", str(jid), "-o", "%T"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            r = None
        if r is not None and r.returncode == 0:
            st = r.stdout.strip().split("\n")[0].strip()
            if st:
                return st
            return _sacct_confirms_ended(jid)
        if r is not None and "Invalid job id" in (r.stderr or ""):
            return ""  # purged from slurmctld: long finished
        time.sleep(min(30, 2 ** attempt))
    raise QueryError("squeue -j {} failed 5 times".format(jid))


def _sacct_confirms_ended(jid):
    """squeue printed nothing: confirm with sacct that the job really ended (not a transient empty answer)."""
    try:
        r = subprocess.run(["sacct", "-n", "-X", "-P", "-j", str(jid), "-o", "State"], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    st = r.stdout.strip().split("\n")[0].split()[0] if r.returncode == 0 and r.stdout.strip() else ""
    st = st.upper()
    if st in ALIVE or st in PENDING:
        return st
    return ""


def _scancel(jid):
    r = subprocess.run(["scancel", str(jid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return r.returncode == 0, (r.stderr or "").strip()


def _read_writer(root):
    p = os.path.join(root, WRITER)
    if not os.path.isfile(p):
        return ""
    with open(p) as fh:
        words = fh.read().split()
    return words[0] if words else ""


def _write_writer(root, jid):
    p = os.path.join(root, WRITER)
    tmp = p + ".tmp.{}".format(os.getpid())
    with open(tmp, "w") as fh:
        fh.write("{} {} {}\n".format(jid, os.uname()[1].split(".")[0], cs._now()))
    os.replace(tmp, p)


def _chain_last(root):
    """The last.ckpt the chain resumes from: the manifest's run, else the only run with a last.ckpt, else None."""
    m = cs._read_manifest(root)
    rid = m.get("run_id")
    if rid:
        return os.path.join(root, rid, "last.ckpt")
    ignored = set(m.get("ignored_runs", []))
    runs = [r for r, has in cs.runs_with_checkpoints(root).items() if has and r not in ignored]
    return os.path.join(root, runs[0], "last.ckpt") if len(runs) == 1 else None


def _snapshot(path):
    """(step, mtime, size) of a complete checkpoint, or None (absent or not readable yet: still being written)."""
    if not path or not os.path.isfile(path):
        return None
    try:
        st = os.stat(path)
        return cs.checkpoint_step(path), st.st_mtime, st.st_size
    except Exception:  # partial zip while the writer saves
        return None


def _lock(root, me):
    lock = os.path.join(root, LOCK)
    for _ in range(2):
        try:
            os.mkdir(lock)
            with open(os.path.join(lock, "owner"), "w") as fh:
                fh.write(me + "\n")
            return lock
        except FileExistsError:
            try:
                with open(os.path.join(lock, "owner")) as fh:
                    owner = fh.read().strip()
            except OSError:
                owner = ""
            if owner and owner != me and job_state(owner):
                raise cs.Refuse("takeover lock {} is held by live job {}: another segment is taking over; "
                                "this one stops (nothing trained)".format(lock, owner))
            cs._log(root, "takeover: broke stale lock of job {}".format(owner or "?"))
            shutil.rmtree(lock, ignore_errors=True)
    raise cs.Refuse("could not take the lock {}".format(lock))


def cmd_takeover(a):
    root, me = a.root, _me()
    if not me:
        print("[handover] not inside a Slurm job: nothing to take over")
        return 0
    cs._read_manifest(root)  # refuses on an undeclared chain, like resolve
    if os.path.exists(os.path.join(root, cs.STOP)):
        print("[handover] CHAIN.stop exists: no takeover (chain_resolve will stop this segment)")
        return 0
    lock = _lock(root, me)
    try:
        prev = _read_writer(root)
        if not prev or prev == me:
            cs._log(root, "takeover: no other writer (writer={})".format(prev or "none"))
        else:
            st = job_state(prev)
            if not st:
                cs._log(root, "takeover: writer {} is no longer queued".format(prev))
            elif st in PENDING:
                ok, err = _scancel(prev)
                cs._log(root, "takeover: writer {} was PENDING (requeued?): scancel ok={} {}".format(prev, ok, err))
                if not ok:
                    raise cs.Refuse("cannot cancel pending writer {}: {}".format(prev, err))
            else:
                _wait_and_stop(a, root, prev, st)
        _write_writer(root, me)
        cs._log(root, "takeover: writer is now {}".format(me))
        print("[handover] job {} is the chain's only writer".format(me))
    finally:
        shutil.rmtree(lock, ignore_errors=True)
    return 0


def _wait_and_stop(a, root, prev, st):
    last = _chain_last(root)
    base = _snapshot(last)
    base_step = base[0] if base else -1
    cs._log(root, "takeover: writer {} is {}; waiting for a checkpoint past step {} (timeout {} s)".format(
        prev, st, base_step, a.timeout))
    t0, seen, reason = time.time(), None, "timeout"
    while time.time() - t0 < a.timeout:
        if not job_state(prev):
            reason = "ended"
            break
        last = last or _chain_last(root)
        snap = _snapshot(last)
        if snap and snap[0] > base_step:
            if seen and seen[:3] == snap and time.time() - seen[3] >= a.stable:
                reason = "checkpoint"
                break
            if not seen or seen[:3] != snap:
                seen = snap + (time.time(),)
        time.sleep(a.poll)
    if reason == "ended":
        cs._log(root, "takeover: writer {} ended by itself".format(prev))
        return
    if reason == "timeout" and a.on_timeout == "refuse":
        raise cs.Refuse("writer {} wrote no new checkpoint in {} s; on-timeout=refuse: this segment stops and the "
                        "writer keeps training".format(prev, a.timeout))
    step = seen[0] if reason == "checkpoint" else base_step
    ok, err = _scancel(prev)
    cs._log(root, "takeover: scancel {} after {} (step {}) ok={} {}".format(prev, reason, step, ok, err))
    if not ok and job_state(prev):
        raise cs.Refuse("scancel {} failed: {}; two writers would follow, this segment stops".format(prev, err))
    t1 = time.time()
    while job_state(prev):
        if time.time() - t1 > a.kill_wait:
            raise cs.Refuse("writer {} still listed by Slurm {} s after scancel; refusing to start a second "
                            "writer".format(prev, a.kill_wait))
        time.sleep(a.poll)
    final = _snapshot(last)
    if reason == "checkpoint" and (not final or final[0] < step):
        raise cs.Refuse("after stopping {}, {} reads step {} (< {} seen before the cancel): the save was cut; "
                        "inspect before resuming".format(prev, last, final and final[0], step))
    cs._log(root, "takeover: writer {} gone {:.0f} s after scancel; last.ckpt step {}".format(
        prev, time.time() - t1, final and final[0]))


def cmd_next(a):
    root = a.root
    me = _me()
    if not me:
        print("[handover] not inside a Slurm job: no successor submitted")
        return 0
    if os.path.exists(os.path.join(root, cs.STOP)) or _stalled(root, a.max_same_start):
        sys.stderr.write("[handover] WARNING: no successor submitted (stop file or stalled chain)\n")
        return 0
    cmd = [a.job_file] + list(a.job_args)
    cmd_str = " ".join(cmd)
    r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "PENDING", "-o", "%i|%o"],
                       stdout=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        sys.stderr.write("[handover] WARNING: squeue failed; no successor submitted (a duplicate would be worse)\n")
        cs._log(root, "overlap-next NOT submitted: squeue failed")
        return 0
    same = [line.partition("|")[0] for line in (r.stdout or "").splitlines()
            if line.partition("|")[2].strip() == cmd_str]
    if len(same) >= a.max_pending:
        print("[handover] successor already queued: job(s) {} (max pending {})".format(",".join(same), a.max_pending))
        cs._log(root, "overlap-next already queued job={}".format(",".join(same)))
        return 0
    env = {k: v for k, v in os.environ.items() if not (k.startswith("SLURM_") and k != "SLURM_CONF")}
    sb = ["sbatch", "--parsable"]
    if a.begin_delay > 0:
        sb.append("--begin=now+{}".format(int(a.begin_delay)))
    r = subprocess.run(sb + cmd, env=env, cwd=os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        sys.stderr.write("[handover] WARNING: sbatch failed: {}\n".format(r.stderr.strip()))
        cs._log(root, "overlap-next NOT submitted: sbatch failed")
        return 0
    jid = r.stdout.strip().split(";")[0]
    print("[handover] successor queued: job {} (no dependency, begin delay {} s)".format(jid, int(a.begin_delay)))
    cs._log(root, "overlap-next queued job={} begin_delay={} (no dependency)".format(jid, int(a.begin_delay)))
    return 0


def _stalled(root, max_same):
    starts = cs.resolved_start_steps(root)
    if not starts:
        return False
    same = 0
    for s in reversed(starts):
        if s != starts[-1]:
            break
        same += 1
    return same >= max_same


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("takeover", help="job start: become the chain's only writer")
    s.add_argument("--root", required=True)
    s.add_argument("--timeout", type=float, default=float(os.environ.get("CHAIN_TAKEOVER_TIMEOUT", 5400)),
                   help="max seconds to wait for the writer's next checkpoint (default 5400 = 1.5 h)")
    s.add_argument("--on-timeout", choices=("kill", "refuse"), default="kill")
    s.add_argument("--stable", type=float, default=float(os.environ.get("CHAIN_TAKEOVER_STABLE", 90)),
                   help="seconds a new last.ckpt must stay unchanged before the writer is cancelled")
    s.add_argument("--poll", type=float, default=float(os.environ.get("CHAIN_TAKEOVER_POLL", 30)))
    s.add_argument("--kill-wait", type=float, default=float(os.environ.get("CHAIN_TAKEOVER_KILL_WAIT", 900)))
    s = sub.add_parser("next", help="after resolve: queue the same job file again, no dependency")
    s.add_argument("--root", required=True)
    s.add_argument("--job-file", required=True)
    s.add_argument("--begin-delay", type=float, default=0)
    s.add_argument("--max-same-start", type=int, default=3)
    s.add_argument("--max-pending", type=int, default=1,
                   help="queue a successor only while fewer than this many jobs with the same command are pending "
                        "(1 for a chain; the pool size for pilot jobs)")
    s.add_argument("job_args", nargs="*")
    a = p.parse_args(argv)
    if not a.cmd:
        p.error("a command is required")
    a.root = os.path.abspath(a.root)
    try:
        return {"takeover": cmd_takeover, "next": cmd_next}[a.cmd](a)
    except (cs.Refuse, QueryError) as e:
        sys.stderr.write("[handover] REFUSED: {}\n".format(e))
        try:
            cs._log(a.root, "{} REFUSED: {}".format(a.cmd, e))
        except OSError:
            pass
        return cs.REFUSE


if __name__ == "__main__":
    sys.exit(main())
