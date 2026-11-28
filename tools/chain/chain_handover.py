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

Review additions (28 Nov 2026, thread "Review the overlap chains for MN5 and Leonardo"):
    lock      a FILE taken with link(2) (owner written before the link, so never empty); a stale lock is removed
              only under a second short lock and only if it still names the dead owner.
    states    a terminal state printed by squeue (CANCELLED, TIMEOUT, ...) reads as gone; a Slurm outage is waited
              out for CHAIN_SLURM_PATIENCE seconds (default 1800) before a refusal.
    yield     takeover --min-writer-age S (or the 2nd word of R/CHAIN.overlap): a writer younger than S is left
              alone; this job queues itself again with --begin and exits 10 (the shell turns it into exit 0).
    check     chain_resolve calls it when R/CHAIN.writer exists: a segment that is not the writer refuses while the
              writer is alive (covers a body that never called takeover, e.g. an old pre-submitted segment).
    fence     background loop inside the segment: if R/CHAIN.writer stops naming this job (and this restart count),
              the segment cancels and kills itself. With --job-file it is also the KEEPER: every --keeper-every
              seconds it makes sure a successor is queued, so a refused or lost successor does not end the chain.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_state as cs  # noqa: E402

WRITER = "CHAIN.writer"
LOCK = ".chain_takeover.lock"
OVERLAP = "CHAIN.overlap"
YIELD = 10
ALIVE = {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "STOPPED", "RESIZING", "SIGNALING", "STAGE_OUT",
         "REQUEUED", "REQUEUE_FED"}
# not running now, but Slurm may start it again: a takeover cancels it
PENDING = {"PENDING", "REQUEUE_HOLD", "SPECIAL_EXIT", "RESV_DEL_HOLD"}
# squeue lists an ended job for MinJobAge seconds: these read as gone. Anything unknown reads as alive.
# NODE_FAIL: Slurm gave the job up; a process on a cut-off node may still write, which only the fence can stop.
TERMINAL = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
            "TIMEOUT", "REVOKED"}


class QueryError(Exception):
    pass


def _me():
    return os.environ.get("SLURM_JOB_ID", "")


def _restart():
    return os.environ.get("SLURM_RESTART_COUNT", "0") or "0"


def _patience():
    return float(os.environ.get("CHAIN_SLURM_PATIENCE", 1800))


def job_state(jid):
    """Slurm state of job jid: a state string, or '' when the job ended (not listed, or listed in a terminal
    state). Raises QueryError when Slurm cannot be asked for CHAIN_SLURM_PATIENCE seconds (never read as 'gone')."""
    t0, attempt = time.time(), 0
    while True:
        try:
            r = subprocess.run(["squeue", "-h", "-j", str(jid), "-o", "%T"], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            r = None
        if r is not None and r.returncode == 0:
            st = r.stdout.strip().split("\n")[0].strip().upper()
            if st:
                return "" if st.split()[0].rstrip("+") in TERMINAL else st
            return _sacct_confirms_ended(jid)
        if r is not None and "Invalid job id" in (r.stderr or ""):
            return ""  # purged from slurmctld: long finished
        if time.time() - t0 >= _patience():
            raise QueryError("squeue -j {} failed for {:.0f} s".format(jid, time.time() - t0))
        time.sleep(min(60, 2 ** attempt))
        attempt += 1


def _slurm_seconds(v):
    """[D-]HH:MM:SS, MM:SS or SS -> seconds; None if not a time."""
    v = (v or "").strip()
    try:
        days = 0
        if "-" in v:
            d, v = v.split("-", 1)
            days = int(d)
        nums = [int(x) for x in v.split(":")]
    except ValueError:
        return None
    while len(nums) < 3:
        nums.insert(0, 0)
    return days * 86400 + nums[0] * 3600 + nums[1] * 60 + nums[2]


def job_elapsed(jid):
    """Seconds a running job has run (squeue %M), or None if unknown."""
    try:
        r = subprocess.run(["squeue", "-h", "-j", str(jid), "-o", "%M"], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _slurm_seconds(r.stdout.strip().split("\n")[0]) if r.returncode == 0 and r.stdout.strip() else None


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


def _read_identity(root):
    """(job id, restart count) of the chain's writer, or ('', '')."""
    p = os.path.join(root, WRITER)
    try:
        with open(p) as fh:
            words = fh.read().split()
    except OSError:
        return "", ""
    if not words:
        return "", ""
    rs = [w.split("=", 1)[1] for w in words[1:] if w.startswith("restart=")]
    return words[0], (rs[0] if rs else "0")


def _read_writer(root):
    return _read_identity(root)[0]


def _write_writer(root, jid):
    p = os.path.join(root, WRITER)
    tmp = p + ".tmp.{}".format(os.getpid())
    with open(tmp, "w") as fh:
        fh.write("{} {} {} restart={}\n".format(jid, os.uname()[1].split(".")[0], cs._now(), _restart()))
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


def _first_word(path):
    try:
        with open(path) as fh:
            w = fh.read().split()
        return w[0] if w else ""
    except OSError:
        return None  # no such file (or a legacy lock directory)


def _link_new(path, text):
    """Create `path` holding `text`, atomically and only if it does not exist (link(2): safe on GPFS/Lustre/NFS).
    The content is complete before the name appears, so a reader never sees an empty lock."""
    tmp = "{}.{}.{}.{}".format(path, os.uname()[1].split(".")[0], os.getpid(), int(time.time() * 1e6))
    with open(tmp, "w") as fh:
        fh.write(text)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        os.unlink(tmp)


def _break_stale(lock, owner, log):
    """Remove `lock` only if it still names `owner` (judged dead by the caller). Done under a second, short lock,
    so two jobs that judged the same owner dead cannot remove each other's fresh lock."""
    brk = lock + ".break"
    if not _link_new(brk, "{} {}\n".format(_me() or "local", cs._now())):
        try:
            if time.time() - os.stat(brk).st_mtime > 300:  # its holder died inside a few-millisecond section
                os.unlink(brk)
        except OSError:
            pass
        return
    try:
        if os.path.isdir(lock):  # lock directory of the first version
            shutil.rmtree(lock, ignore_errors=True)
        elif _first_word(lock) == owner:
            os.unlink(lock)
            log("broke stale lock of job {}".format(owner or "?"))
    except OSError:
        pass
    finally:
        os.unlink(brk)


def take_lock(lock, me, log=lambda m: None, tries=20):
    """Take the lock file or raise Refuse when a live job holds it."""
    for _ in range(tries):
        if _link_new(lock, "{} {}\n".format(me, cs._now())):
            return lock
        if os.path.isdir(lock):
            owner = _first_word(os.path.join(lock, "owner")) or ""
        else:
            owner = _first_word(lock)
            if owner is None:  # released between our link and our read
                continue
        if owner and owner != me and job_state(owner):
            raise cs.Refuse("lock {} is held by live job {}: another segment is taking over; "
                            "this one stops (nothing trained)".format(lock, owner))
        _break_stale(lock, owner, log)
        time.sleep(0.05)
    raise cs.Refuse("could not take the lock {}".format(lock))


def release_lock(lock, me):
    """Remove the lock only if it is still ours."""
    if _first_word(lock) == me:
        try:
            os.unlink(lock)
        except OSError:
            pass


def _lock(root, me):
    return take_lock(os.path.join(root, LOCK), me, lambda m: cs._log(root, "takeover: " + m))


def _min_age(a, root):
    if a.min_writer_age is not None:
        return a.min_writer_age
    try:
        with open(os.path.join(root, OVERLAP)) as fh:
            w = fh.read().split()
        return float(w[1]) if len(w) > 1 else 0.0
    except (OSError, ValueError):
        return 0.0


def _my_command():
    """[job file, args...] of this job, from scontrol; [] if it cannot be read."""
    try:
        r = subprocess.run(["scontrol", "show", "job", _me()], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    for line in (r.stdout or "").splitlines():
        if line.strip().startswith("Command="):
            words = line.strip()[len("Command="):].split()
            return words if words and os.path.isfile(words[0]) else []
    return []


def _cancel_and_wait_gone(a, root, prev, why):
    ok, err = _scancel(prev)
    cs._log(root, "takeover: scancel {} {} ok={} {}".format(prev, why, ok, err))
    if not ok and job_state(prev):
        raise cs.Refuse("scancel {} failed: {}; two writers would follow, this segment stops".format(prev, err))
    t1 = time.time()
    while job_state(prev):
        if time.time() - t1 > a.kill_wait:
            raise cs.Refuse("writer {} still listed by Slurm {} s after scancel; refusing to start a second "
                            "writer".format(prev, a.kill_wait))
        time.sleep(a.poll)
    return time.time() - t1


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
        prev, prev_restart = _read_identity(root)
        if not prev:
            cs._log(root, "takeover: no other writer (writer=none)")
        elif prev == me:
            cs._log(root, "takeover: no other writer (writer={} restart {} -> {})".format(prev, prev_restart, _restart()))
            if prev_restart != _restart() and a.requeue_wait > 0:
                # this job was requeued: give a copy that may survive on a cut-off node time to fence itself
                _write_writer(root, me)
                cs._log(root, "takeover: requeued job, waiting {:.0f} s for the old copy's fence".format(a.requeue_wait))
                time.sleep(a.requeue_wait)
        else:
            st = job_state(prev)
            if not st:
                cs._log(root, "takeover: writer {} is no longer queued".format(prev))
            elif st in PENDING:
                _cancel_and_wait_gone(a, root, prev, "while {} (requeued?)".format(st))
            else:
                min_age = _min_age(a, root)
                age = job_elapsed(prev) if min_age > 0 else None
                if age is not None and age < min_age:
                    return _yield(a, root, prev, age, min_age)
                _wait_and_stop(a, root, prev, st)
        _write_writer(root, me)
        cs._log(root, "takeover: writer is now {}".format(me))
        print("[handover] job {} is the chain's only writer".format(me))
    finally:
        release_lock(lock, me)
    return 0


def _yield(a, root, prev, age, min_age):
    """The writer is young: leave it alone, queue this job again for when it is old enough, train nothing."""
    delay = max(60, int(min_age - age))
    cmd = _my_command()
    cs._log(root, "takeover: YIELD writer {} ran {:.0f} s < min age {:.0f} s; requeue in {} s".format(
        prev, age, min_age, delay))
    if cmd:
        _submit(root, cmd, delay, max_pending=1, count_running=False)
    else:
        cs._log(root, "takeover: yield could not read this job's command; the writer's keeper must requeue")
    print("[handover] yield: writer {} is young; this job trains nothing".format(prev))
    return YIELD


def _wait_and_stop(a, root, prev, st):
    last = _chain_last(root)
    base = _snapshot(last)
    base_step = base[0] if base else -1
    cs._log(root, "takeover: writer {} is {}; waiting for a checkpoint past step {} (timeout {} s)".format(
        prev, st, base_step, a.timeout))
    t0, seen, reason = time.time(), None, "timeout"
    while time.time() - t0 < a.timeout:
        cur = job_state(prev)
        if not cur:
            reason = "ended"
            break
        if cur in PENDING:  # requeued while we waited: it holds no checkpoint in flight, cancel it now
            _cancel_and_wait_gone(a, root, prev, "when it became {} during the wait".format(cur))
            return
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
    gone = _cancel_and_wait_gone(a, root, prev, "after {} (step {})".format(reason, step))
    final = _snapshot(last)
    if reason == "checkpoint" and (not final or final[0] < step):
        raise cs.Refuse("after stopping {}, {} reads step {} (< {} seen before the cancel): the save was cut; "
                        "inspect before resuming".format(prev, last, final and final[0], step))
    cs._log(root, "takeover: writer {} gone {:.0f} s after scancel; last.ckpt step {}".format(
        prev, gone, final and final[0]))


def cmd_next(a):
    root = a.root
    me = _me()
    if not me:
        print("[handover] not inside a Slurm job: no successor submitted")
        return 0
    if os.path.exists(os.path.join(root, cs.STOP)) or _stalled(root, a.max_same_start):
        sys.stderr.write("[handover] WARNING: no successor submitted (stop file or stalled chain)\n")
        return 0
    _submit(root, [a.job_file] + list(a.job_args), a.begin_delay, a.max_pending, a.count_running)
    return 0


def _submit(root, cmd, begin_delay, max_pending=1, count_running=False):
    """Queue `cmd` with no dependency unless enough jobs with the same command already wait. count_running also
    counts RUNNING jobs other than this one (a successor that is inside its takeover). True if a job was queued."""
    cmd_str = " ".join(cmd)
    states = "PENDING,RUNNING" if count_running else "PENDING"
    try:
        r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", states, "-o", "%i|%o"],
                           stdout=subprocess.PIPE, universal_newlines=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        r = None
    if r is None or r.returncode != 0:
        sys.stderr.write("[handover] WARNING: squeue failed; no successor submitted (a duplicate would be worse)\n")
        cs._log(root, "overlap-next NOT submitted: squeue failed")
        return False
    same = [line.partition("|")[0].strip() for line in (r.stdout or "").splitlines()
            if line.partition("|")[2].strip() == cmd_str]
    same = [j for j in same if j != _me()]
    if len(same) >= max_pending:
        print("[handover] successor already queued: job(s) {} (max pending {})".format(",".join(same), max_pending))
        cs._log(root, "overlap-next already queued job={}".format(",".join(same)))
        return False
    env = {k: v for k, v in os.environ.items() if not (k.startswith("SLURM_") and k != "SLURM_CONF")}
    sb = ["sbatch", "--parsable"]
    if begin_delay > 0:
        sb.append("--begin=now+{}".format(int(begin_delay)))
    r = subprocess.run(sb + cmd, env=env, cwd=os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        sys.stderr.write("[handover] WARNING: sbatch failed: {}\n".format(r.stderr.strip()))
        cs._log(root, "overlap-next NOT submitted: sbatch failed")
        return False
    jid = r.stdout.strip().split(";")[0]
    print("[handover] successor queued: job {} (no dependency, begin delay {} s)".format(jid, int(begin_delay)))
    cs._log(root, "overlap-next queued job={} begin_delay={} (no dependency)".format(jid, int(begin_delay)))
    return True


def cmd_check(a):
    """chain_resolve's guard: refuse to train while CHAIN.writer names another job that Slurm still lists."""
    root, me = a.root, _me()
    prev = _read_writer(root)
    if not me or not prev or prev == me:
        return 0
    st = job_state(prev)
    if st:
        raise cs.Refuse("CHAIN.writer names job {} ({}) and this job {} did not take the chain over "
                        "(no chain_takeover in this body?): refusing to be a second writer".format(prev, st, me))
    return 0


def cmd_fence(a):
    """Runs beside the training (started by chain_fence, in the job's process group). FENCE: when CHAIN.writer names
    another job or another restart of this job, cancel this job and kill its process group. KEEPER (--job-file):
    keep one successor queued."""
    root, me = a.root, _me()
    if not me:
        return 0
    mine = (me, _restart())
    t0, last_keep, kept = time.time(), time.time(), 0
    while True:
        time.sleep(a.every)
        if a.pid:
            try:
                os.kill(a.pid, 0)
            except OSError:
                return 0  # the job body ended
        ident = _read_identity(root)
        if ident[0] and ident != mine:
            cs._log(root, "fence: CHAIN.writer names {} restart {} but this is {} restart {}: stopping this "
                          "segment".format(ident[0], ident[1], mine[0], mine[1]))
            sys.stderr.write("[fence] this segment is no longer the chain's writer: stopping\n")
            _scancel(me)
            time.sleep(a.grace)
            try:
                os.killpg(os.getpgid(a.pid) if a.pid else os.getpgrp(), signal.SIGTERM)
            except OSError:
                pass
            return cs.REFUSE
        if a.job_file and kept < a.keeper_max and time.time() - last_keep >= a.keeper_every:
            last_keep = time.time()
            if not os.path.exists(os.path.join(root, cs.STOP)) and not _stalled(root, a.max_same_start):
                delay = max(0, a.begin_delay - (time.time() - t0))
                if _submit(root, [a.job_file] + list(a.job_args), delay, 1, count_running=True):
                    kept += 1
                    cs._log(root, "keeper: no successor was queued; queued one ({} of {} this segment)".format(
                        kept, a.keeper_max))


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
    s.add_argument("--min-writer-age", type=float, default=None,
                   help="yield (exit 10, requeue this job) when the writer has run less than this many seconds; "
                        "default: 2nd word of <root>/CHAIN.overlap, else 0 = never yield")
    s.add_argument("--requeue-wait", type=float, default=float(os.environ.get("CHAIN_REQUEUE_WAIT", 150)),
                   help="a requeued writer waits this long so a surviving old copy can fence itself")
    s = sub.add_parser("next", help="after resolve: queue the same job file again, no dependency")
    s.add_argument("--root", required=True)
    s.add_argument("--job-file", required=True)
    s.add_argument("--begin-delay", type=float, default=0)
    s.add_argument("--max-same-start", type=int, default=3)
    s.add_argument("--max-pending", type=int, default=1,
                   help="queue a successor only while fewer than this many jobs with the same command are pending "
                        "(1 for a chain; the pool size for pilot jobs)")
    s.add_argument("--count-running", action="store_true",
                   help="also count RUNNING jobs with the same command, other than this one")
    s.add_argument("job_args", nargs="*")
    s = sub.add_parser("check", help="chain_resolve guard: refuse while CHAIN.writer names another live job")
    s.add_argument("--root", required=True)
    s = sub.add_parser("fence", help="background: stop this segment when it is no longer CHAIN.writer; keep a successor queued")
    s.add_argument("--root", required=True)
    s.add_argument("--pid", type=int, default=0, help="the job body's pid; the fence ends with it")
    s.add_argument("--every", type=float, default=float(os.environ.get("CHAIN_FENCE_EVERY", 60)))
    s.add_argument("--grace", type=float, default=float(os.environ.get("CHAIN_FENCE_GRACE", 20)))
    s.add_argument("--job-file", default=None, help="with it the fence is also the keeper")
    s.add_argument("--begin-delay", type=float, default=0)
    s.add_argument("--keeper-every", type=float, default=float(os.environ.get("CHAIN_KEEPER_EVERY", 900)))
    s.add_argument("--keeper-max", type=int, default=8)
    s.add_argument("--max-same-start", type=int, default=3)
    s.add_argument("job_args", nargs="*")
    a = p.parse_args(argv)
    if not a.cmd:
        p.error("a command is required")
    a.root = os.path.abspath(a.root)
    try:
        return {"takeover": cmd_takeover, "next": cmd_next, "check": cmd_check, "fence": cmd_fence}[a.cmd](a)
    except (cs.Refuse, QueryError) as e:
        sys.stderr.write("[handover] REFUSED: {}\n".format(e))
        try:
            cs._log(a.root, "{} REFUSED: {}".format(a.cmd, e))
        except OSError:
            pass
        return cs.REFUSE


if __name__ == "__main__":
    sys.exit(main())
