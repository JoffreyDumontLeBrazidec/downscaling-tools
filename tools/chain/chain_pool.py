#!/usr/bin/env python3
"""Pilot pool: several chains share a set of generic queued "pilot" jobs; each pilot that starts trains the chain
that needs it most. (Thread "Speed up chained runs on MN5 and Leonardo", 28 Nov 2026.)

Why: with one successor per chain, chain A's successor may sit in the queue while chain B's successor starts, and
B's successor cuts a B segment that still had hours left. With a pool, whichever pilot starts first goes to the
chain that is idle (no live writer) or, if none is idle, to the chain whose writer has the least time left. All
chains in a pool must ask for the SAME resources (nodes, GPUs, wall), because a pilot carries one job file.

Pool file (TSV, '#' comments), one chain per line:   <ckpt_root> <TAB> <end_step> <TAB> <body script> [args...]
The body is the chain's ordinary overlap-ready job body (chain_takeover, chain_resolve, chain_next_auto ...). Inside a
pilot it sees CHAIN_POOL_PILOT=1, so chain_next_auto queues nothing: the pilot queues the next pilot.

    chain_pool.py pick --pool P      -> POOL_ACTION=train|none, POOL_ROOT, POOL_BODY (shell lines), and the claim
    chain_pool.py status --pool P    -> one line per chain: state, writer, claim, time left

pick: pool lock (mkdir); skip chains that are done, stopped or refused; a chain whose CHAIN.claim names another live
job counts as taken. Choice: an idle chain first (oldest last.ckpt first); else the live chain with the least time
left, only if that is below --takeover-below seconds (default: always). The pick writes <root>/CHAIN.claim = this job.
Python 3.6, stdlib only.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chain_handover as ch  # noqa: E402
import chain_state as cs  # noqa: E402

CLAIM = "CHAIN.claim"
POOL_LOCK = ".pool.lock"


def read_pool(path):
    out = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise cs.Refuse("{}:{}: expected root<TAB>end_step<TAB>body [args]".format(path, n))
            out.append({"root": os.path.abspath(parts[0]), "end": int(parts[1]), "body": parts[2].strip()})
    return out


def _first_word(path):
    try:
        with open(path) as fh:
            w = fh.read().split()
        return w[0] if w else ""
    except OSError:
        return ""


def time_left(jid):
    """Seconds left of a running job (squeue %L: [D-]HH:MM:SS, MM:SS, UNLIMITED). None if unknown."""
    r = subprocess.run(["squeue", "-h", "-j", str(jid), "-o", "%L"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True)
    v = (r.stdout or "").strip().split("\n")[0].strip()
    if r.returncode != 0 or not v or v in ("INVALID", "NOT_SET"):
        return None
    if v == "UNLIMITED":
        return float("inf")
    days = 0
    if "-" in v:
        d, v = v.split("-", 1)
        days = int(d)
    nums = [int(x) for x in v.split(":")]
    while len(nums) < 3:
        nums.insert(0, 0)
    return days * 86400 + nums[0] * 3600 + nums[1] * 60 + nums[2]


def chain_view(c, me):
    """State of one chain for the pick: done/stopped/refused/taken/idle/live."""
    root = c["root"]
    try:
        res = cs._resolve(root, c["end"])
    except cs.Refuse as e:
        return dict(c, state="refused", why=str(e))
    if res["CHAIN_ACTION"] in ("done", "stopped"):
        return dict(c, state=res["CHAIN_ACTION"])
    claim = _first_word(os.path.join(root, CLAIM))
    writer = _first_word(os.path.join(root, ch.WRITER))
    if claim and claim != me and claim != writer and ch.job_state(claim):
        return dict(c, state="taken", claim=claim)  # another pilot picked it and is taking over
    if writer and writer != me:
        st = ch.job_state(writer)
        if st and st not in ch.PENDING:
            return dict(c, state="live", writer=writer, left=time_left(writer))
    last = os.path.join(root, res["CHAIN_RUN_ID"], "last.ckpt") if res["CHAIN_RUN_ID"] else ""
    mtime = os.stat(last).st_mtime if last and os.path.isfile(last) else 0.0
    return dict(c, state="idle", mtime=mtime)


def choose(views, takeover_below):
    idle = sorted((v for v in views if v["state"] == "idle"), key=lambda v: v["mtime"])
    if idle:
        return idle[0], "idle"
    live = [v for v in views if v["state"] == "live" and v.get("left") is not None]
    live.sort(key=lambda v: v["left"])
    if live and live[0]["left"] < takeover_below:
        return live[0], "least time left ({:.0f} s)".format(live[0]["left"])
    return None, "every active chain is taken or has a writer with more than {} s left".format(takeover_below)


def _lock(pool_dir, me):
    lock = os.path.join(pool_dir, POOL_LOCK)
    for _ in range(2):
        try:
            os.mkdir(lock)
            with open(os.path.join(lock, "owner"), "w") as fh:
                fh.write(me + "\n")
            return lock
        except FileExistsError:
            owner = _first_word(os.path.join(lock, "owner"))
            if owner and owner != me and ch.job_state(owner):
                raise cs.Refuse("pool lock held by live job {}".format(owner))
            shutil.rmtree(lock, ignore_errors=True)
    raise cs.Refuse("could not take the pool lock {}".format(lock))


def _pool_log(pool, line):
    with open(os.path.join(os.path.dirname(pool), "POOL.log"), "a") as fh:
        fh.write("{} job={} {}\n".format(cs._now(), ch._me() or "local", line))


def cmd_pick(a):
    me = ch._me() or "local"
    chains = read_pool(a.pool)
    lock = _lock(os.path.dirname(a.pool), me)
    try:
        views = [chain_view(c, me) for c in chains]
        active = [v for v in views if v["state"] in ("idle", "live", "taken")]
        pick, why = choose(views, a.takeover_below)
        if pick:
            claim = os.path.join(pick["root"], CLAIM)
            with open(claim + ".tmp", "w") as fh:
                fh.write("{} {}\n".format(me, cs._now()))
            os.replace(claim + ".tmp", claim)
            cs._log(pick["root"], "pool: claimed by pilot {} ({})".format(me, why))
    finally:
        shutil.rmtree(lock, ignore_errors=True)
    summary = " ".join("{}={}".format(os.path.basename(v["root"]), v["state"]) for v in views)
    _pool_log(a.pool, "pick {} ({}); chains: {}".format(pick["root"] if pick else "none", why, summary))
    print("POOL_ACTION={}".format("train" if pick else "none"))
    print("POOL_ROOT={}".format(shlex.quote(pick["root"] if pick else "")))
    print("POOL_BODY={}".format(shlex.quote(pick["body"] if pick else "")))
    print("POOL_ACTIVE={}".format(len(active)))
    sys.stderr.write("[pool] {} -> {} ({})\n".format(summary, pick["root"] if pick else "none", why))
    return 0


def cmd_status(a):
    for v in (chain_view(c, "") for c in read_pool(a.pool)):
        print("{:9s} {} writer={} claim={} left={}".format(v["state"], v["root"], v.get("writer", "-"),
                                                          v.get("claim", "-"), v.get("left", "-")))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("pick")
    s.add_argument("--pool", required=True)
    s.add_argument("--takeover-below", type=float, default=float("inf"),
                   help="take over a live chain only if its writer has less than this many seconds left")
    s = sub.add_parser("status")
    s.add_argument("--pool", required=True)
    a = p.parse_args(argv)
    if not a.cmd:
        p.error("a command is required")
    a.pool = os.path.abspath(a.pool)
    try:
        return {"pick": cmd_pick, "status": cmd_status}[a.cmd](a)
    except (cs.Refuse, ch.QueryError) as e:
        sys.stderr.write("[pool] REFUSED: {}\n".format(e))
        return cs.REFUSE


if __name__ == "__main__":
    sys.exit(main())
