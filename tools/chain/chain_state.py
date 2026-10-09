#!/usr/bin/env python3
"""Decide what a training segment does from the run's state on disk, never from a submit-time variable.

Why (2026-10-09): on Atos `SBATCH_EXPORT=NONE` dropped `WK_MODE=resume` / `PL_MODE=resume` given in front of
`sbatch`, the job files fell back on their default (warm start from the donor), and weight decay 0 trained a
donor restart for hours. Any chain whose mode comes from the submitter can do this on any cluster. Here the
mode is read from the checkpoint folder of the run, and every ambiguous state refuses instead of guessing.

The chain's checkpoint root (the folder that holds one sub-folder per run id, `<root>/<run_id>/last.ckpt`)
carries a manifest `CHAIN.json`, written once on the login node:

    chain_state.py init  --root R --note "fork from donor 12dcefea"    # a new chain (no run yet)
    chain_state.py adopt --root R --run-id X                           # an existing run (live chain, moved run)

Each segment, inside the job, before training:

    eval "$(chain_state.py resolve --root R --end-step N)"   # sets CHAIN_ACTION, CHAIN_RUN_ID, CHAIN_EXPECT_STEP

    fresh   no run yet: start from the body's own start (donor / fork); the guard expects step 0
    resume  resume run CHAIN_RUN_ID from its last.ckpt; the guard expects step CHAIN_EXPECT_STEP
    done    the run's last.ckpt is at or past N: exit 0 without training

and after training `chain_state.py record --root R` records the run id and fails if a second run appeared.
Refusals exit 3 with the reason on stderr. Python 3.6 compatible (the Atos login nodes), stdlib only.
"""
import argparse
import json
import os
import pickle
import sys
import time
import zipfile

MANIFEST = "CHAIN.json"
LOG = "CHAIN.log"
REFUSE = 3


class Refuse(Exception):
    pass


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(root, line):
    job = os.environ.get("SLURM_JOB_ID", "local")
    host = os.uname()[1].split(".")[0]
    with open(os.path.join(root, LOG), "a") as fh:
        fh.write("{} job={} host={} {}\n".format(_now(), job, host, line))


def _read_manifest(root):
    path = os.path.join(root, MANIFEST)
    if not os.path.isfile(path):
        raise Refuse(
            "no {} in {}: this folder is not a declared chain. For a new chain run "
            "'chain_state.py init --root {}'; for an existing or moved run, "
            "'chain_state.py adopt --root {} --run-id <id>'. Refusing rather than guessing a start.".format(
                MANIFEST, root, root, root
            )
        )
    with open(path) as fh:
        return json.load(fh)


def _write_manifest(root, manifest):
    path = os.path.join(root, MANIFEST)
    tmp = path + ".tmp.{}".format(os.getpid())
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def runs_with_checkpoints(root):
    """Run folders under root that hold any checkpoint, as {run_id: has_last_ckpt}."""
    found = {}
    if not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d) or name.startswith("."):
            continue
        ckpts = [f for f in os.listdir(d) if f.endswith(".ckpt")]
        if ckpts:
            found[name] = os.path.isfile(os.path.join(d, "last.ckpt"))
    return found


# ── global_step from a Lightning checkpoint without torch and without loading tensors ──────────────────


class _Stub(object):
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()

    # list- and dict-like classes are rebuilt with APPENDS / SETITEMS
    def append(self, item):
        pass

    def extend(self, items):
        pass

    def __setitem__(self, key, value):
        pass


class _StubUnpickler(pickle.Unpickler):
    """Reads the checkpoint dict, replacing every non-builtin class and every tensor by a stub."""

    _SAFE = {("collections", "OrderedDict"), ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice")}

    def find_class(self, module, name):
        if (module, name) in self._SAFE:
            return pickle.Unpickler.find_class(self, module, name)
        return _Stub

    def persistent_load(self, pid):
        return None


def checkpoint_step(path):
    """The `global_step` stored in a Lightning checkpoint (torch zip format)."""
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(pkl) != 1:
            raise Refuse("{}: expected one data.pkl in the checkpoint archive, found {}".format(path, len(pkl)))
        with zf.open(pkl[0]) as fh:
            obj = _StubUnpickler(fh).load()
    if not isinstance(obj, dict) or not isinstance(obj.get("global_step"), int):
        raise Refuse("{}: no integer global_step in the checkpoint".format(path))
    return obj["global_step"]


# ── commands ───────────────────────────────────────────────────────────────────────────────────────────


def cmd_init(a):
    os.makedirs(a.root, exist_ok=True)
    if os.path.exists(os.path.join(a.root, MANIFEST)):
        raise Refuse("{} already exists in {}: this chain is declared; nothing to init".format(MANIFEST, a.root))
    existing = sorted(runs_with_checkpoints(a.root))
    if existing and not a.ignore_existing:
        raise Refuse(
            "{} already holds runs with checkpoints ({}). To continue one of them use 'adopt --run-id'; to start a "
            "new run beside them (e.g. a fork of one of them) pass --ignore-existing.".format(a.root, ", ".join(existing))
        )
    manifest = {"version": 1, "created": _now(), "note": a.note, "run_id": None, "ignored_runs": existing}
    _write_manifest(a.root, manifest)
    _log(a.root, "init note={!r} ignored_runs={}".format(a.note, existing))
    print("chain declared in {} (no run yet; ignored runs: {})".format(a.root, existing or "none"))


def cmd_adopt(a):
    last = os.path.join(a.root, a.run_id, "last.ckpt")
    if not os.path.isfile(last):
        raise Refuse("cannot adopt {}: {} does not exist".format(a.run_id, last))
    step = checkpoint_step(last)
    path = os.path.join(a.root, MANIFEST)
    if os.path.exists(path):
        m = _read_manifest(a.root)
        if m.get("run_id") not in (None, a.run_id):
            raise Refuse("{} already records run {}; refusing to switch it to {}".format(path, m["run_id"], a.run_id))
    else:
        m = {"version": 1, "created": _now(), "note": a.note, "run_id": None, "ignored_runs": []}
    others = sorted(r for r in runs_with_checkpoints(a.root) if r != a.run_id)
    m["run_id"] = a.run_id
    m["ignored_runs"] = sorted(set(m.get("ignored_runs", [])) | set(others))
    _write_manifest(a.root, m)
    _log(a.root, "adopt run_id={} step={} ignored_runs={}".format(a.run_id, step, m["ignored_runs"]))
    print("chain {} now continues run {} (last.ckpt at step {}; ignored runs: {})".format(
        a.root, a.run_id, step, m["ignored_runs"] or "none"))


def _resolve(root, end_step):
    m = _read_manifest(root)
    ignored = set(m.get("ignored_runs", []))
    runs = {r: has_last for r, has_last in runs_with_checkpoints(root).items() if r not in ignored}
    run_id = m.get("run_id")

    if run_id is None:
        if not runs:
            return {"CHAIN_ACTION": "fresh", "CHAIN_RUN_ID": "", "CHAIN_EXPECT_STEP": "0", "CHAIN_LAST_CKPT": ""}
        if len(runs) > 1:
            raise Refuse("{} holds {} runs and the manifest records none ({}): which one is the chain is not "
                         "decidable; adopt the right one".format(root, len(runs), ", ".join(sorted(runs))))
        run_id = list(runs)[0]
        m["run_id"] = run_id
        _write_manifest(root, m)
        _log(root, "record run_id={} (adopted at resolve: the only run of an undeclared-id chain)".format(run_id))

    stray = sorted(r for r in runs if r != run_id)
    if stray:
        raise Refuse("{} holds run(s) {} besides the chain's run {}: a segment started a second run (a silent "
                     "restart?). Inspect, then move the stray folder aside or list it in ignored_runs.".format(
                         root, ", ".join(stray), run_id))
    last = os.path.join(root, run_id, "last.ckpt")
    if not os.path.isfile(last):
        raise Refuse("the chain's run {} has no {}: a resume would fail or restart; refusing".format(run_id, last))
    step = checkpoint_step(last)
    action = "done" if end_step is not None and step >= end_step else "resume"
    return {"CHAIN_ACTION": action, "CHAIN_RUN_ID": run_id, "CHAIN_EXPECT_STEP": str(step), "CHAIN_LAST_CKPT": last}


def cmd_resolve(a):
    out = _resolve(a.root, a.end_step)
    _log(a.root, "resolve action={} run_id={} expect_step={} end_step={}".format(
        out["CHAIN_ACTION"], out["CHAIN_RUN_ID"] or "-", out["CHAIN_EXPECT_STEP"], a.end_step))
    if a.format == "shell":
        for k in sorted(out):
            print("{}='{}'".format(k, out[k]))
    else:
        print(json.dumps(out))
    sys.stderr.write("[chain] {}: {} run={} step={} end={}\n".format(
        a.root, out["CHAIN_ACTION"], out["CHAIN_RUN_ID"] or "(new)", out["CHAIN_EXPECT_STEP"], a.end_step))


def cmd_record(a):
    m = _read_manifest(a.root)
    ignored = set(m.get("ignored_runs", []))
    runs = sorted(r for r in runs_with_checkpoints(a.root) if r not in ignored)
    if m.get("run_id") is None:
        if len(runs) == 1:
            m["run_id"] = runs[0]
            _write_manifest(a.root, m)
            _log(a.root, "record run_id={}".format(runs[0]))
        elif not runs:
            _log(a.root, "record: no checkpoint written yet")
            print("[chain] no checkpoint written yet; the next segment starts fresh again")
            return
    run_id = m["run_id"]
    stray = [r for r in runs if r != run_id]
    if stray:
        _log(a.root, "record STRAY runs {} besides {}".format(stray, run_id))
        raise Refuse("a second run appeared in {}: {} besides {}".format(a.root, stray, run_id))
    last = os.path.join(a.root, run_id, "last.ckpt")
    step = checkpoint_step(last) if os.path.isfile(last) else None
    _log(a.root, "record run_id={} last_step={}".format(run_id, step))
    print("[chain] run {} last.ckpt step {}".format(run_id, step))


def cmd_step(a):
    print(checkpoint_step(a.ckpt))


def cmd_status(a):
    m = _read_manifest(a.root)
    print(json.dumps(m, indent=2, sort_keys=True))
    for r, has_last in sorted(runs_with_checkpoints(a.root).items()):
        step = checkpoint_step(os.path.join(a.root, r, "last.ckpt")) if has_last else None
        tag = "chain" if r == m.get("run_id") else ("ignored" if r in m.get("ignored_runs", []) else "STRAY")
        print("{:8s} {} last.ckpt step {}".format(tag, r, step))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("init", help="declare a new chain in a checkpoint root (login node)")
    s.add_argument("--root", required=True)
    s.add_argument("--note", default="")
    s.add_argument("--ignore-existing", action="store_true", help="start a new run beside runs already there")
    s = sub.add_parser("adopt", help="declare an existing run as the chain's run (live or moved chain)")
    s.add_argument("--root", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--note", default="")
    s = sub.add_parser("resolve", help="what this segment does (inside the job, before training)")
    s.add_argument("--root", required=True)
    s.add_argument("--end-step", type=int, default=None)
    s.add_argument("--format", choices=("shell", "json"), default="shell")
    s = sub.add_parser("record", help="record the run id and catch a second run (inside the job, after training)")
    s.add_argument("--root", required=True)
    s = sub.add_parser("step", help="print the global_step of a checkpoint")
    s.add_argument("ckpt")
    s = sub.add_parser("status", help="manifest and runs of a chain")
    s.add_argument("--root", required=True)
    a = p.parse_args(argv)
    if not a.cmd:
        p.error("a command is required")
    if hasattr(a, "root"):
        a.root = os.path.abspath(a.root)
    try:
        {"init": cmd_init, "adopt": cmd_adopt, "resolve": cmd_resolve, "record": cmd_record,
         "step": cmd_step, "status": cmd_status}[a.cmd](a)
    except Refuse as e:
        sys.stderr.write("[chain] REFUSED: {}\n".format(e))
        if a.cmd in ("resolve", "record") and os.path.isdir(getattr(a, "root", "")):
            try:
                _log(a.root, "{} REFUSED: {}".format(a.cmd, e))
            except OSError:
                pass
        return REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
