"""python -m health <command>: the offline network-health audit of a checkpoint (review note s4 tier C).

    audit CKPT     CPU checks 1, 3, 6 (numpy only, any login node; ~1 min per 2 GB file); --gpu adds 2, 4, 7, 9 and
                   the graph audit 8 (one GPU, ~10 min); writes audit.json + SUMMARY.txt; check 12 diffs against
                   --prev and the lane reference.
    diff A B       check 12 between two audit.json files.
    registry       the reference checkpoint of each lane.
    revive SRC DST --audit A   write a revived copy (by hand only; needs torch).

Readings, not gates: the exit code is 0 whatever the readings say.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import registry as reg
from . import report
from .static_checks import capacity_summary, run_static

DEFAULT_ROOT = os.path.expanduser(os.environ.get("HEALTH_ROOT", "~/perm/health"))


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ckpt_id(path: str, run_id: str | None, step) -> str:
    stem = Path(path).name
    m = re.search(r"step_?(\d+)", stem)
    s = int(step) if step is not None else (int(m.group(1)) if m else None)
    ema = re.search(r"ema(\d+)", stem)
    run = (run_id or Path(path).parent.name)[:8]
    tag = f"{run}_{s // 1000}k" if s is not None and s >= 1000 else f"{run}_{s}" if s is not None else Path(stem).stem
    return tag + (f"_ema{ema.group(1)}" if ema else "")


def revive_commands(path: str, audit_json: str, audit: dict) -> list[str]:
    cap = audit["capacity"]["processor"]
    cmds = []
    zero = cap["units_dead_zero"] + cap["heads_dead_zero"]
    grad = cap["units_dead_grad"] + cap["heads_dead_grad"]
    dst = re.sub(r"\.ckpt$", "", path) + "_revived.ckpt"
    if zero:
        cmds.append(f"# {zero} zero-weight units/heads (widening saddle): anemoi-core fork tool, or the health one below")
        cmds.append(f"python -m anemoi.training.utils.dead_units {path} {dst}")
    if grad or zero:
        cmds.append(f"# {grad} gradient-dead + {zero} zero-weight units/heads: input side redrawn, output side zeroed")
        cmds.append(f"python -m health revive {path} {dst} --audit {audit_json} --what grad,zero")
    return cmds


def cmd_audit(a) -> int:
    t0 = time.time()
    r, blocks, checks = run_static(a.ckpt, skip_nonfinite=a.skip_nonfinite)
    cid = a.id or ckpt_id(a.ckpt, r.run_id, r.global_step)
    out_dir = Path(a.out or os.path.join(DEFAULT_ROOT, cid))
    meta = {
        "ckpt": os.path.abspath(a.ckpt),
        "ckpt_id": cid,
        "lane": a.lane,
        "kind": r.kind,
        "size_gb": round(r.size_gb, 3),
        "global_step": r.global_step,
        "run_id": r.run_id,
        "health_git": _git_sha(),
        "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": os.uname().nodename,
    }
    if a.gpu:
        import torch

        from .backends import FLOW_SIGMAS, BundlesBackend
        from .forward_checks import graph_audit, run_flow, run_grad_pass

        be = BundlesBackend(a.ckpt, a.bundles, device=a.device, precision=a.precision, local_scope_json=a.local_scope_json)
        sig = tuple(float(s) for s in a.sigmas.split(",")) if a.sigmas else None
        checks |= run_grad_pass(be.root, be.cases(sig) if sig else be.cases(), a.device)
        flow_cases = [c for c in be.cases(FLOW_SIGMAS)][: len(FLOW_SIGMAS)]
        checks |= run_flow(be.root, flow_cases)
        if be.graph is not None:
            checks |= graph_audit(be.graph)
        meta["gpu"] = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else a.device,
                       "bundles": [n for n, _ in be.batches], "precision": a.precision}
    rules = report.load_bands(a.bands)
    audit = {
        "meta": meta,
        "blocks": [b.__dict__ for b in blocks],
        "checks": checks,
        "capacity": {g: capacity_summary(blocks, checks, g) for g in ("encoder", "processor", "decoder")},
    }
    audit["headline"] = report.headline(audit)
    audit["bands"] = report.evaluate(checks, rules)
    audit["revive"] = revive_commands(a.ckpt, str(out_dir / "audit.json"), audit)
    audit["diffs"] = {}
    for label, other in (("previous", a.prev), ("reference", reg.reference_audit(a.lane) if a.ref == "auto" else a.ref)):
        if other and os.path.exists(other):
            audit["diffs"][label] = report.diff(json.load(open(other)), audit, rules)
    meta["seconds"] = round(time.time() - t0, 1)
    report.write(audit, out_dir)
    sys.stdout.write(report.summary_text(audit))
    sys.stdout.write(f"  written {out_dir}/audit.json and SUMMARY.txt ({meta['seconds']} s)\n")
    return 0


def cmd_diff(a) -> int:
    rules = report.load_bands(a.bands)
    d = report.diff(json.load(open(a.a)), json.load(open(a.b)), rules, a.threshold)
    print(f"check 12: {a.b} vs {a.a}: {len(d['moved'])} readings moved > {a.threshold:.0%}")
    for mv in d["moved"][: a.top]:
        print(f"  {mv['metric']}: {mv['before']:.4g} -> {mv['after']:.4g}")
    for k in ("only_before", "only_after"):
        if d[k]:
            print(f"  {k}: {len(d[k])} readings (roles differ), e.g. {d[k][:3]}")
    return 0


def cmd_registry(a) -> int:
    for lane, e in reg.load().items():
        if a.lane and lane != a.lane:
            continue
        print(f"{lane}: {e.get('label')}  ({e.get('cluster')})\n  ckpt  {e.get('ckpt')}\n  audit {e.get('audit_json')}\n  why   {e.get('why')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m health", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="audit one checkpoint")
    p.add_argument("ckpt")
    p.add_argument("--id", help="name of the output folder (default <run8>_<step>k[_emaN])")
    p.add_argument("--out", help=f"output folder (default {DEFAULT_ROOT}/<id>)")
    p.add_argument("--lane", help="lane key of registry.yaml, for the reference diff")
    p.add_argument("--prev", help="audit.json of the previous checkpoint of the same run (check 12)")
    p.add_argument("--ref", default="auto", help="reference audit.json, or 'auto' = registry entry of --lane")
    p.add_argument("--bands", help="bands file (default health/bands.yaml)")
    p.add_argument("--skip-nonfinite", action="store_true", help="skip the full NaN/Inf scan (saves ~half the time)")
    p.add_argument("--gpu", action="store_true", help="also run checks 2, 4, 7, 8, 9 (needs torch, the model code, a GPU)")
    p.add_argument("--bundles", nargs="+", default=[], help="prepml *input_bundle.nc files or globs (2 dates = owner default)")
    p.add_argument("--sigmas", help="comma list for checks 2/4/7 (default 0.2,2,20,80)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="bf16", help="autocast dtype of the forward (training runs bf16-mixed)")
    p.add_argument("--local-scope-json", help="graph cut for a box checkpoint on global bundles (eval.predict)")
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("diff", help="check 12 between two audit.json")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--bands")
    p.set_defaults(func=cmd_diff)
    p = sub.add_parser("registry", help="the reference checkpoint of each lane")
    p.add_argument("--lane")
    p.set_defaults(func=cmd_registry)
    p = sub.add_parser("revive", help="write a revived copy (by hand only)", add_help=False)
    p.set_defaults(func=None)
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "revive":
        from .revive import main as revive_main

        return revive_main(rest)
    if rest:
        ap.error(f"unrecognized arguments: {rest}")
    if args.cmd == "audit" and args.gpu and not args.bundles:
        ap.error("--gpu needs --bundles")
    return args.func(args)
