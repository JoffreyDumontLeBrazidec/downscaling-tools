#!/usr/bin/env python3
"""Resolve a T1d lane for the job scripts and for gate G2 (2026-10-07).

Loads the lane through eval/config/loader.py (base chain + predict.sampler_overrides), then:
  - writes <outdir>/<lane>.sampler.json  (the resolved predict.sampler, the --extra-args-json of the job and,
    after the run, the sampling_config_json attribute of every prediction file),
  - writes <outdir>/<lane>.scope.json    (predict.local_scope, the --local-scope-json of the job),
  - writes <outdir>/<lane>.ckpt          (the BASE checkpoint path passed as --name-ckpt),
  - prints the resolved block, its denoiser calls and its applied levels (make_lanes_t1d.applied_sigmas),
  - refuses unless predict.checkpoint is the 1.2M parent (run 551dfd1e, step 400,000), never the donor 12dcefea.
With --check-files the inference and base checkpoint files must exist (run it on Atos).
Run from the downscaling-tools worktree root (it imports eval.config.loader).
Usage: python resolve_lane_t1d.py <lane> --outdir <dir> [--check-files]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from eval.config.loader import load_lane  # noqa: E402

import make_lanes_t1d as mk  # noqa: E402

RUN_1P2M = "551dfd1eb2a649d49f099acd30b59483"
INFER_1P2M = "inference-anemoi-by_step-epoch_171-step_400000.ckpt"
DONOR = "12dcefeafa92457aab286d4931ffe63f"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lane")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--check-files", action="store_true")
    args = ap.parse_args()
    cfg = load_lane(args.lane)
    pred = cfg["predict"]
    ck = pred.get("checkpoint")
    if not ck:
        raise SystemExit(f"FATAL {args.lane}: no predict.checkpoint (is the base tc_o320_o1280_p12m_ctrl?)")
    ckp = Path(ck)
    if DONOR in ck or ckp.parent.name != RUN_1P2M or ckp.name != INFER_1P2M:
        raise SystemExit(f"FATAL {args.lane}: checkpoint {ck} is not the 1.2M parent {RUN_1P2M}/{INFER_1P2M}")
    base = ckp.with_name(ckp.name[len("inference-"):])
    if args.check_files:
        for p in (ckp, base):
            if not p.is_file():
                raise SystemExit(f"FATAL missing checkpoint file {p}")
    sampler = pred["sampler"]
    scope = pred.get("local_scope") or {}
    if scope.get("mode") != "bbox" or not scope.get("cut_graph"):
        raise SystemExit(f"FATAL {args.lane}: expected the cut-graph bbox scope of tc_o320_o1280, got {scope}")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.lane}.sampler.json").write_text(json.dumps(sampler) + "\n")
    (out / f"{args.lane}.scope.json").write_text(json.dumps(scope) + "\n")
    (out / f"{args.lane}.ckpt").write_text(str(base) + "\n")
    sig = mk.applied_sigmas(sampler)
    calls = mk.ml.calls(sampler)
    print(f"LANE {args.lane}")
    print(f"CHECKPOINT inference={ckp} base(--name-ckpt)={base} run={ckp.parent.name}")
    print(f"SCOPE {json.dumps(scope)}")
    print(f"SAMPLER {json.dumps(sampler)}")
    print(f"APPLIED levels={len(sig)} calls={calls} S_churn={sampler.get('S_churn')} S_noise={sampler.get('S_noise')} sigmas=[{', '.join(f'{s:.6g}' for s in sig)}, 0]")
    print(f"MEMBERS {pred['members']} DATES {pred['dates']} STEPS(lane) {pred['steps']} GPUS {pred.get('num_gpus_per_model')}")
    qos = (cfg.get("resource_profiles", {}).get("predict") or {}).get("qos")
    print(f"QOS(predict) {qos}")


if __name__ == "__main__":
    main()
