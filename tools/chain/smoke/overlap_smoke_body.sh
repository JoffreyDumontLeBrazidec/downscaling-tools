#!/bin/bash
# Overlap-mode smoke on real Slurm, no GPU, no torch (thread "Speed up chained runs on MN5 and Leonardo").
# Every segment: takeover -> resolve -> queue successor with NO dependency -> fake trainer that writes last.ckpt
# every SMOKE_PERIOD s (+100 steps) and appends "<job> <step> <utc>" to steps.log, until END_STEP.
# The successor usually starts within minutes on a CPU partition, so one chain exercises many handovers.
# Audit afterwards: python3 overlap_smoke_audit.py <root>  (one writer at a time, steps strictly increasing).
# Usage (from the per-cluster job file): overlap_smoke_body.sh <root> <end_step>
set -eo pipefail
ROOT="$1" END_STEP="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CHAIN_TOOLS="$HERE/chain" CHAIN_PY=python3 KERNEL_GUARD_PY=python3
export CHAIN_TAKEOVER_POLL="${CHAIN_TAKEOVER_POLL:-10}" CHAIN_TAKEOVER_STABLE="${CHAIN_TAKEOVER_STABLE:-20}"
export CHAIN_TAKEOVER_TIMEOUT="${CHAIN_TAKEOVER_TIMEOUT:-600}" CHAIN_TAKEOVER_KILL_WAIT="${CHAIN_TAKEOVER_KILL_WAIT:-300}"
source "$CHAIN_TOOLS/chain_segment.sh"
echo "[smoke] job $SLURM_JOB_ID start $(date -u +%FT%TZ) on $(hostname -s)"
chain_takeover "$ROOT"
chain_resolve "$ROOT" "$END_STEP"
chain_next_auto "$ROOT"   # CHAIN.overlap (delay) or CHAIN.pool on disk decides
chain_fence "$ROOT"   # fence (self-stop when no longer CHAIN.writer) and keeper (a lost successor is queued again)
last="${CHAIN_LAST_CKPT:-$ROOT/run1/last.ckpt}"
echo "[smoke] job $SLURM_JOB_ID trains from step $CHAIN_EXPECT_STEP at $(date -u +%FT%TZ)"
python3 - "$last" "$CHAIN_EXPECT_STEP" "$END_STEP" "${SMOKE_PERIOD:-60}" "$ROOT/steps.log" <<'PY'
import io, os, pickle, sys, time, zipfile
ckpt, step, end, period, log = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
os.makedirs(os.path.dirname(ckpt), exist_ok=True)
job = os.environ.get("SLURM_JOB_ID", "local")
while step < end:
    time.sleep(period)
    step += 100
    buf = io.BytesIO(); pickle.dump({"global_step": step, "job": job}, buf, protocol=2)
    with zipfile.ZipFile(ckpt, "w") as zf:          # in place, not atomic: harsher than Lightning 2.6.1, whose _atomic_save writes a temp file and renames
        zf.writestr("archive/data.pkl", buf.getvalue())
    with open(log, "a") as fh:
        fh.write("{} {} {}\n".format(job, step, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
PY
chain_record "$ROOT"
echo "[smoke] job $SLURM_JOB_ID reached $END_STEP at $(date -u +%FT%TZ)"
