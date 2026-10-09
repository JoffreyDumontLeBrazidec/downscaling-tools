#!/bin/bash
# A chain segment body as the method prescribes, around toy_train.py. Usage: toy_job.sh <ckpt_root> <end_step> <donor> [extra overrides]
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/../chain_segment.sh"
CKPT_ROOT="$1" END_STEP="$2" DONOR="$3"; shift 3
chain_resolve "$CKPT_ROOT" "$END_STEP"
if [[ "$CHAIN_ACTION" == fresh ]]; then
  start=("system.input.warm_start=$DONOR" "training.load_weights_only=True" "training.run_id=null")
else
  mapfile -t start < <(chain_resume_overrides)
fi
set +e
"$CHAIN_PY" "$CHAIN_TOOLS/chain_guard_launch.py" --expect-step "$CHAIN_EXPECT_STEP" --root "$CKPT_ROOT" -- \
  "$here/toy_train.py" "system.output.checkpoints.root=$CKPT_ROOT" "training.max_steps=$END_STEP" "${start[@]}" "$@"
rc=$?
set -e
chain_record "$CKPT_ROOT"
exit $rc
