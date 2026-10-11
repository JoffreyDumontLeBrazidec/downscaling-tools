#!/bin/bash
# A self-submitting segment (chain_next) around a fake trainer that adds $TOY_INC steps to last.ckpt, or fails
# without writing a checkpoint when $TOY_FAIL is set. Usage: toy_next_job.sh <ckpt_root> <end_step>
set -eo pipefail
source "$TOY_CHAIN_TOOLS/chain_segment.sh"
CKPT_ROOT="$1" END_STEP="$2"
chain_resolve "$CKPT_ROOT" "$END_STEP"
chain_next "$CKPT_ROOT"
set +e
"$CHAIN_PY" "$TOY_CHAIN_TOOLS/tests/toy_fake_train.py" "$CKPT_ROOT" "$CHAIN_RUN_ID" "$CHAIN_EXPECT_STEP" "${TOY_INC:-1000}"
rc=$?
set -e
chain_record "$CKPT_ROOT"
exit $rc
