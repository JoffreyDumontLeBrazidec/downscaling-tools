#!/bin/bash
# Overlap-mode segment around toy_writer.py (trains until cancelled). Usage: toy_overlap_job.sh <ckpt_root> <end_step>
set -eo pipefail
source "$TOY_CHAIN_TOOLS/chain_segment.sh"
CKPT_ROOT="$1" END_STEP="$2"
chain_takeover "$CKPT_ROOT"
chain_resolve "$CKPT_ROOT" "$END_STEP"
chain_next_overlap "$CKPT_ROOT" 0
exec "$CHAIN_PY" "$TOY_CHAIN_TOOLS/tests/toy_writer.py" "$CHAIN_LAST_CKPT" "$CHAIN_EXPECT_STEP" 0.5
