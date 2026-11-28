#!/bin/bash
# Pilot job of a chain pool (chain_pool.py). Call it from a pilot sbatch file that asks for the resources every
# chain of the pool uses:   bash "$CHAIN_TOOLS/pilot_job.sh" <pool.tsv> [begin_delay_s] [takeover_below_s]
# 1. pick the chain that needs a writer (idle first, else least time left) and claim it;
# 2. queue the next pilot (no dependency) while fewer pilots are pending than chains are active;
# 3. exec that chain's own body with CHAIN_POOL_PILOT=1 (its chain_next_auto then queues nothing).
set -eo pipefail
POOL="$1" DELAY="${2:-0}" BELOW="${3:-}"
[[ -f "$POOL" ]] || { echo "[pool] usage: pilot_job.sh <pool.tsv> [begin_delay_s] [takeover_below_s]" >&2; exit 2; }
CHAIN_TOOLS="${CHAIN_TOOLS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CHAIN_PY="${CHAIN_PY:-python}"
pick_args=(pick --pool "$POOL"); [[ -n "$BELOW" ]] && pick_args+=(--takeover-below "$BELOW")
out="$("$CHAIN_PY" "$CHAIN_TOOLS/chain_pool.py" "${pick_args[@]}")" || { echo "[pool] pick refused" >&2; exit 3; }
eval "$out"
# A pilot that found nothing to train must not queue its successor at once: on a quiet queue that is a job storm.
[[ "$POOL_ACTION" != train && "$DELAY" -lt 900 ]] && DELAY=900
if [[ "${POOL_ACTIVE:-0}" -gt 0 && -n "${SLURM_JOB_ID:-}" ]]; then
  cmd="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^[[:space:]]*Command=//p' | head -n 1)" || true
  read -r -a w <<< "$cmd"
  if [[ ${#w[@]} -gt 0 && -f "${w[0]}" ]]; then
    "$CHAIN_PY" "$CHAIN_TOOLS/chain_handover.py" next --root "$(dirname "$POOL")" --begin-delay "$DELAY" \
        --max-pending "$POOL_ACTIVE" --job-file "${w[0]}" -- "${w[@]:1}" || true
  else
    echo "[pool] WARNING: cannot read this pilot's job file; no next pilot" >&2
  fi
fi
if [[ "$POOL_ACTION" != train ]]; then
  echo "[pool] nothing to train (active chains: ${POOL_ACTIVE:-0})"; exit 0
fi
echo "[pool] pilot ${SLURM_JOB_ID:-local} trains $POOL_ROOT"
export CHAIN_POOL_PILOT=1 CHAIN_POOL_ROOT="$POOL_ROOT"
read -r -a body <<< "$POOL_BODY"
exec bash "${body[@]}"
