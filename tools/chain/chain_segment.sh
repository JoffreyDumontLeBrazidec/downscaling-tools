# shellcheck shell=bash
# Source from a training job body (after the venv is active). Every segment of a chain is the SAME job file;
# what it does is read from the run's checkpoint folder, never from a variable given to sbatch.
#
#   source "$CHAIN_TOOLS/chain_segment.sh"
#   chain_resolve "$CKPT_ROOT" "$END_STEP"           # exits 0 if done, 3 if the state is ambiguous
#   if [[ $CHAIN_ACTION == fresh ]]; then start=( <the body's donor or fork overrides> )
#   else mapfile -t start < <(chain_resume_overrides); fi
#   srun ... python "$CHAIN_TOOLS/chain_guard_launch.py" --expect-step "$CHAIN_EXPECT_STEP" --root "$CKPT_ROOT" \
#        -- <entry> "${overrides[@]}" "${start[@]}"
#   rc=$?; chain_record "$CKPT_ROOT"; exit $rc
#
# CHAIN_TOOLS defaults to the folder of this file. Full method: tools/chain/README.md.

CHAIN_TOOLS="${CHAIN_TOOLS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CHAIN_PY="${CHAIN_PY:-python}"

chain_resolve() {
  local root="$1" end_step="$2" out
  [[ -n "$root" && -n "$end_step" ]] || { echo "[chain] usage: chain_resolve <ckpt_root> <end_step>" >&2; exit 2; }
  out="$("$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" resolve --root "$root" --end-step "$end_step")" || {
    echo "[chain] segment refused (state above); nothing trained" >&2; exit 3; }
  eval "$out"
  export CHAIN_ACTION CHAIN_RUN_ID CHAIN_EXPECT_STEP CHAIN_LAST_CKPT
  if [[ "$CHAIN_ACTION" == done ]]; then
    echo "[chain] run $CHAIN_RUN_ID is at step $CHAIN_EXPECT_STEP >= $end_step: nothing to do"
    exit 0
  fi
}

# The anemoi overrides of a full resume of CHAIN_RUN_ID (weights, optimiser, step, EMA, scheduler).
chain_resume_overrides() {
  printf '%s\n' "training.run_id=$CHAIN_RUN_ID" "training.fork_run_id=null" "system.input.warm_start=null" \
    "training.transfer_learning=False" "training.load_weights_only=False"
}

chain_record() {
  "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" record --root "$1" || echo "[chain] WARNING: record refused (see above); the next segment will refuse too" >&2
  local ok="$1/.chain_guard/${SLURM_JOB_ID:-local}.ok"
  [[ -f "$ok" ]] || echo "[chain] WARNING: no guard marker $ok: the step guard did not confirm this segment's start" >&2
}
