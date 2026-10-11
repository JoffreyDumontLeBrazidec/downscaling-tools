# shellcheck shell=bash
# Source from a training job body (after the venv is active). Every segment of a chain is the SAME job file;
# what it does is read from the run's checkpoint folder, never from a variable given to sbatch.
#
#   source "$CHAIN_TOOLS/chain_segment.sh"
#   chain_resolve "$CKPT_ROOT" "$END_STEP"           # exits 0 if done or stopped, 3 if the state is ambiguous
#   chain_next "$CKPT_ROOT"                          # optional: queue this job file again, afterany this job
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
  if [[ "$CHAIN_ACTION" == stopped ]]; then
    echo "[chain] $root/CHAIN.stop exists: the chain is stopped; nothing trained, no successor"
    exit 0
  fi
}

# On-demand segments: queue this same job file once more, with --dependency=afterany on this job, so a chain needs
# one submission instead of a pre-submitted series. Call it right after chain_resolve (so a segment that dies in
# its first minutes still has its successor queued). No successor when: not inside Slurm, CHAIN.stop exists, the
# last <max_same_start> segments (default 3) all started at the same step (no progress), or a job running the same
# command is already pending (an older pre-submitted segment, or a resubmission by hand). A failed submission
# warns and lets this segment train; the chain then ends after it.
chain_next() {
  local root="$1" max_same="${2:-3}" cmd jid pending v
  local -a unset_env=() cmd_words=()
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[chain] chain_next: not inside a Slurm job; no successor submitted"
    return 0
  fi
  if ! "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" next-check --root "$root" --max-same-start "$max_same"; then
    echo "[chain] WARNING: no successor submitted (reason above); the chain ends after this segment" >&2
    return 0
  fi
  cmd="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^[[:space:]]*Command=//p' | head -n 1)" || true
  read -r -a cmd_words <<< "$cmd"
  if [[ ${#cmd_words[@]} -eq 0 || ! -f "${cmd_words[0]}" ]]; then
    echo "[chain] WARNING: cannot read this job's script from scontrol (Command='$cmd'); no successor submitted" >&2
    "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" log --root "$root" "next NOT submitted: no job script (Command='$cmd')" || true
    return 0
  fi
  pending="$(squeue -h -u "${USER:-$(id -un)}" -t PENDING -o '%i|%o' 2>/dev/null \
             | awk -F'|' -v c="$cmd" '$2 == c { print $1; exit }')" || true
  if [[ -n "$pending" ]]; then
    echo "[chain] successor already queued: job $pending runs the same command; nothing submitted"
    "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" log --root "$root" "next already queued job=$pending" || true
    return 0
  fi
  # A job's own SLURM_* variables must not leak into the successor's submission (SLURM_CONF is kept).
  for v in $(compgen -e); do
    [[ "$v" == SLURM_* && "$v" != SLURM_CONF ]] && unset_env+=(-u "$v")
  done
  if jid="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && env "${unset_env[@]}" sbatch --parsable \
              --dependency="afterany:$SLURM_JOB_ID" "${cmd_words[@]}")"; then
    jid="${jid%%;*}"
    echo "[chain] successor queued: job $jid (afterany:$SLURM_JOB_ID)"
    "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" log --root "$root" "next queued job=$jid afterany=$SLURM_JOB_ID" || true
  else
    echo "[chain] WARNING: sbatch of the successor failed; the chain ends after this segment" >&2
    "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" log --root "$root" "next NOT submitted: sbatch failed" || true
  fi
  return 0
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
