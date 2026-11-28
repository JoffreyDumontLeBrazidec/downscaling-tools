# shellcheck shell=bash
# Source from a training job body (after the venv is active). Every segment of a chain is the SAME job file;
# what it does is read from the run's checkpoint folder, never from a variable given to sbatch.
#
#   source "$CHAIN_TOOLS/chain_segment.sh"
#   chain_store_guard --config "$CONFIG"              # exits 49 if a store named by the config lost chunk files
#   chain_resolve "$CKPT_ROOT" "$END_STEP"           # exits 0 if done or stopped, 3 if the state is ambiguous,
#                                                    # 97 if the job would import the old Triton kernel (rule 47)
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
# Rule 47: the job's own interpreter (venv python, job PYTHONPATH) must import the FIXED Triton kernel.
KERNEL_GUARD_PY="${KERNEL_GUARD_PY:-python}"

chain_kernel_guard() {
  "$KERNEL_GUARD_PY" "$CHAIN_TOOLS/kernel_guard.py" check || {
    echo "[chain] segment refused: old or unknown Triton kernel on this job's import path (rule 47); nothing trained" >&2
    exit 97; }
}

chain_resolve() {
  local root="$1" end_step="$2" out
  [[ -n "$root" && -n "$end_step" ]] || { echo "[chain] usage: chain_resolve <ckpt_root> <end_step>" >&2; exit 2; }
  chain_kernel_guard
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

# Refuse the segment (exit 49) when a zarr store it reads has lost chunk or metadata files: purged scratch files
# read back as the fill value (zeros) without any error (Atos purge of 9 Nov 2026). Arguments go to
# `store_guard.py check`: store paths and/or `--config <yaml>`. Listings only, no chunk is read.
# STORE_GUARD_ALLOW_INCOMPLETE=1 downgrades the refusal to a warning (timing smokes only).
chain_store_guard() {
  "$CHAIN_PY" "$CHAIN_TOOLS/store_guard.py" check --quiet "$@"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[chain] segment refused: incomplete input store (store_guard exit $rc); nothing trained" >&2
    exit "$rc"
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

# ── Overlap mode (28 Nov 2026, chain_handover.py) ─────────────────────────────────────────────────────────
# The successor is queued with NO dependency so it waits in the queue while this segment trains. At its start it
# waits for the running segment's next checkpoint, scancels it, waits until Slurm no longer lists it, and only then
# trains. Use INSTEAD of chain_next:
#   chain_takeover "$CKPT_ROOT"                       # FIRST, before chain_resolve (exit 3 if it cannot be sole writer)
#   chain_resolve  "$CKPT_ROOT" "$END_STEP"
#   chain_next_overlap "$CKPT_ROOT" "$BEGIN_DELAY_S"  # 0 = queue now; else sbatch --begin=now+S
chain_takeover() {
  local root="$1"; shift
  "$CHAIN_PY" "$CHAIN_TOOLS/chain_handover.py" takeover --root "$root" "$@" || {
    echo "[chain] segment refused: could not become the chain's only writer; nothing trained" >&2; exit 3; }
}

chain_next_overlap() {
  local root="$1" delay="${2:-0}" max_pending="${3:-1}" cmd
  local -a cmd_words=()
  [[ -n "${SLURM_JOB_ID:-}" ]] || { echo "[chain] chain_next_overlap: not inside a Slurm job"; return 0; }
  cmd="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^[[:space:]]*Command=//p' | head -n 1)" || true
  read -r -a cmd_words <<< "$cmd"
  if [[ ${#cmd_words[@]} -eq 0 || ! -f "${cmd_words[0]}" ]]; then
    echo "[chain] WARNING: cannot read this job's script (Command='$cmd'); no successor" >&2
    "$CHAIN_PY" "$CHAIN_TOOLS/chain_state.py" log --root "$root" "overlap-next NOT submitted: no job script" || true
    return 0
  fi
  "$CHAIN_PY" "$CHAIN_TOOLS/chain_handover.py" next --root "$root" --begin-delay "$delay" \
      --max-pending "$max_pending" --job-file "${cmd_words[0]}" -- "${cmd_words[@]:1}" || true
}

# One body for every mode, chosen ON DISK (never by a submit-time variable):
#   inside a pilot job (CHAIN_POOL_PILOT=1), or <root>/CHAIN.pool exists: queue nothing (the pool's pilots do);
#   <root>/CHAIN.overlap exists: overlap mode, its first word = begin delay in seconds (0 = queue now);
#   otherwise: the classic afterany successor (chain_next).
# Call it where chain_next was called, and call chain_takeover before chain_resolve (harmless in classic mode).
chain_next_auto() {
  local root="$1" delay
  if [[ "${CHAIN_POOL_PILOT:-}" == 1 || -f "$root/CHAIN.pool" ]]; then
    echo "[chain] pool chain (pilot job or $root/CHAIN.pool): the pool queues pilots; this body queues nothing"; return 0; fi
  if [[ -f "$root/CHAIN.overlap" ]]; then
    delay="$(awk 'NR==1{print $1}' "$root/CHAIN.overlap")"
    [[ "$delay" =~ ^[0-9]+$ ]] || delay=0
    chain_next_overlap "$root" "$delay"
  else
    chain_next "$root"
  fi
}
