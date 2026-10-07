#!/bin/bash
# Submit the T1d dense-trajectory jobs. Every GPU submission waits for the owner's typed go.
#   T1D_SB=<sandbox> T1D_OUT=<output root> bash submit_traj.sh smoke   # 1 draw, 12 levels (23 calls)
#   T1D_SB=<sandbox> T1D_OUT=<output root> bash submit_traj.sh full    # 16 draws, 240 levels (479 calls each)
# ROUTE=global (fallback) submits the 4-GPU sharded variant instead of the 1-GPU cut graph.
set -euo pipefail
WHAT="${1:?smoke|full}"
: "${T1D_SB:?}" "${T1D_OUT:?}"
ROUTE="${ROUTE:-box}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="$T1D_OUT/logs"; mkdir -p "$LOGS"
RES=()
if [[ "$ROUTE" == global ]]; then RES=(--ntasks-per-node=4 --gpus-per-node=4 --mem=0 --time=03:00:00); fi
sub() {  # name date step nlev outdir seeds...
  local name=$1; shift
  local jid
  jid=$(ROUTE=$ROUTE T1D_SB=$T1D_SB sbatch --parsable --job-name="$name" ${RES[@]+"${RES[@]}"} \
        --output="$LOGS/%x_%j.out" "$HERE/traj_states.sbatch" "$@")
  echo "$name $jid $*" | tee -a "$T1D_OUT/jobs.txt"
}
case "$WHAT" in
  smoke) sub t1d_smoke12 20230826 024 12 "$T1D_OUT/smoke12_d20230826_l024" 1000 ;;
  full)
    sub t1d_d0826_l024 20230826 024 240 "$T1D_OUT/d20230826_l024" 1000 1001 1002 1003
    sub t1d_d0826_l120 20230826 120 240 "$T1D_OUT/d20230826_l120" 1010 1011 1012 1013
    sub t1d_d0828_l024 20230828 024 240 "$T1D_OUT/d20230828_l024" 1020 1021 1022 1023
    sub t1d_d0828_l120 20230828 120 240 "$T1D_OUT/d20230828_l120" 1030 1031 1032 1033 ;;
  *) echo "smoke|full"; exit 2 ;;
esac
