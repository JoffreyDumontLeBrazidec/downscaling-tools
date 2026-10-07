#!/bin/bash
# Submit the T1d diagnostic GPU jobs. EVERY GPU SUBMISSION WAITS FOR THE OWNER'S TYPED GO.
#   T1D_OUT=<output root> bash submit_traj.sh smoke            # 1 draw, 12 levels (23 calls)
#   T1D_OUT=<output root> bash submit_traj.sh full             # 16 draws, 240 levels (479 calls each), 4 jobs
#   T1D_OUT=<output root> bash submit_traj.sh verify <sched.json>   # batch 2: candidate schedules, 4 jobs (verify.sbatch)
# Cluster: T1D_HOST=ac (default, A100) or ag (GH200), from t1d_env.sh / the environment, as stage A.
# ROUTE=global (AC only, fallback) submits the 4-GPU sharded variant of smoke/full instead of the cut graph.
set -euo pipefail
WHAT="${1:?smoke|full|verify}"
: "${T1D_OUT:?}"
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh"
ROUTE="${ROUTE:-box}"
HERE="$T1D_S/dp/jobs"
LOGS="$T1D_OUT/logs"; mkdir -p "$LOGS"
RES=($(t1d_host_opts))
if [[ "$ROUTE" == global ]]; then
  [[ "$T1D_HOST" == ac ]] || { echo "ROUTE=global is AC only"; exit 2; }
  RES=(--ntasks-per-node=4 --gpus-per-node=4 --mem=0 --time=03:00:00)
fi
sub() {  # name script args...
  local name=$1 script=$2; shift 2
  local jid
  jid=$(T1D_HOST=$T1D_HOST ROUTE=$ROUTE sbatch --parsable --job-name="$name" ${RES[@]+"${RES[@]}"} \
        --output="$LOGS/%x_%j.out" "$HERE/$script" "$@")
  echo "$name $jid $T1D_HOST $script $*" | tee -a "$T1D_OUT/jobs.txt"
}
J=$(t1d_jn)
BUNDLES=("20230826 024 1000 1001 1002 1003" "20230826 120 1010 1011 1012 1013"
         "20230828 024 1020 1021 1022 1023" "20230828 096 1030 1031 1032 1033")
case "$WHAT" in
  smoke) sub t1d_${J}smoke12 traj_states.sbatch 20230826 024 12 "$T1D_OUT/smoke12_d20230826_l024" 1000 ;;
  full)
    for b in "${BUNDLES[@]}"; do set -- $b; d=$1; s=$2; shift 2
      sub t1d_${J}d${d:4}_l$s traj_states.sbatch "$d" "$s" 240 "$T1D_OUT/d${d}_l$s" "$@"; done ;;
  verify)
    SCHEDS="$(realpath "${2:?schedule file from verify_schedules make}")"   # the job cd's into $T1D_CODE
    [[ -f "$SCHEDS" ]] || { echo "no schedule file $SCHEDS"; exit 2; }
    for b in "${BUNDLES[@]}"; do set -- $b; d=$1; s=$2; shift 2
      sub t1d_${J}verify_d${d:4}_l$s verify.sbatch "$d" "$s" "$T1D_OUT/verify" "$SCHEDS" "$@"; done ;;
  *) echo "smoke|full|verify"; exit 2 ;;
esac
