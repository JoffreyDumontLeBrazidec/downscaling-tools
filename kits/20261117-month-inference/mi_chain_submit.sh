#!/bin/bash
# Submit the Atos chain of a date list, GROUP dates per job: retrieval, then input stores (afterok).
# Usage: bash mi_chain_submit.sh <file with one yyyymmdd per line> [GROUP=2] [MODE=ret|noret]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /ec/res4/scratch/ecm5702/agent-work/20261117-month-inference/logs
G=${2:-2}; MODE=${3:-ret}; i=0; batch=""
go() {
  local tag=$1; shift
  local dep="" r=none
  if [[ $MODE == ret ]]; then r=$(sbatch --parsable --job-name=mi_ret_$tag "$HERE/mi_retrieve.sbatch" "$@"); dep="--dependency=afterok:$r"; fi
  b=$(sbatch --parsable $dep --job-name=mi_bin_$tag "$HERE/mi_build_in.sbatch" "$@")
  echo "CHAIN $tag dates $* : retrieve $r, stores $b"
}
while IFS=, read -r d _; do
  [[ "$d" =~ ^[0-9]{8}$ ]] || continue
  batch="$batch $d"; i=$((i + 1))
  if (( i % G == 0 )); then go "$d" $batch; batch=""; fi
done < "$1"
if [[ -n "$batch" ]]; then go last $batch; fi
