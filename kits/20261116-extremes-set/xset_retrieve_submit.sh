#!/bin/bash
# Submit the retrievals of a date list, GROUP dates per job (default 2), jobs in parallel (MARS queues them).
# Usage: bash xset_retrieve_submit.sh <file with one yyyymmdd per line (a csv with a "date" header works)> [GROUP]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p /ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs
G=${2:-2}; i=0; batch=""
while IFS=, read -r d _; do
  [[ "$d" =~ ^[0-9]{8}$ ]] || continue
  batch="$batch $d"; i=$((i + 1))
  if (( i % G == 0 )); then sbatch --job-name=xset_ret_$((i / G)) "$HERE/xset_retrieve.sbatch" $batch; batch=""; fi
done < "$1"
if [[ -n "$batch" ]]; then sbatch --job-name=xset_ret_last "$HERE/xset_retrieve.sbatch" $batch; fi
