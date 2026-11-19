#!/bin/bash
# MI4-S26 Atos chain: dates of the list go round-robin into MI4_LANES lanes; in a lane each retrieval waits (afterany) for the
# previous date's build, each build waits (afterok) for its retrieval. Optional FIRST job id: every lane starts afterok on it
# (the measured first date). Usage: bash mi4_chain_submit.sh <dates file> [FIRST_JOBID]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); source $HERE/mi4_set.sh
mkdir -p $MI4_A/logs $MI4_A/state
first=${2:-}; declare -a prev
for ((l = 0; l < MI4_LANES; l++)); do prev[$l]=$first; done
i=0
while read -r d; do
  [[ "$d" =~ ^[0-9]{8}$ ]] || continue
  l=$((i % MI4_LANES)); i=$((i + 1))
  dep=""; [[ -n "${prev[$l]}" ]] && dep="--dependency=$([[ ${prev[$l]} == "$first" ]] && echo afterok || echo afterany):${prev[$l]}"
  r=$(sbatch --parsable $dep --job-name=mi4_ret_$d $HERE/mi4_retrieve.sbatch $d)
  b=$(sbatch --parsable --dependency=afterok:$r --job-name=mi4_bin_$d $HERE/mi4_build.sbatch $d)
  prev[$l]=$b
  echo "CHAIN lane $l $d: retrieve $r build $b" | tee -a $MI4_A/state/jobs.txt
done < "$1"
