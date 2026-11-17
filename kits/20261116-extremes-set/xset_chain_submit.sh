#!/bin/bash
# Submit the Atos chain of a date list, GROUP dates per job: retrieval, then input stores (afterok), then evaluation
# points and 9 km ensemble rows (afterok, and after the station cache job when one is given).
# Usage: bash xset_chain_submit.sh <csv with a "date" header or one yyyymmdd per line> [GROUP=2] [CACHE_JOBID|none] [MODE=ret]
#   MODE ret            submit the retrievals too
#        noret          the GRIB files are complete: the store job starts at once (it fails on a missing .ok file)
#        after:ID[:ID]  the retrievals are already submitted as these job ids: every store job waits for all of them
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
A=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}; mkdir -p $A/logs; LOG="-o $A/logs/%x-%j.out"
G=${2:-2}; CJ=${3:-none}; MODE=${4:-ret}; i=0; batch=""
go() {
  local tag=$1; shift
  local dep=""
  if [[ $MODE == ret ]]; then
    r=$(sbatch --parsable $LOG --job-name=xset_ret_$tag "$HERE/xset_retrieve.sbatch" "$@"); dep="--dependency=afterok:$r"
  elif [[ $MODE == after:* ]]; then r=${MODE#after:}; dep="--dependency=afterok:$r"
  else r=none; fi
  b=$(sbatch --parsable $LOG $dep --job-name=xset_bin_$tag "$HERE/xset_build_in.sbatch" "$@")
  dep="--dependency=afterok:$b"; [[ $CJ != none ]] && dep="$dep,afterok:$CJ"
  w=$(sbatch --parsable $LOG $dep --job-name=xset_rows_$tag "$HERE/xset_rows.sbatch" "$@")
  echo "CHAIN $tag dates $* : retrieve $r, stores $b, rows $w"
}
while IFS=, read -r d _; do
  [[ "$d" =~ ^[0-9]{8}$ ]] || continue
  batch="$batch $d"; i=$((i + 1))
  if (( i % G == 0 )); then go "$d" $batch; batch=""; fi
done < "$1"
if [[ -n "$batch" ]]; then go last $batch; fi
