#!/bin/bash
# Extremes set XSET-1: submit what one defined row still needs on Jupiter. Safe to call again: it never submits while
# jobs of the row are in the queue, it only asks for the dates whose files are missing, and it stops after 3 submissions
# of the same row (XSET_FORCE=1 lifts that). A new model (not GL512) gets its smoke first (xset_smoke.sbatch).
# Usage: bash xset_submit_row.sh <ROW> [smoke]        "smoke": submit the smoke of this row and nothing else
# A SCREEN-TC row (XSET_SCREEN=TC in its definition) asks only for the 22 cyclone dates, 5 members, in 3 jobs (8, 8, 6
# dates). XSET_DRYRUN=1 prints the date groups that would be submitted and submits nothing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
row=${1:?row}; only=${2:-}
XOUT=${XSET_JOUT:-/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116}
XK=${XSET_JWORK:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set}
T=$XK/downscaling-tools
st=$(bash $HERE/xset_row_state.sh "$row"); echo "$st"
val() { echo "$st" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
[[ $(val defined) == 1 ]] || { echo "FATAL row $row is not defined (xset_define_row.sh)"; exit 3; }
[[ $(val jobs) == 0 && $(val smokejobs) == 0 ]] || { echo "SUBMIT_NONE $row: jobs of this row or model are in the queue"; exit 0; }
if [[ $only == smoke ]]; then
  [[ $(val rowsmoke) != 1 ]] || { echo "SUBMIT_NONE $row: smoke already passed"; exit 0; }
else
  [[ $(val files) != $(val need) || $(val part) != 0 ]] || { echo "SUBMIT_NONE $row: $(val need) files, nothing to draw"; exit 0; }
fi
dry=${XSET_DRYRUN:-0}
# never sbatch an unpushed or modified kit
[[ $dry == 1 ]] || [[ -z "$(git -C $T status --porcelain -- kits/20261116-extremes-set)" ]] || { echo "FATAL the kit checkout has local changes"; exit 6; }
[[ $dry == 1 ]] || [[ -n "$(git -C $T branch -r --contains HEAD 2>/dev/null)" ]] || { echo "FATAL the kit HEAD is on no remote branch (push first)"; exit 6; }
# the row resolves and its code tree is at its pin (same checks as inside the job)
( set +u; source $HERE/xset_runs.sh "$row" && source $HERE/xset_env.sh ) >/dev/null || { echo "FATAL row $row does not resolve (model file, pins or code tree)"; exit 7; }
mkdir -p $XK/logs $XOUT/rowdefs
model=$(val model)
if [[ $dry != 1 ]] && [[ $(val smoke) != 1 || $only == smoke ]]; then
  ns=$(grep -c ' smoke' $XOUT/rowdefs/$row.submits 2>/dev/null || true)
  [[ ${ns:-0} -lt 2 || ${XSET_FORCE:-0} == 1 ]] || { echo "FATAL the smoke of row $row was submitted $ns times without a pass: read $XK/logs/xsmoke_$model-*.out, then XSET_FORCE=1"; exit 8; }
  id=$(sbatch --parsable --job-name=xsmoke_$model $HERE/xset_smoke.sbatch "$row")
  echo "$(date -u +%FT%TZ) smoke $id" >> $XOUT/rowdefs/$row.submits
  echo "SUBMIT_SMOKE $row model=$model job=$id"; exit 0
fi
n=$(val submits); [[ ${n:-0} -lt 3 || ${XSET_FORCE:-0} == 1 ]] || { echo "FATAL row $row was submitted $n times and is not complete: read the job logs in $XK/logs, then XSET_FORCE=1"; exit 8; }
# full row: the six date groups of the first rows (7 start dates each, in the order of the frozen list);
# SCREEN-TC row: the 22 cyclone dates (tag TC or both) in groups of 8; only dates with a missing file
mapfile -t dates < <(awk -F, 'NR>1 && $1 ~ /^[0-9]{8}$/ {print $1}' $HERE/../lists/dates_xset1.csv)
[[ ${#dates[@]} == 42 ]] || { echo "FATAL ${#dates[@]} dates in the frozen list (want 42)"; exit 9; }
gs=7; ng=6
if [[ $(val screen) == TC ]]; then
  mapfile -t dates < <(awk -F, 'NR>1 && $1 ~ /^[0-9]{8}$/ && ($2 == "TC" || $2 == "both") {print $1}' $HERE/../lists/dates_xset1.csv)
  [[ ${#dates[@]} == 22 ]] || { echo "FATAL ${#dates[@]} cyclone dates in the frozen list (want 22)"; exit 9; }
  gs=8; ng=3
fi
ids=""
for g in $(seq 1 $ng); do
  need=""
  for d in "${dates[@]:$(( (g - 1) * gs )):$gs}"; do
    for L in 012 024 036 048 060 072; do
      [[ -s $XOUT/rows/$row/predictions_${d}_step$L.nc ]] || { need="$need $d"; break; }
    done
  done
  [[ -n "$need" ]] || continue
  if [[ $dry == 1 ]]; then echo "  DRYRUN group $g dates$need ($(echo $need | wc -w) dates)"; continue; fi
  id=$(sbatch --parsable --job-name=xset_${row}_$g $HERE/xset_predict.sbatch "$row" $need)
  ids="$ids $id"; echo "  group $g job $id dates$need"
done
[[ $dry == 1 ]] && { echo "DRYRUN_DONE $row screen=$(val screen)"; exit 0; }
echo "$(date -u +%FT%TZ) row$ids" >> $XOUT/rowdefs/$row.submits
echo "SUBMIT_ROW $row jobs$ids"
