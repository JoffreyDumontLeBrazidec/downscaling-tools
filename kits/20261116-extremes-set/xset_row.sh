#!/bin/bash
# Extremes set XSET-1: ONE COMMAND PER MODEL ROW. Run it ON ac6-100 (Atos login node; it reaches Jupiter by ssh).
#
#   bash xset_row.sh <ROW> <MODEL> <STEP> [EMA_TAG] [--wait] [--table "ROW ROW ..."]
#
#   ROW      row name, new or existing (letters, digits, underscore), e.g. GL512_450k, FS242_freeze
#   MODEL    model file jupiter/models/<MODEL>.env (run directory, config, code tree, pins); GL512 exists
#   STEP     training step of the export;  EMA_TAG which weights [ema2000]
#   --wait   stay and poll every 20 minutes until the table is written (otherwise: do the next step and exit 3)
#   --table  rows of the final table next to ENS9, INPUT and CF9 [GL512_450k and this row]
#   --smoke-only   define the row and run only its smoke (one case, 2 members, about 0.7 GH200-h); nothing else is drawn.
#                  Use it days before a row is due, on any existing export of the model: it proves the model file.
# Optional environment, passed to the row definition: EMA_FILE, RAW_FILE, XSET_SCHED_JSON, XSET_SAMPLER_JSON.
#
# The command is a state machine and can be run again at any time; each call does the next missing step, never twice:
#   1 define the row on Jupiter (files + sha256 of the weights, written once)
#   2 smoke of a new model (2 draws, fatal sanity bounds), once per model
#   3 six Jupiter jobs of 7 start dates (252 files = 42 dates x 6 leads, 10 members; about 100 GH200-h)
#   4 copy the row to Atos          5 score job (stations, cyclones against IBTrACS, IMERG, tail check; about 1 h)
#   6 table with detectable effects in A/table_<ROW>/  ->  prints ROW_DONE
# Exit: 0 table written, 3 a step is running or was just started (call again), other = stopped on an error.
set -uo pipefail
row=${1:?row}; model=${2:?model}; step=${3:?step}; shift 3
tag=ema2000; wait=0; trows=""; smokeonly=0
while (( $# )); do
  case "$1" in --wait) wait=1 ;; --smoke-only) smokeonly=1 ;; --table) trows=${2:?rows}; shift ;; --*) echo "FATAL unknown option $1"; exit 2 ;; *) tag=$1 ;; esac
  shift
done
A=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set
K=$A/downscaling-tools/kits/20261116-extremes-set
JK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/downscaling-tools/kits/20261116-extremes-set/jupiter
J="ssh -o BatchMode=yes -o ConnectTimeout=30 jupiter"
[[ -z "$trows" ]] && { trows="GL512_450k"; [[ $row != GL512_450k ]] && trows="$trows $row"; }
envs=""
for v in EMA_FILE RAW_FILE XSET_SCHED_JSON XSET_SAMPLER_JSON; do [[ -n "${!v:-}" ]] && envs="$envs $v=$(printf %q "${!v}")"; done
say() { echo "[xset_row $row $(date -u +%FT%TZ)] $*"; }
val() { echo "$st" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }

step_once() {   # returns 0 = done, 3 = in progress, other = error
  if [[ $smokeonly == 1 ]]; then
    $J "$envs bash $JK/xset_define_row.sh $row $model $step $tag" || { say "DEFINE_FAIL (Jupiter unreachable or row refused)"; return 1; }
    st=$($J "bash $JK/xset_row_state.sh $row") || { say "STATE_FAIL"; return 1; }
    say "$st"
    if [[ $(val rowsmoke) == 1 ]]; then say "SMOKE_PASS model $(val model) through row $row (Jupiter XOUT/smoke/$row, log xsmoke_$(val model)-*.out)"; return 0; fi
    if [[ $(val smokejobs) != 0 ]]; then say "SMOKE in the queue"; return 3; fi
    $J "XSET_FORCE=${XSET_FORCE:-0} bash $JK/xset_submit_row.sh $row smoke" || { say "SUBMIT_FAIL"; return 1; }
    return 3
  fi
  local na; na=$(ls $A/rows/$row/predictions_*_step*.nc 2>/dev/null | wc -l)
  if [[ -f $A/scores/$row/SCORE_DONE || ( -s $A/scores/$row/obs_points/scores_per_date.csv && -s $A/scores/$row/tc_cases_scored.csv ) ]] \
     && ! squeue -h -u "$USER" -o %j | grep -qx "xset_score_$row"; then
    local have=""; for r in $trows; do
      if [[ -s $A/scores/$r/obs_points/scores_per_date.csv ]]; then have="$have $r"; else say "table: row $r is not scored yet, left out"; fi
    done
    bash $K/xset_table.sh $have || { say "TABLE_FAIL"; return 1; }
    mkdir -p $A/table_$row && cp $A/table/xset_table.md $A/table/xset_table.csv $A/table_$row/
    say "ROW_DONE rows in the table:$have; table $A/table_$row/xset_table.md ($(wc -l < $A/table_$row/xset_table.md) lines); scores $A/scores/$row; tail check $A/scores/$row/tail_check.txt"
    return 0
  fi
  if [[ $na == 252 ]]; then
    if squeue -h -u "$USER" -o %j | grep -qx "xset_score_$row"; then say "SCORING (job xset_score_$row in the queue)"; return 3; fi
    if ls $A/logs/xset_score_$row-*.out >/dev/null 2>&1 && grep -lq "_FAIL" $A/logs/xset_score_$row-*.out 2>/dev/null && [[ ${XSET_FORCE:-0} != 1 ]]; then
      say "SCORE_FAIL earlier (see $A/logs/xset_score_$row-*.out); fix, then XSET_FORCE=1"; return 1
    fi
    local id; id=$(sbatch --parsable --job-name=xset_score_$row $K/xset_score.sbatch $row) || { say "sbatch failed"; return 1; }
    say "SCORE_SUBMITTED job $id"; return 3
  fi
  $J "$envs bash $JK/xset_define_row.sh $row $model $step $tag" || { say "DEFINE_FAIL (Jupiter unreachable or row refused)"; return 1; }
  st=$($J "bash $JK/xset_row_state.sh $row") || { say "STATE_FAIL"; return 1; }
  say "$st"
  if [[ $(val files) == 252 && $(val part) == 0 && $(val jobs) == 0 ]]; then
    bash $K/xset_copy_from_jupiter.sh $row || { say "COPY_FAIL"; return 1; }
    [[ $(ls $A/rows/$row/predictions_*_step*.nc | wc -l) == 252 ]] || { say "COPY_INCOMPLETE"; return 1; }
    step_once; return $?
  fi
  if [[ $(val jobs) != 0 || $(val smokejobs) != 0 ]]; then say "DRAWING ($(val files) of 252 files, $(val jobs) row jobs, $(val smokejobs) smoke jobs in the queue)"; return 3; fi
  $J "XSET_FORCE=${XSET_FORCE:-0} bash $JK/xset_submit_row.sh $row" || { say "SUBMIT_FAIL"; return 1; }
  return 3
}

while true; do
  step_once; rc=$?
  [[ $rc == 3 && $wait == 1 ]] || exit $rc
  sleep 1200
done
