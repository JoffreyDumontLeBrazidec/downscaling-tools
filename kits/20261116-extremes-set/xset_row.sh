#!/bin/bash
# Extremes set XSET-1: ONE COMMAND PER MODEL ROW. Run it ON ac6-100 (Atos login node; it reaches Jupiter by ssh).
#
#   bash xset_row.sh <ROW> <MODEL> <STEP> [EMA_TAG] [--wait] [--table "ROW ROW ..."] [--screen [--kind sampler]]
#
#   ROW      row name, new or existing (letters, digits, underscore), e.g. GL512_450k, FS242_freeze
#   MODEL    model file jupiter/models/<MODEL>.env (run directory, config, code tree, pins); GL512 exists
#   STEP     training step of the export;  EMA_TAG which weights [ema2000]
#   --wait   stay and poll every 20 minutes until the table is written (otherwise: do the next step and exit 3)
#   --table  rows of the final table next to ENS9, INPUT and CF9 [GL512_450k and this row]
#   --smoke-only   define the row and run only its smoke (one case, 2 members, about 0.7 GH200-h); nothing else is drawn.
#                  Use it days before a row is due, on any existing export of the model: it proves the model file.
#   --screen SCREEN-TC row (21 Nov 2026, audit note epics/misc/20261121_information_gain_audit.md s8): the 22 cyclone dates
#            x members 1-5 x 6 leads = 132 files, about 26 GH200-h (3 Jupiter jobs). The row name gets the suffix _SCRTC
#            (added if missing). Scoring = cyclones only; the end is A/screen_<ROW>/screen_read.md (xset_screen_read.py)
#            against the --table rows (full rows, cut to members 1-5; default GL512_450k), printing SCREEN_DONE and a
#            flag "worth a full row" / not / no call. A SCREEN NEVER GIVES A VERDICT and never enters a table or a paper.
#   --kind   pair kind of the screen read: training [default; line 0.4 hPa / 0.3 m/s] or sampler (same weights; line 0)
# Optional environment: XSET_WORK (Atos work folder) and XSET_JWORK (Jupiter work folder) for another checkout.
# Optional environment, passed to the row definition: EMA_FILE, RAW_FILE, XSET_SCHED_JSON, XSET_SAMPLER_JSON,
# XSET_AG_FILE, XSET_AG_W, XSET_AG_WMAP (autoguidance by an earlier export of the same run; see jupiter/xset_define_row.sh).
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
tag=ema2000; wait=0; trows=""; smokeonly=0; screen=0; kind=training
while (( $# )); do
  case "$1" in --wait) wait=1 ;; --smoke-only) smokeonly=1 ;; --screen) screen=1 ;; --kind) kind=${2:?kind}; shift ;;
    --table) trows=${2:?rows}; shift ;; --*) echo "FATAL unknown option $1"; exit 2 ;; *) tag=$1 ;; esac
  shift
done
A=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}
K=$A/downscaling-tools/kits/20261116-extremes-set
JK=${XSET_JWORK:-/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set}/downscaling-tools/kits/20261116-extremes-set/jupiter
[[ $kind == training || $kind == sampler ]] || { echo "FATAL --kind $kind (training or sampler)"; exit 2; }
need=252
if [[ $screen == 1 ]]; then
  [[ $row == *_SCRTC ]] || { row=${row}_SCRTC; echo "[xset_row] SCREEN-TC: row name is $row"; }
  need=132; export XSET_SCREEN=TC
else
  [[ $row != *_SCRTC ]] || { echo "FATAL $row ends in _SCRTC: a screen row needs --screen"; exit 2; }
fi
J="ssh -o BatchMode=yes -o ConnectTimeout=30 jupiter"
[[ -z "$trows" ]] && { trows="GL512_450k"; [[ $row != GL512_450k ]] && trows="$trows $row"; }
envs=""
jx=""; [[ -n "${XSET_JWORK:-}" ]] && jx="XSET_JWORK=$(printf %q "$XSET_JWORK") "
for v in XSET_SCREEN EMA_FILE RAW_FILE XSET_SCHED_JSON XSET_SAMPLER_JSON XSET_AG_FILE XSET_AG_W XSET_AG_WMAP; do [[ -n "${!v:-}" ]] && envs="$envs $v=$(printf %q "${!v}")"; done
say() { echo "[xset_row $row $(date -u +%FT%TZ)] $*"; }
val() { echo "$st" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }

step_once() {   # returns 0 = done, 3 = in progress, other = error
  if [[ $smokeonly == 1 ]]; then
    $J "$jx$envs bash $JK/xset_define_row.sh $row $model $step $tag" || { say "DEFINE_FAIL (Jupiter unreachable or row refused)"; return 1; }
    st=$($J "${jx}bash $JK/xset_row_state.sh $row") || { say "STATE_FAIL"; return 1; }
    say "$st"
    if [[ $(val rowsmoke) == 1 ]]; then say "SMOKE_PASS model $(val model) through row $row (Jupiter XOUT/smoke/$row, log xsmoke_$(val model)-*.out)"; return 0; fi
    if [[ $(val smokejobs) != 0 ]]; then say "SMOKE in the queue"; return 3; fi
    $J "${jx}XSET_FORCE=${XSET_FORCE:-0} bash $JK/xset_submit_row.sh $row smoke" || { say "SUBMIT_FAIL"; return 1; }
    return 3
  fi
  local na; na=$(ls $A/rows/$row/predictions_*_step*.nc 2>/dev/null | wc -l)
  if [[ $screen == 1 && -f $A/scores/$row/SCORE_DONE ]] && ! squeue -h -u "$USER" -o %j | grep -qx "xset_score_$row"; then
    local refs="" r; for r in $trows; do
      [[ $r == "$row" ]] && continue
      if [[ -s $A/scores/$r/tc_members.csv ]]; then refs="$refs --ref $r=$A/scores/$r"; else say "screen read: row $r is not scored yet, left out"; fi
    done
    [[ -n "$refs" ]] || { say "SCREEN_READ_FAIL no scored reference row among: $trows"; return 1; }
    module load python3 >/dev/null 2>&1
    python3 $K/xset_screen_read.py --screen $row=$A/scores/$row $refs --ens $A/scores/ENS9 --kind $kind \
      --dates $K/lists/dates_xset1.csv --out $A/screen_$row || { say "SCREEN_READ_FAIL"; return 1; }
    say "SCREEN_DONE (a screen, not a verdict) $(grep -m1 'SCREEN FLAG' $A/screen_$row/screen_read.md); read $A/screen_$row/screen_read.md"
    return 0
  fi
  if [[ $screen == 0 ]] && [[ -f $A/scores/$row/SCORE_DONE || ( -s $A/scores/$row/obs_points/scores_per_date.csv && -s $A/scores/$row/tc_cases_scored.csv ) ]] \
     && ! squeue -h -u "$USER" -o %j | grep -qx "xset_score_$row"; then
    local have=""; for r in $trows; do
      if [[ -s $A/scores/$r/obs_points/scores_per_date.csv ]]; then have="$have $r"; else say "table: row $r is not scored yet, left out"; fi
    done
    bash $K/xset_table.sh $have || { say "TABLE_FAIL"; return 1; }
    mkdir -p $A/table_$row && cp $A/table/xset_table.md $A/table/xset_table.csv $A/table_$row/
    say "ROW_DONE rows in the table:$have; table $A/table_$row/xset_table.md ($(wc -l < $A/table_$row/xset_table.md) lines); scores $A/scores/$row; tail check $A/scores/$row/tail_check.txt"
    return 0
  fi
  if [[ $na == $need ]]; then
    if squeue -h -u "$USER" -o %j | grep -qx "xset_score_$row"; then say "SCORING (job xset_score_$row in the queue)"; return 3; fi
    if ls $A/logs/xset_score_$row-*.out >/dev/null 2>&1 && grep -lq "_FAIL" $A/logs/xset_score_$row-*.out 2>/dev/null && [[ ${XSET_FORCE:-0} != 1 ]]; then
      say "SCORE_FAIL earlier (see $A/logs/xset_score_$row-*.out); fix, then XSET_FORCE=1"; return 1
    fi
    local sarg=""; [[ $screen == 1 ]] && sarg=screen
    local id; id=$(sbatch --parsable --job-name=xset_score_$row $K/xset_score.sbatch $row $sarg) || { say "sbatch failed"; return 1; }
    say "SCORE_SUBMITTED job $id"; return 3
  fi
  $J "$jx$envs bash $JK/xset_define_row.sh $row $model $step $tag" || { say "DEFINE_FAIL (Jupiter unreachable or row refused)"; return 1; }
  st=$($J "${jx}bash $JK/xset_row_state.sh $row") || { say "STATE_FAIL"; return 1; }
  say "$st"
  [[ $(val need) == $need ]] || { say "ROW_KIND_MISMATCH row $row on Jupiter needs $(val need) files, this call $need (screen and full rows do not mix)"; return 1; }
  if [[ $(val files) == $need && $(val part) == 0 && $(val jobs) == 0 ]]; then
    bash $K/xset_copy_from_jupiter.sh $row || { say "COPY_FAIL"; return 1; }
    [[ $(ls $A/rows/$row/predictions_*_step*.nc | wc -l) == $need ]] || { say "COPY_INCOMPLETE"; return 1; }
    step_once; return $?
  fi
  if [[ $(val jobs) != 0 || $(val smokejobs) != 0 ]]; then say "DRAWING ($(val files) of $need files, $(val jobs) row jobs, $(val smokejobs) smoke jobs in the queue)"; return 3; fi
  $J "${jx}XSET_FORCE=${XSET_FORCE:-0} bash $JK/xset_submit_row.sh $row" || { say "SUBMIT_FAIL"; return 1; }
  return 3
}

while true; do
  step_once; rc=$?
  [[ $rc == 3 && $wait == 1 ]] || exit $rc
  sleep 1200
done
