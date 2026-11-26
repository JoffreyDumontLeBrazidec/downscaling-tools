#!/bin/bash
# MeanFlow month rows on MI-S24 (Sep 2024, 48r1; 26 Nov 2026, thread "Score MeanFlow on Sep 2024 for the paper"): FG-199
# glmf_alljump_50k step STEP x 1000 (default 500), EMA-2000, rows GLAJ<STEP>K_J10, _J4, _J2, _J1 (mi_runs.sh), from THIS checkout.
# Chain, all on account gkpdm with --kill-on-invalid-dep=yes (a failed link cancels what waits on it, no scancel); the month jobs
# with --nice=100, the short gate and smokes without (they are the critical path; times sized for backfill):
#   1. gate mi_xcheck.sbatch GLAJ400K_GJX10 (skipped when outputs/month_inference_s25/xcheck/GLAJ400K_GJX10/MI_XCHECK_PASS exists):
#      the month path must reproduce the XSET-1 row GLAJ400K_GJX10 bit for bit (20240804 +24 h and +72 h, 10 members);
#   2. per row a smoke on 20240905 (afterok gate); 3. per row the 30 dates of dates_s24.txt in groups of 3 (afterok its smoke;
#      the smoke date is skipped there because its files exist).
# Usage: bash mi_submit_meanflow_s24.sh            (DRY_RUN=1 prints the sbatch lines only; ROWS="J10 J4" for a subset)
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
J=/e/data1/jureap-data/ecmwf/users/jdumont
LOGS=${LOGS:-$J/agent-work/20261126-meanflow-s24/logs}
STEP=${STEP:-500}
ROWS=${ROWS:-"J10 J4 J2 J1"}
SMOKE=${SMOKE:-20240905}
GATE_MARK=$J/outputs/month_inference_s25/xcheck/GLAJ400K_GJX10/MI_XCHECK_PASS
mkdir -p "$LOGS"
REC=$LOGS/submit_meanflow_s24_$(date -u +%Y%m%dT%H%M%SZ).txt
SB=(sbatch --parsable --account=gkpdm --kill-on-invalid-dep=yes --export=ALL,MI_K=$HERE)
sub() {   # sub <name> <time> <dependency or -> <nice> <script> <args...>
  local name=$1 tl=$2 dep=$3 nc=$4; shift 4
  local cmd=("${SB[@]}" --job-name=$name --time=$tl --nice=$nc --output=$LOGS/%x-%j.out)
  [[ $dep == - ]] || cmd+=(--dependency=afterok:$dep)
  cmd+=("$@")
  if [[ -n "${DRY_RUN:-}" ]]; then echo "DRY ${cmd[*]}" >&2; echo "DRY_$name"; return; fi
  local id; id=$("${cmd[@]}"); echo "$name $id ${cmd[*]}" >> $REC; echo "$id"
}
mapfile -t DATES < <(grep -E '^202409[0-9]{2}$' $HERE/../dates_s24.txt)
(( ${#DATES[@]} == 30 )) || { echo "FATAL ${#DATES[@]} dates in dates_s24.txt"; exit 2; }
echo "kit $(git -C $HERE rev-parse --short HEAD) $HERE; step $STEP; rows $ROWS; record $REC" | tee -a $REC
if [[ -f $GATE_MARK ]]; then gate=-; echo "gate marker exists: $GATE_MARK" | tee -a $REC
else gate=$(sub mi_xcheck_mf 00:40:00 - 0 $HERE/mi_xcheck.sbatch GLAJ400K_GJX10); echo "gate $gate" | tee -a $REC; fi
for r in $ROWS; do
  R=GLAJ${STEP}K_$r
  # wall per group of 3 dates = 600 draws: XSET-1 draw_s 5.92 s (10 calls) / 0.68 s (1 call) + month packing + about 10 min start
  case $r in J10) tg=02:30:00; ts=01:00:00 ;; J4) tg=01:30:00; ts=00:45:00 ;; *) tg=01:00:00; ts=00:45:00 ;; esac
  s=$(sub mi_${R}_s24s $ts $gate 0 $HERE/mi_predict.sbatch $R $SMOKE)
  ids=""
  for ((i = 0; i < 30; i += 3)); do
    g=$(( i / 3 + 1 ))
    id=$(sub mi_${R}_s24_$g $tg $s 100 $HERE/mi_predict.sbatch $R "${DATES[@]:i:3}")
    ids="$ids $id"
  done
  echo "row $R smoke $s months$ids" | tee -a $REC
done
