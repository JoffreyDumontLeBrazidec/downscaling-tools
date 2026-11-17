#!/bin/bash
# Extremes set XSET-1: state of one model row on Jupiter, one line of KEY=value pairs (read by xset_row.sh on Atos).
# Usage: bash xset_row_state.sh <ROW>
set -uo pipefail
row=${1:?row}
XOUT=${XSET_JOUT:-/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116}
HERE=$(cd "$(dirname "$0")" && pwd)
model=GL512; defined=0
case "$row" in GL512_500k|GL512_450k|GL512_237k|REG_GL512_30k|REG_GL512_20k) defined=1 ;; esac
if [[ -f $XOUT/rowdefs/$row.env ]]; then defined=1; model=$(grep -m1 '^MODEL=' $XOUT/rowdefs/$row.env | cut -d= -f2); fi
files=$(ls $XOUT/rows/$row/predictions_*_step*.nc 2>/dev/null | wc -l)
part=$(ls $XOUT/rows/$row/*.part 2>/dev/null | wc -l)
hist=$(ls $XOUT/hist/${row}_*.npz 2>/dev/null | wc -l)
q=$(squeue -h -u "$USER" -o "%j %T" 2>/dev/null) || { echo "STATE_FAIL squeue"; exit 1; }
jobs=$(echo "$q" | grep -c "^xset_${row}_[0-9]* " || true)
smokejobs=$(echo "$q" | grep -c "^xsmoke_${model} " || true)
smoke=0; [[ $model == GL512 || -f $XOUT/smoke/MODEL_${model}_PASS ]] && smoke=1
rowsmoke=0; [[ -f $XOUT/smoke/$row/XSET_SMOKE_PASS ]] && rowsmoke=1
submits=$(grep -c ' row' $XOUT/rowdefs/$row.submits 2>/dev/null || true)
echo "ROW_STATE row=$row defined=$defined model=$model files=$files part=$part hist=$hist jobs=$jobs smoke=$smoke rowsmoke=$rowsmoke smokejobs=$smokejobs submits=${submits:-0} kit=$(git -C "$HERE" rev-parse --short HEAD) utc=$(date -u +%FT%TZ)"
