#!/bin/bash
# The table of the extremes set from the scored rows on Atos. ENS9 is the reference row; INPUT and CF9 come from its files.
# Usage: bash xset_table.sh <LABEL> [<LABEL> ...]       e.g. bash xset_table.sh GL512_500k
# Row against row (paired differences of two model rows, e.g. two checkpoints of one run): give the reference with REF
# and another output folder with OUT; ENS9 is then left out.
#   REF=GL512_500k OUT=$A/table_450k_vs_500k bash xset_table.sh GL512_500k GL512_450k
set -eu
A=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set
K=$A/downscaling-tools/kits/20261116-extremes-set
module load python3 >/dev/null 2>&1
ref=${REF:-ENS9}; out=${OUT:-$A/table}
rows=""; [[ $ref == ENS9 ]] && rows="--row ENS9=$A/scores/ENS9"
for l in "$@"; do rows="$rows --row $l=$A/scores/$l"; done
python3 $K/xset_table.py $rows --ref $ref --dates $K/lists/dates_xset1.csv --hist $A/hist --out $out
echo "[table] finished ref=$ref out=$out $(date -u +%FT%TZ)"
