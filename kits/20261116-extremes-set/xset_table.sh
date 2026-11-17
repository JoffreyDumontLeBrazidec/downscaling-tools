#!/bin/bash
# The table of the extremes set from the scored rows on Atos. ENS9 is the reference row; INPUT and CF9 come from its files.
# Usage: bash xset_table.sh <LABEL> [<LABEL> ...]       e.g. bash xset_table.sh GL512_500k
set -eu
A=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set
K=$A/downscaling-tools/kits/20261116-extremes-set
module load python3 >/dev/null 2>&1
rows="--row ENS9=$A/scores/ENS9"; for l in "$@"; do rows="$rows --row $l=$A/scores/$l"; done
python3 $K/xset_table.py $rows --ref ENS9 --dates $K/lists/dates_xset1.csv --hist $A/hist --out $A/table
echo "[table] finished $(date -u +%FT%TZ)"
