#!/bin/bash
# MI-S25: copy one model row's finished date folders (20 npz each) from Jupiter to Atos for GRIB encoding. Run ON ac6-101/ac6-100
# (the Jupiter ssh master). Verifies by sorted (name, size) lists, not du (du counts directory entries: 4 KB trap of 18 Nov).
# Usage: bash mi_copy_from_jupiter.sh <ROW> [date ...]   (no dates = every date folder with 20 files on Jupiter)
set -eu
row=${1:?row}; shift
J=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/month_inference_s25/rows/$row
A=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference/rows/$row
mkdir -p $A
rsync -a jupiter:$J/grid_o1280.npz $A/
dates=("$@")
[[ ${#dates[@]} -gt 0 ]] || mapfile -t dates < <(ssh jupiter "cd $J && for d in 2*; do [[ \$(ls \$d/pred_*.npz 2>/dev/null | wc -l) == 20 ]] && echo \$d; done")
for d in "${dates[@]}"; do
  [[ -f $A/$d.ok ]] && { echo "have $d"; continue; }
  rsync -a --info=stats1 jupiter:$J/$d $A/ | grep -E "bytes/sec" | tr '\n' ' '
  j=$(ssh jupiter "cd $J/$d && ls -l pred_*.npz | awk '{print \$5, \$9}' | sort | md5sum")
  a=$(cd $A/$d && ls -l pred_*.npz | awk '{print $5, $9}' | sort | md5sum)
  [[ "$j" == "$a" && $(ls $A/$d/pred_*.npz | wc -l) == 20 ]] && { date -u +%FT%TZ > $A/$d.ok; echo "OK $d"; } || echo "MISMATCH $d"
done
echo "COPY_BACK_DONE $(date -u +%FT%TZ)"
