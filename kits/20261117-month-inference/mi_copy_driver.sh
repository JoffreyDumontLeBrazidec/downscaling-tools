#!/bin/bash
# Lock-aware copy driver Atos -> Jupiter for one month set (19 Nov 2026, MI-J25). Run 3 copies at once under nohup ON ac6-101/ac6-100
# (Jupiter ssh master): each takes the next date whose store has stores/check/in_<d>.done, copies it, and verifies it by the md5 of
# the sorted (relative path, size) list on both sides, never du (4 KB directory trap of 18 Nov). Marker state/copied_<d>.ok.
# Loops every 5 min until every date of the list is copied or MAXR rounds pass. Nothing is deleted on either side. No sbatch.
# Usage: nohup bash mi_copy_driver.sh <dates file> <stream tag> [MAXR=180] > $A/logs/cpj_<tag>.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); source $HERE/mi_set.sh
A=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference; JD=/e/data1/jureap-data/ecmwf/users/jdumont/datasets
list=${1:?dates file}; tag=${2:?tag}; maxr=${3:-180}
mapfile -t dates < <(grep -E '^[0-9]{8}' "$list" | cut -c1-8)
lsum() { find "$1" -type f -printf '%P %s\n' | sort | md5sum | cut -c1-32; }
for ((r = 1; r <= maxr; r++)); do
  left=0
  for d in "${dates[@]}"; do
    [[ -f $A/state/copied_$d.ok ]] && continue
    left=$((left + 1))
    [[ -f $A/stores/check/in_$d.done ]] || continue
    mkdir $A/state/cplock_$d 2>/dev/null || continue
    J=$JD/mi_$(mi_set_of $d)_eefo_o320_pf10_l012-240
    ssh jupiter "mkdir -p $J/in $J/check"
    t0=$(date +%s)
    rsync -a $A/stores/in/$d.zarr jupiter:$J/in/ && rsync -a $A/stores/check/in_$d.* jupiter:$J/check/
    a=$(lsum $A/stores/in/$d.zarr); j=$(ssh jupiter "find $J/in/$d.zarr -type f -printf '%P %s\n' | sort | md5sum | cut -c1-32")
    b=$(du -sb --apparent-size $A/stores/in/$d.zarr | cut -f1)
    if [[ -n "$a" && "$a" == "$j" ]]; then date -u +%FT%TZ > $A/state/copied_$d.ok; echo "OK $d $b B $(( $(date +%s) - t0 )) s $tag $(date -u +%FT%TZ)"
    else echo "MISMATCH $d atos=$a jupiter=$j $tag $(date -u +%FT%TZ)"; fi
    rmdir $A/state/cplock_$d
  done
  [[ $left == 0 ]] && { echo "COPY_SET_DONE $list $tag $(date -u +%FT%TZ)"; exit 0; }
  sleep 300
done
echo "COPY_DRIVER_TIMEOUT $tag left=$left $(date -u +%FT%TZ)"
