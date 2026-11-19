#!/bin/bash
# MI4-S26 copy driver Atos -> Jupiter (19 Nov 2026), from kits/20261117-month-inference/mi_copy_driver.sh. Run 3 at once under nohup
# ON the Atos login node that holds the Jupiter ssh master (ac6-100/ac6-101): each takes the next built date, rsyncs its store,
# verifies it by the md5 of the sorted (relative path, size) list on both sides (never du), writes state/copied_<d>.ok, and THEN
# deletes the Atos copy of that store (staging only; the kept native GRIB in keep/ stays). Loops every 5 min until the list is done.
# Usage: nohup bash mi4_copy_driver.sh <dates file> <stream tag> [MAXR=1200] > $MI4_A/logs/cp_<tag>.log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); source $HERE/mi4_set.sh
A=$MI4_A; J=$MI4_JD; list=${1:?dates file}; tag=${2:?tag}; maxr=${3:-1200}
mapfile -t dates < <(grep -E '^[0-9]{8}' "$list" | cut -c1-8)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=25"
lsum() { find "$1" -type f -printf '%P %s\n' | sort | md5sum | cut -c1-32; }
$SSH jupiter "mkdir -p $J/in $J/check" || { echo "FATAL no jupiter ssh $(hostname)"; exit 1; }
for ((r = 1; r <= maxr; r++)); do
  left=0
  [[ -f $A/state/STOP ]] && { echo "STOP $tag $(date -u +%FT%TZ)"; exit 0; }
  for d in "${dates[@]}"; do
    [[ -f $A/state/copied_$d.ok ]] && continue
    left=$((left + 1))
    [[ -f $A/stores/check/in_$d.done && -d $A/stores/in/$d.zarr ]] || continue
    mkdir $A/state/cplock_$d 2>/dev/null || continue
    t0=$(date +%s); b=$(du -sb --apparent-size $A/stores/in/$d.zarr | cut -f1)
    rsync -a -e "$SSH" $A/stores/in/$d.zarr jupiter:$J/in/ && rsync -a -e "$SSH" $A/stores/check/in_$d.* jupiter:$J/check/
    a=$(lsum $A/stores/in/$d.zarr); j=$($SSH jupiter "find $J/in/$d.zarr -type f -printf '%P %s\n' | sort | md5sum | cut -c1-32")
    if [[ -n "$a" && "$a" == "$j" ]]; then
      date -u +%FT%TZ > $A/state/copied_$d.ok
      echo "OK $d $b B $(( $(date +%s) - t0 )) s $(( b / ($(date +%s) - t0 + 1) / 1000000 )) MB/s $tag $(date -u +%FT%TZ)"
      rm -rf $A/stores/in/$d.zarr && echo "ATOS_STORE_DELETED $d"
    else echo "MISMATCH $d atos=$a jupiter=$j $tag $(date -u +%FT%TZ)"; fi
    rmdir $A/state/cplock_$d
  done
  [[ $left == 0 ]] && { echo "COPY_SET_DONE $list $tag $(date -u +%FT%TZ)"; exit 0; }
  sleep 300
done
echo "COPY_DRIVER_TIMEOUT $tag left=$left $(date -u +%FT%TZ)"
