#!/bin/bash
# MI-S25: copy the finished input stores (those with a .done check) from Atos to the durable Jupiter dataset folder.
# Run ON ac6-100 (the ssh master to Jupiter lives there). rsync skips what is already there; nothing is deleted on either side.
set -eu
source "$(cd "$(dirname "$0")" && pwd)/mi_set.sh"
A=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference
JD=/e/data1/jureap-data/ecmwf/users/jdumont/datasets
for f in $A/stores/check/in_*.done; do
  [[ -f $f ]] || continue
  d=$(basename $f .done); d=${d#in_}
  J=$JD/mi_$(mi_set_of $d)_eefo_o320_pf10_l012-240; ssh jupiter "mkdir -p $J/in $J/check"
  rsync -a --info=stats1 $A/stores/in/$d.zarr jupiter:$J/in/ | grep -E "Total transferred|bytes/sec" | tr '\n' ' '; echo " store $d"
  rsync -a $A/stores/check/in_$d.* jupiter:$J/check/
done
for s in s25 j25 jul25; do ssh jupiter "ls $JD/mi_${s}_eefo_o320_pf10_l012-240/in 2>/dev/null | wc -l"; done
echo "COPY_DONE $(date -u +%FT%TZ)"
