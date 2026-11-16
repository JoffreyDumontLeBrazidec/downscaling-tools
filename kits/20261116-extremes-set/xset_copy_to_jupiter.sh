#!/bin/bash
# Copy what the GPU cluster needs from Atos to Jupiter: input stores, 9 km ensemble row files (the template of every model
# row), histograms, check files, points and the station cache. Run it ON ac6-100 (the ssh master to Jupiter lives there).
# rsync skips what is already there, so it can be called after each group of dates. Nothing is deleted on either side.
# Usage: bash xset_copy_to_jupiter.sh
set -eu
A=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set
XK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set
XOUT=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116
ssh jupiter "mkdir -p $XK/stores/in $XK/points $XK/cache $XOUT/rows/ENS9 $XOUT/hist $XOUT/check"
# only complete stores (those with a check file) and complete row dates
for f in $A/stores/check/in_*.done; do
  [[ -f $f ]] || continue
  d=$(basename $f .done); d=${d#in_}
  rsync -a --info=stats1 $A/stores/in/$d.zarr jupiter:$XK/stores/in/ | grep -E "Total transferred|bytes/sec" | tr '\n' ' '; echo " store $d"
done
for f in $A/check/rows_*.done; do
  [[ -f $f ]] || continue
  d=$(basename $f .done); d=${d#rows_}
  rsync -a $A/rows/ENS9/predictions_${d}_step*.nc jupiter:$XOUT/rows/ENS9/
  rsync -a $A/hist/ENS9_${d}_step*.npz $A/hist/INPUT_${d}_step*.npz jupiter:$XOUT/hist/
  rsync -a $A/check/rows_${d}.* jupiter:$XOUT/check/
  rsync -a $A/points/points_${d}.npz jupiter:$XK/points/
  echo "rows $d copied"
done
[[ -d $A/cache/xset1_v1 ]] && rsync -a $A/cache/xset1_v1 jupiter:$XK/cache/
ssh jupiter "ls $XK/stores/in | wc -l; ls $XOUT/rows/ENS9 | wc -l; du -sh $XK/stores/in $XOUT/rows/ENS9 $XOUT/hist"
echo "COPY_DONE $(date -u +%FT%TZ)"
