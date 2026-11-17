#!/bin/bash
# Bring one model row of the extremes set (prediction files and its global histograms) from Jupiter to Atos for scoring.
# Run it ON ac6-100. rsync skips what is there; nothing is deleted on either side.
# Usage: bash xset_copy_from_jupiter.sh <ROW>
set -eu
row=${1:?row}
A=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}
XOUT=${XSET_JOUT:-/e/data1/jureap-data/ecmwf/users/jdumont/outputs/xset1_20261116}
mkdir -p $A/rows/$row $A/hist
rsync -a --info=stats1 --exclude='*.part' jupiter:$XOUT/rows/$row/ $A/rows/$row/ | grep -E "Total transferred" || true
rsync -a --include="${row}_*.npz" --exclude='*' jupiter:$XOUT/hist/ $A/hist/
echo "files: $(ls $A/rows/$row/predictions_*_step*.nc | wc -l) of 252, histograms: $(ls $A/hist/${row}_*.npz | wc -l) of 252, $(du -sh $A/rows/$row | cut -f1)"
echo "COPY_BACK_DONE $row $(date -u +%FT%TZ)"
