#!/bin/bash
# MI-S25: measured sizes and times on Atos (GRIB per file, store per date, retrieval and build seconds from the logs).
A=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference
echo "== GRIB files (bytes)"; for f in $A/grib/*.grib; do [[ -f $f.ok ]] && echo "$(basename $f) $(stat -c %s $f) fields=$(cat $f.ok)"; done
echo "== stores (apparent bytes, files)"; for s in $A/stores/in/*.zarr; do [[ -d $s ]] && echo "$(basename $s) $(du -s --apparent-size -B1 $s | cut -f1) $(find $s -type f | wc -l)"; done
echo "== totals"; du -s --apparent-size -B1 $A/grib $A/stores/in 2>/dev/null
echo "== retrieval seconds"; grep -h "done in_" $A/logs/mi_ret_*.out 2>/dev/null | awk '{print $4, $6, $7}'
echo "== builds"; grep -h -E "BUILT in|Maximum resident|Elapsed" $A/logs/mi_bin_*.out 2>/dev/null
