#!/bin/bash
# o2560 box read: login-node CPU dry runs (no GPU, no sampling). A100 over all 12 dates x 12 leads 6-72 h (every row of the
# split resolves, truth and input finite, tp box means per lead, EMA file differs from raw); B24 and B50 on one case each
# (their own code pin and weights load strictly). Logs in logs/dry_<row>.log; prints DRY_ALL_OK or the failing rows.
KITK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/kit
LOGD=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs
bad=""
for spec in "A100:all" "B24:20241105:48" "B50:20241112:72"; do
  row=${spec%%:*}; rest=${spec#*:}
  ( source $KITK/rows.sh $row || exit 9
    if [[ $rest == all ]]; then c=""; for L in 6 12 18 24 30 36 42 48 54 60 66 72; do c="$c${c:+,}$(cases_for $L)"; done; else c=$rest; fi
    source $KITK/env.sh || exit 9
    export MODE=dry OMP_NUM_THREADS=16 OUTDIR=$OUT/dry/$row CASES=$c
    cd /tmp && python $KITK/o2560_box_predict.py ) > $LOGD/dry_$row.log 2>&1
  rc=$?; echo "[dry] $row rc=$rc $(grep -c 'DRY case' $LOGD/dry_$row.log) cases $(date -u +%FT%TZ)"
  grep -q DRY_DONE $LOGD/dry_$row.log && [[ $rc == 0 ]] || { bad="$bad $row"; tail -20 $LOGD/dry_$row.log; }
done
[[ -z "$bad" ]] && echo "DRY_ALL_OK $(date -u +%FT%TZ)" || echo "DRY_ALL_FAIL$bad $(date -u +%FT%TZ)"
