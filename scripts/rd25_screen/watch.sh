#!/bin/bash
set -euo pipefail
WORK=/home/ecm5702/agent-work/20260915-rd25-evaluation
LOCK=$WORK/outputs/watch.lock
mkdir "$LOCK" || { echo "A watcher lock already exists."; exit 2; }
trap 'mv "$LOCK" "$LOCK.finished.$$"' EXIT
for iteration in $(seq 1 120); do
  todo=0
  for pair in RD25k:50063675 R47k:50063676; do
    ARM=$(echo "$pair" | cut -d: -f1)
    JOB=$(echo "$pair" | cut -d: -f2)
    if test -f "$WORK/outputs/scoring_$ARM.job"; then continue; fi
    todo=1
    STATE=$(ssh -o BatchMode=yes -o ConnectTimeout=15 ag-login "sacct -j $JOB -X -n -P --format=State" | head -n1 | cut -d'|' -f1)
    echo "$(date -u +%FT%TZ) $ARM $JOB $STATE"
    case "$STATE" in
      COMPLETED)
        sbatch --parsable --job-name=rd25_score_$ARM "$WORK/tools/scripts/rd25_screen/score.sbatch" "$ARM" > "$WORK/outputs/scoring_$ARM.job"
        cat "$WORK/outputs/scoring_$ARM.job"
        ;;
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
        echo "Prediction failed. Scoring was not submitted for $ARM."; exit 3 ;;
    esac
  done
  if test "$todo" = 0; then echo "Both scoring jobs are submitted."; exit 0; fi
  sleep 60
done
echo "The two-hour watch limit was reached; inspect the queue before resuming."
exit 4
