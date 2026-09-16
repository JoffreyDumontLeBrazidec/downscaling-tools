#!/bin/bash
# Continue the two already-submitted AG screens for at most 48 hours.
# Run on ac6-100. Existing job markers are retained across a restart.
set -euo pipefail
WORK=/home/ecm5702/agent-work/20260915-rd25-evaluation
LOCK=$WORK/outputs/watch.lock
mkdir "$LOCK" || { echo "A watcher lock already exists."; exit 2; }
printf '%s\n' "$$" > "$LOCK/pid"
trap 'mv "$LOCK" "$LOCK.finished.$$"' EXIT
deadline=$(( $(date +%s) + 48 * 3600 ))
printf '%s\n' "$deadline" > "$LOCK/deadline_unix"
while test "$(date +%s)" -lt "$deadline"; do
  todo=0
  for pair in RD25k:50063675 R47k:50063676; do
    ARM=${pair%%:*}
    JOB=${pair##*:}
    marker="$WORK/outputs/scoring_$ARM.job"
    if test -s "$marker"; then continue; fi
    if test -f "$WORK/outputs/prediction_$ARM.failed"; then continue; fi
    if test -e "$marker" || test -e "$marker.submitting"; then
      echo "An incomplete submission record exists for $ARM. Inspect sacct before resuming."
      exit 5
    fi
    todo=1
    if ! raw=$(ssh -o BatchMode=yes -o ConnectTimeout=15 ag-login "sacct -j $JOB -X -n -P --format=State"); then
      echo "The AG state query failed. The watcher stops; no authentication retry was made."
      exit 6
    fi
    STATE=$(printf '%s\n' "$raw" | head -n1 | cut -d'|' -f1)
    echo "$(date -u +%FT%TZ) $ARM $JOB $STATE"
    case "$STATE" in
      COMPLETED)
        if ! sbatch --parsable --job-name=rd25_score_$ARM "$WORK/tools/scripts/rd25_screen/score.sbatch" "$ARM" > "$marker.submitting"; then
          echo "The scoring submission failed or is uncertain for $ARM. Inspect the saved response and sacct."
          exit 7
        fi
        if ! grep -Eq '^[0-9]+(;[^[:space:]]+)?$' "$marker.submitting"; then
          echo "The scoring response for $ARM is not a job id. Inspect it before resuming."
          exit 8
        fi
        mv "$marker.submitting" "$marker"
        cat "$marker"
        ;;
      FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*)
        printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$JOB" "$STATE" > "$WORK/outputs/prediction_$ARM.failed"
        echo "Prediction failed for $ARM. Scoring was not submitted; the other arm remains watched."
        ;;
    esac
  done
  if test "$todo" = 0; then echo "Each arm has a scoring submission or a recorded prediction failure."; exit 0; fi
  sleep 60
done
echo "The 48-hour watch limit was reached. Inspect the queue before resuming."
exit 4
