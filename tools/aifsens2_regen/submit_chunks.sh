#!/bin/bash
# Submit the chunked retrieval arrays for every (block, group, input time) still missing, waiting
# for the account's submission quota when it refuses. Records the job ids in
# ic/raw/<block>/.live_jobs so the state is visible to anyone who looks.
#
# The order matters: February first so that one more month can be assembled and forecast, then
# March. April and May were submitted by hand already.
set -uo pipefail
T=/home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen
R=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/ic/raw
cd "$T" || exit 1

# block group time   (February's sfc_0000 is deliberately absent: its monthly request is being served)
WORK="
202602 sfc 1200
202602 q 0000
202602 q 0600
202602 q 1200
202602 q 1800
202603 sfc 0000
202603 sfc 0600
202603 pl 0000
202603 pl 0600
202603 pl 1200
202603 pl 1800
202603 q 0000
202603 q 0600
202603 q 1200
202603 q 1800
"

echo "$WORK" | while read -r b g t; do
    [ -z "$b" ] && continue
    [ -f "$R/$b/${g}_${t}.grib" ] && { echo "$(date -u +%H:%M) $b $g $t already on disk, skipping"; continue; }
    for try in $(seq 1 60); do
        if j=$(sbatch --parsable --array=1-4%2 retrieve_chunk.sbatch "$b" "$g" "$t" 2>/dev/null); then
            echo "$(date -u +%H:%M) SUBMITTED $b $g $t = $j"
            cur=$(cat "$R/$b/.live_jobs" 2>/dev/null || true)
            printf '%s' "${cur:+$cur,}$j" > "$R/$b/.live_jobs"
            break
        fi
        sleep 120
    done
done
echo "$(date -u +%H:%M) all chunk arrays submitted or given up on"
for b in 202602 202603 202604 202605; do echo "$b -> $(cat $R/$b/.live_jobs 2>/dev/null)"; done
