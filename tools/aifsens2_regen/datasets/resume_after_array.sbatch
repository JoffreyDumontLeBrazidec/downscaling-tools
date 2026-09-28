#!/bin/bash
#SBATCH --job-name=aifsv2-full2026-resume
#SBATCH --qos=nf
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --output=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/runs/full2026/logs/%x_%j.out
# Runs after a load array ends (afterany). Any part that did not COMPLETE (timeout, failure, cancel)
# is resubmitted with a 24 h limit; a rerun part skips the groups its registry already flags as done.
# Then finalise, the verifier and another resume round are chained behind the rerun.
set -uo pipefail
S=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build
ARR=${ARR:?set ARR to the load array id}
sacct -j $ARR -X -n --format=JobID%-20,State%-14 | sed "s/  */ /g"
bad=$(sacct -j $ARR -X -n --format=JobID%-20,State%-14 | awk "\$2!=\"COMPLETED\"{print \$1}" | grep "_" | sed "s/.*_//" | grep -E "^[0-9]+$" | sort -n | uniq | tr "\n" "," | sed "s/,\$//")
if [ -z "$bad" ]; then echo "RESUME: all parts of $ARR completed; finalise chain proceeds"; exit 0; fi
echo "RESUME: parts not completed in $ARR: $bad"
for j in ${CANCEL_ON_RESUME:-}; do scancel $j && echo "cancelled stale dependant $j"; done
JL=$(sbatch --parsable --time=24:00:00 --array=$bad $S/runs/full2026/load.sbatch)
JF=$(sbatch --parsable --dependency=afterok:$JL $S/runs/full2026/finalise.sbatch)
JV=$(sbatch --parsable --dependency=afterany:$JF $S/verify_full2026.sbatch)
JR=$(sbatch --parsable --dependency=afterany:$JL --export=ALL,ARR=$JL,CANCEL_ON_RESUME="$JF $JV" $S/runs/full2026/resume_after_array.sbatch)
echo "RESUBMITTED load=$JL finalise=$JF verify=$JV next_resume=$JR"
