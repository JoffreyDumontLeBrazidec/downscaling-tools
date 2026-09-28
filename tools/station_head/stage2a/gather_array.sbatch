#!/bin/bash
#SBATCH --job-name=sh2a_gather
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%A_%a.out
# Gather the station feature table of every stage 2a case whose prediction file is
# finished and which has not been gathered yet. Eight array tasks share the pending
# list; each one re-checks that a prediction file is complete before reading it, so
# the job is safe to run while the inference arrays are still producing files.
#
# Submit with:  sbatch --array=0-7 gather_array.sbatch
# The manifest must be refreshed first with refresh_manifest.sh.
set -uo pipefail
S=/home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-8}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
module load python3 vtb ecmwf-toolbox
python3 -u "$S/gather_driver.py" --shard "$SHARD" --nshards "$NSHARDS" "$@"
echo "GATHER_RC=$?"
