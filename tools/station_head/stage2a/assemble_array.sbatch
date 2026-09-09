#!/bin/bash
#SBATCH --job-name=sh2a_assemble
#SBATCH --qos=nf
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-11
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%A_%a.out
# Build the training arrays of the station head from the gathered case tables, one
# npz file per case and per target variable. Twelve array tasks cover the three
# targets with four shards each: task i works on target i/4 and shard i%4.
#
#   sbatch --export=ALL,TAG=firstcut assemble_array.sbatch
set -uo pipefail
S=/home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a
TAG=${TAG:-firstcut}
TARGETS=(2t 2d 10ff)
I=${SLURM_ARRAY_TASK_ID:-0}
T=${TARGETS[$((I / 4))]}
SHARD=$((I % 4))
module load python3 vtb ecmwf-toolbox
echo "target=$T shard=$SHARD of 4 tag=$TAG"
python3 -u "$S/dataset.py" --target "$T" --tag "$TAG" --shard "$SHARD" --nshards 4
echo "ASSEMBLE_RC=$?"
