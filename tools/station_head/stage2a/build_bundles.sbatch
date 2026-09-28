#!/bin/bash
#SBATCH --job-name=sh2a_bundle
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%A_%a.out
# Build the truth-free input bundles of one slice of the stage 2a case list.
#
# The station head does not need a truth field in the bundle: it is scored against
# station reports, and the analysis control comes from the separate
# analysis_at_stations tables. Leaving the target out keeps a bundle at about half
# a gigabyte instead of two, which is what makes the first cut fit on scratch. The
# builder is therefore called with --allow-missing-target-unsafe, the harness's own
# explicit prediction-only escape hatch.
set -uo pipefail
module load ecmwf-toolbox
export DATA_DIR=/home/mlx/ai-ml/datasets/ DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /home/ecm5702/hpcperm/sandbox/20260902-hres-local-branch/activate.sh || exit 3
cd /home/ecm5702/dev/downscaling-tools
EXTRA=""
if [ -n "${ONLY_CASE:-}" ]; then EXTRA="--only-case ${ONLY_CASE}"; fi
if [ -n "${ONLY_MEMBERS:-}" ]; then EXTRA="${EXTRA} --only-members ${ONLY_MEMBERS}"; fi
if [ -n "${SPLIT:-}" ]; then EXTRA="${EXTRA} --split ${SPLIT}"; fi
env -u PYTHONPATH python -u /home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a/build_bundles.py \
  --slice "${SLURM_ARRAY_TASK_ID:-0}" --nslices "${NSLICES:-1}" ${EXTRA}
echo "BUNDLE_RC=$?"
