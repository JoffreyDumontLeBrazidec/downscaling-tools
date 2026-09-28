#!/bin/bash
#SBATCH --job-name=sigma_loss_smoke
#SBATCH --nodes=1
#SBATCH --qos=ng
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/2026-06-17/sigma_loss_smoke-%j.out

set -eux

# Legacy DS runtime: required to unpickle cfec83a3-era (ds_tensor) checkpoints.
source /home/ecm5702/dev/runtimes/ds/activate.sh

export ANEMOI_BASE_SEED=1234
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=1

WORKDIR=/home/ecm5702/dev/downscaling-tools
cd "$WORKDIR"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

CKPT=/home/ecm5702/scratch/aifs/checkpoint/cfec83a3cd0644778e2bfcbacfa9f4fc/anemoi-by_epoch-epoch_043-step_200000.ckpt
OUTDIR=/ec/res4/scratch/ecm5702/eval/o96_o320/sigma_loss_smoke
DUMMY_PRED="${OUTDIR}/_dummy_predictions"
mkdir -p "$DUMMY_PRED"

# sigma_loss requires --checkpoint (runs the model). --predictions-dir is required
# by the evaluate subcommand but unused by this evaluator.
srun python -u -m eval.cli evaluate \
    --lane o96_o320_sigma_loss_smoke \
    --host atos_ac \
    --predictions-dir "$DUMMY_PRED" \
    --checkpoint "$CKPT" \
    --only sigma_loss \
    --output-dir "$OUTDIR" \
    --overwrite

echo "=== per_sigma.csv ==="
find "$OUTDIR" -name per_sigma.csv -exec head -20 {} \;
echo "=== metrics.json ==="
find "$OUTDIR" -name metrics.json -path '*sigma_loss*' -exec cat {} \;
echo "=== plots ==="
find "$OUTDIR" -name '*.png' -path '*sigma_loss*'
