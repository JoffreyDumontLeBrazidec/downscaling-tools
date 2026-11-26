#!/bin/bash
# CorrDiff baseline training, published defaults (make_cd_configs.py). MODE=regression first, then MODE=diffusion with
# CD_REG_CKPT=<regression .mdlus>. Resumes from the newest checkpoint in the run folder (PhysicsNeMo's own resume).
#   sbatch --export=ALL,MODE=regression cd_train.sbatch
#   sbatch --export=ALL,MODE=diffusion,CD_REG_CKPT=$CD/run/reg/checkpoints_regression/<file>.mdlus cd_train.sbatch
# Smoke: add SMOKE=1 (1 node, 30 min, training_duration 2560 samples, run folder $CD/smoke_<mode>).
#SBATCH --job-name=cd_train
#SBATCH --account=DestE_340_26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/logs/%x_%j.out
set -euo pipefail
source /leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/kit/cd_env.sh
: "${MODE:?MODE=regression or diffusion}"
cd $CD_REPO/examples/weather/corrdiff
EXTRA=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  EXTRA+=(training.hp.training_duration=2560 training.io.checkpoint_dir=$CD/smoke_$MODE training.io.print_progress_freq=256
          training.io.validation_freq=1280 training.io.save_checkpoint_freq=2560)
fi
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=29511
echo "start $(date -u) MODE=$MODE nodes=$SLURM_NNODES repo=$(git -C $CD_REPO describe --tags) data=$CD_DATA"
srun --kill-on-bad-exit=1 bash -c "torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --node_rank=\$SLURM_NODEID \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29511 --rdzv_id=$SLURM_JOB_ID \
  train.py --config-name=config_training_e2v4_${MODE}.yaml ${EXTRA[*]:-}"
echo "end $(date -u)"
