#!/bin/bash
# CorrDiff on Jupiter: training body (called by cd_train_jupiter.sbatch and cd_smoke_jupiter.sbatch).
# cd_torchrun MODE CKPT_DIR LOG [hydra overrides...]  -> torchrun on every node of the allocation, rc of srun
cd_torchrun() {
  local MODE=$1 DIR=$2 LOG=$3; shift 3
  local MASTER; MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
  export MASTER_ADDR=$MASTER MASTER_PORT=29511
  export NCCL_IB_TIMEOUT=30 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1
  cd $CD_REPO/examples/weather/corrdiff
  echo "[torchrun] $(date -u) MODE=$MODE nodes=$SLURM_NNODES dir=$DIR overrides: $*" | tee -a $LOG
  srun --kill-on-bad-exit=1 --ntasks-per-node=1 bash -c "torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --node_rank=\$SLURM_NODEID \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29511 --rdzv_id=$SLURM_JOB_ID \
    train.py --config-name=config_training_e2v4_${MODE}.yaml training.io.checkpoint_dir=$DIR $*" >> $LOG 2>&1
  local rc=$?
  echo "[torchrun] $(date -u) rc=$rc" | tee -a $LOG
  return $rc
}
is_oom() { grep -qE "OutOfMemoryError|CUDA out of memory" "$1"; }
# last progress line of a log, plus the maximum reserved GPU memory over all its ticks
last_tick() { echo "$(grep sec_per_sample "$1" | tail -1 | sed 's/.*\] - //' | cut -c1-400) | max_reserved_gb $(grep -oE 'peak_gpu_mem_reserved_gb [0-9.]+' "$1" | awk '$2>m{m=$2} END{print m+0}') | ticks $(grep -c sec_per_sample "$1")"; }
