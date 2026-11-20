#!/bin/bash
# CorrDiff on Jupiter: training body (called by cd_train_jupiter.sbatch and cd_smoke_jupiter.sbatch).
# cd_torchrun MODE CKPT_DIR LOG [hydra overrides...]  -> torchrun on every node of the allocation, rc of srun
cd_torchrun() {
  local MODE=$1 DIR=$2 LOG=$3; shift 3
  local MASTER; MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
  # a new port per call: an out-of-memory retry in the same allocation must not hit "address in use" on the old one
  local PORT=$(( 20000 + (RANDOM * 7 + SECONDS) % 20000 ))
  export MASTER_ADDR=$MASTER MASTER_PORT=$PORT
  export NCCL_IB_TIMEOUT=30 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1
  cd $CD_REPO/examples/weather/corrdiff
  echo "[torchrun] $(date -u) MODE=$MODE nodes=$SLURM_NNODES port=$PORT dir=$DIR overrides: $*" | tee -a $LOG
  # Slurm >= 22.05: a step does not inherit --cpus-per-task from sbatch; give it the whole node, no binding (torchrun
  # spawns 4 ranks x 4 loader workers under this one task)
  srun --kill-on-bad-exit=1 --ntasks-per-node=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-288} --cpu-bind=none bash -c "torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --node_rank=\$SLURM_NODEID \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$PORT --rdzv_id=$SLURM_JOB_ID \
    train.py --config-name=config_training_e2v4_${MODE}.yaml training.io.checkpoint_dir=$DIR $*" >> $LOG 2>&1
  local rc=$?
  echo "[torchrun] $(date -u) rc=$rc" | tee -a $LOG
  return $rc
}
is_oom() { grep -qE "OutOfMemoryError|CUDA out of memory" "$1"; }
# last progress line of a log, plus the maximum reserved GPU memory over all its ticks
last_tick() { echo "$(grep sec_per_sample "$1" | tail -1 | sed 's/.*\] - //' | cut -c1-400) | max_reserved_gb $(grep -oE 'peak_gpu_mem_reserved_gb [0-9.]+' "$1" | awk '$2>m{m=$2} END{print m+0}') | ticks $(grep -c sec_per_sample "$1")"; }
