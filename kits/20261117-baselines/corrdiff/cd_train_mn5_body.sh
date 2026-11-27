#!/bin/bash
# CorrDiff on MN5: training body (sourced by cd_train_mn5.sbatch and cd_smoke_mn5.sbatch, after cd_env_mn5.sh).
# cd_srun MODE DIR LOG [hydra overrides...] -> train_cd.py on every GPU of the allocation (srun, 4 tasks per node, one
# per H100; PhysicsNeMo's DistributedManager reads SLURM_PROCID / SLURM_LOCALID), rc of srun. Config
# config_training_e2v4${CD_CONF_SUFFIX}_MODE.yaml from $CD_CONF_DIR. train_cd.py = train.py + train_cd_v2.2.2.patch.
cd_srun() {
  local MODE=$1 DIR=$2 LOG=$3; shift 3
  # a new port per call: an out-of-memory retry in the same allocation must not hit "address in use" on the old one
  export MASTER_PORT=$(( 20000 + (RANDOM * 7 + SECONDS) % 20000 ))
  export NCCL_IB_TIMEOUT=30 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1 TORCH_COMPILE_DISABLE=1
  export OMP_NUM_THREADS=4
  cd $CD_REPO/examples/weather/corrdiff
  echo "[srun] $(date -u) MODE=$MODE nodes=$SLURM_NNODES port=$MASTER_PORT dir=$DIR config=$CD_CONF_DIR/config_training_e2v4${CD_CONF_SUFFIX}_${MODE}.yaml overrides: $*" | tee -a $LOG
  srun --kill-on-bad-exit=1 --nodes=$SLURM_NNODES --ntasks-per-node=4 --cpus-per-task=20 --cpu-bind=none \
    python train_cd.py --config-dir=$CD_CONF_DIR --config-name=config_training_e2v4${CD_CONF_SUFFIX}_${MODE}.yaml \
    training.io.checkpoint_dir=$DIR "$@" >> $LOG 2>&1
  local rc=$?
  echo "[srun] $(date -u) rc=$rc" | tee -a $LOG
  return $rc
}
is_oom() { grep -qE "OutOfMemoryError|CUDA out of memory" "$1"; }
# last progress line of a log, plus the maximum reserved GPU memory over all its ticks
last_tick() { echo "$(grep sec_per_sample "$1" | tail -1 | sed 's/.*\] - //' | cut -c1-400) | max_reserved_gb $(grep -oE 'peak_gpu_mem_reserved_gb [0-9.]+' "$1" | awk '$2>m{m=$2} END{print m+0}') | ticks $(grep -c sec_per_sample "$1")"; }
# mean sec_per_sample over the ticks after the first (the first holds start-up), and mean data wait per tick
tick_stats() { grep sec_per_sample "$1" | sed 's/.*\] - //' | awk '{for(i=1;i<NF;i++){if($i=="sec_per_sample")s[NR]=$(i+1); if($i=="sec_per_tick")t[NR]=$(i+1); if($i=="sec_data_wait")w[NR]=$(i+1)}} END{n=0; for(k=2;k<=NR;k++){S+=s[k];T+=t[k];W+=w[k];n++} if(n>0) printf "sec_per_sample_mean %.4f sec_per_tick_mean %.3f sec_data_wait_mean %.3f over %d ticks", S/n, T/n, W/n, n; else print "no ticks after the first"}'; }
# Slurm time limit (D-HH:MM:SS, HH:MM:SS or MM:SS) -> seconds
tlim_s() { echo "$1" | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>=3)s+=3600*$(n-2); if(n>=4)s+=86400*$(n-3); print s}'; }
# export done and verified (cd_export_mn5.sbatch writes $CD/state/VERIFY_OK)
cd_data_ok() {
  [[ -f $CD/state/VERIFY_OK ]] || { echo "FATAL: $CD/state/VERIFY_OK missing: the export is not verified (cd_export_mn5.sbatch)"; return 1; }
  [[ -f ${CD_DATA%.zarr}.stats.json ]] || { echo "FATAL: ${CD_DATA%.zarr}.stats.json missing"; return 1; }
  echo "[data] $(cat $CD/state/VERIFY_OK)"
}
cd_code_ok() {
  grep -q CD_KIT_PATCH $CD_REPO/examples/weather/corrdiff/train_cd.py 2>/dev/null || { echo "FATAL: train_cd.py missing (cd_apply_train_patch.sh)"; return 1; }
  echo "[code] kit $(cat $CD_KIT/KIT_SHA 2>/dev/null || echo NO_KIT_SHA) physicsnemo $(git -C $CD_REPO rev-parse --short HEAD 2>/dev/null || cat $CD/PHYSICSNEMO_REV)"
}
