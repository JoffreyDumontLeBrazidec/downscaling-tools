#!/bin/bash
# CorrDiff on Jupiter: training chain segment (4 nodes x 4 GH200, 12 h). Fresh run folder $CD/run/<reg|diff>,
# PhysicsNeMo's own resume from the newest checkpoint there. Self-extending chain: once a segment logs its first progress
# line it submits its successor (afterany on itself) unless state/chain_<mode>/DONE or STOP exists or CD_MAX_SEG
# segments were started (no cascade of failing segments); the segment that sees "Training Completed" writes DONE and its
# successor exits at once. CD_VAL_FREQ: validation every N samples (default 50,000).
# Checkpoint level: state/chain_<mode>/LEVEL (set by an earlier segment) else state/REG_LEVEL (smoke) else 0; on an
# out-of-memory failure before any checkpoint the next level is tried in the same allocation (same model and loss).
#   sbatch --export=ALL,MODE=regression cd_train_jupiter.sbatch
#   sbatch --export=ALL,MODE=diffusion,CD_REG_CKPT=<regression .mdlus> cd_train_jupiter.sbatch
#   stop the chain: touch $CD/state/chain_<mode>/STOP
#SBATCH --job-name=cd_reg
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/logs/%x_%j.out
set -uo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/kit/cd_env_jupiter.sh
python -c "import physicsnemo, torch; print(physicsnemo.__version__, torch.cuda.is_available())"
source $CD_KIT/cd_data_gate_jupiter.sh; source $CD_KIT/cd_train_jupiter_body.sh
MODE=${MODE:-regression}; [[ $MODE == regression ]] && TAG=reg || TAG=diff
ST=$CD/state/chain_$MODE; RUN=$CD/run/$TAG; mkdir -p $ST $RUN
MAX_SEG=${CD_MAX_SEG:-6}
echo "segment start $(date -u) job $SLURM_JOB_ID host $(hostname) MODE=$MODE run=$RUN repo $(cat $CD/PHYSICSNEMO_REV)"
[[ -f $ST/DONE ]] && { echo "CHAIN_DONE: $(cat $ST/DONE)"; exit 0; }
[[ -f $ST/STOP ]] && { echo "CHAIN_STOP present"; exit 0; }
[[ $MODE == diffusion ]] && { : "${CD_REG_CKPT:?CD_REG_CKPT=<regression .mdlus>}"; [[ -f $CD_REG_CKPT ]] || { echo "FATAL $CD_REG_CKPT missing"; exit 1; }; }
SEG=$(( $(cat $ST/SEGMENTS 2>/dev/null || echo 0) + 1 )); echo $SEG > $ST/SEGMENTS
echo "$SLURM_JOB_ID seg $SEG start $(date -u)" >> $ST/jobs.txt
cd_data_gate || exit 1
L0=$(cat $ST/LEVEL 2>/dev/null || cat $CD/state/REG_LEVEL 2>/dev/null || echo 0)
[[ $MODE == diffusion ]] && L0=$(cat $ST/LEVEL 2>/dev/null || echo 1)
for L in $(seq $L0 3); do
  echo $L > $ST/LEVEL
  LOG=$CD/logs/${TAG}_seg${SEG}_${SLURM_JOB_ID}_L$L.log; : > $LOG
  cd_torchrun $MODE $RUN $LOG training.perf.songunet_checkpoint_level=$L training.io.validation_freq=${CD_VAL_FREQ:-50000} &
  TP=$!; SUBMITTED=0
  # the successor is submitted only once this segment logs its first progress line (no cascade of failing segments)
  while kill -0 $TP 2>/dev/null; do
    if (( SUBMITTED == 0 )) && grep -q sec_per_sample $LOG; then
      SUBMITTED=1; echo "first tick $(date -u): $(last_tick $LOG)"
      if (( SEG < MAX_SEG )) && [[ ! -f $ST/STOP && ! -f $ST/DONE ]]; then
        NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID --export=ALL,MODE=$MODE $CD_KIT/cd_train_jupiter.sbatch)
        echo "successor $NEXT (afterany:$SLURM_JOB_ID) seg $((SEG+1)) $(date -u)" | tee -a $ST/jobs.txt
      fi
    fi
    sleep 30
  done
  wait $TP; rc=$?
  if grep -q "Training Completed" $LOG; then echo "Training Completed job $SLURM_JOB_ID $(date -u) $(last_tick $LOG)" > $ST/DONE; cat $ST/DONE; exit 0; fi
  if is_oom $LOG && ! grep -q "sec_per_sample" $LOG; then echo "L$L out of memory before the first tick, next level"; continue; fi
  echo "segment end rc=$rc $(date -u) last tick: $(last_tick $LOG)"; exit $rc
done
echo "FATAL: no checkpoint level fits"; exit 1
