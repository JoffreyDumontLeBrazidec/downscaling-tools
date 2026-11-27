#!/bin/bash
# CorrDiff Europe on MN5: smoke of both stages with the m5 configs (regression on random 448 crops + EMA, patched
# diffusion + EMA), at most 30 min. Node count from the command line (1 node = the speed and memory smoke, 2 nodes =
# the scaling smoke, same script). Run folders $CD/smoke/<job id>/..., summary appended to $CD/state/SMOKE_SUMMARY.txt.
#  1. regression part 1: CD_SMOKE_REG samples (default 320) on crops, from CD_REG_INIT if set (a widened .mdlus) else
#     from scratch; batch ladder 4:0 2:1 1:2 1:3 (batch_size_per_gpu:checkpoint level, no accumulation) until one
#     fits; one full-box validation at that batch; checkpoint + EMA saved at the end. The rung that fits is written to
#     $CD/state/FIT_regression (read by cd_train_mn5.sbatch) on the 1-node smoke.
#  2. regression part 2: same run folder, training_duration doubled -> resumes from part 1 (model, Adam, EMA must load:
#     REG_RESUME_OK line) and saves again.
#  3. diffusion: CD_SMOKE_DIFF samples (default 320) with the part-2 regression checkpoint, the config's batch, level
#     ladder 1, 2, 3; then a resume part as in 2 (DIFF_RESUME_OK).
# Each part runs under its own timeout; the diffusion runs even if the regression resume failed.
#   sbatch --nodes=1 [--export=ALL,CD_REG_INIT=<widened .mdlus>] cd_smoke_mn5.sbatch
#   sbatch --nodes=2 [--export=ALL,CD_REG_INIT=<widened .mdlus>] cd_smoke_mn5.sbatch
#SBATCH --job-name=cd_m5_smoke
#SBATCH --account=ehpc01
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5/logs/%x_%j.out
set -uo pipefail
ulimit -c 0
source /gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5/kit/cd_env_mn5.sh
source $CD_KIT/cd_train_mn5_body.sh
NODES=$SLURM_JOB_NUM_NODES; GPUS=$(( NODES * 4 ))
S=$CD/smoke/$SLURM_JOB_ID; mkdir -p $S; SUM=$CD/state/SMOKE_SUMMARY.txt
say() { echo "[$SLURM_JOB_ID n$NODES] $*" | tee -a $SUM; }
say "smoke start $(date -u) host $(hostname) nodes $NODES"
python -c "import physicsnemo, torch; print('physicsnemo', physicsnemo.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())" | tee -a $SUM
cd_code_ok | tee -a $SUM; cd_data_ok | tee -a $SUM || exit 1
NR=${CD_SMOKE_REG:-320}; ND=${CD_SMOKE_DIFF:-320}
COMMON="training.io.validation_steps=2 wandb.mode=disabled"
summ() {  # stage label, log, nodes -> speed and memory line
  local ts; ts=$(tick_stats $2)
  local sps; sps=$(echo "$ts" | awk '{print $2}')
  say "$1 $(echo "$ts") | s_per_unit_per_node $(python -c "print(f'{float(\"${sps:-nan}\") * $3:.4f}')" 2>/dev/null) | peak_alloc_gb $(grep -oE 'peak_gpu_mem_gb [0-9.]+' $2 | awk '$2>m{m=$2} END{print m+0}') peak_reserved_gb $(grep -oE 'peak_gpu_mem_reserved_gb [0-9.]+' $2 | awk '$2>m{m=$2} END{print m+0}') (rank 0)"
}

# 1-2. regression on crops, save, resume
[[ -n "${CD_REG_INIT:-}" ]] && say "regression start weights $CD_REG_INIT"
FIT=""
for RUNG in 4:0 2:1 1:2 1:3; do
  B=${RUNG%:*}; L=${RUNG#*:}; T=$(( B * GPUS ))
  R=$S/reg_b${B}_L$L; mkdir -p $R/checkpoints_regression   # a fresh run folder per try, each from the same start
  [[ -n "${CD_REG_INIT:-}" ]] && cp $CD_REG_INIT $R/checkpoints_regression/CorrDiffRegressionUNet.0.0.mdlus
  LOG=$S/reg_p1_b${B}_L$L.log; : > $LOG
  CD_CAP="timeout --signal=TERM --kill-after=60 420" cd_srun regression $R $LOG training.hp.batch_size_per_gpu=$B training.hp.total_batch_size=$T \
    training.perf.songunet_checkpoint_level=$L training.hp.training_duration=$NR training.io.print_progress_freq=$(( T * 2 )) \
    training.io.validation_freq=$NR training.io.save_checkpoint_freq=$NR $COMMON; rc=$?
  if grep -q "Training Completed" $LOG; then FIT="$B $L"; say "REG part1 PASS bpg $B level $L total batch $T: $(last_tick $LOG)"; summ "REG crops (s per crop)" $LOG $NODES; break; fi
  if is_oom $LOG; then say "REG bpg $B level $L OOM"; continue; fi
  say "REG part1 FAIL bpg $B level $L rc=$rc (not OOM): $(grep -E 'Error|error' $LOG | tail -3 | tr '\n' ' ' | cut -c1-600)"; break
done
REG_CKPT=""
if [[ -n "$FIT" ]]; then
  (( NODES == 1 )) && echo "$FIT" > $CD/state/FIT_regression && say "FIT_regression = $FIT (bpg level)"
  grep -q "EMA initialised\|EMA loaded" $S/reg_p1_b${B}_L$L.log && say "REG EMA: $(grep -h -oE 'EMA (initialised|loaded)[^|]*' $S/reg_p1_b${B}_L$L.log | head -1 | cut -c1-200)"
  ls $R/checkpoints_regression_ema/*.mdlus >/dev/null 2>&1 && say "REG SAVE_OK raw $(ls $R/checkpoints_regression/*.mdlus | sort -V | tail -1) ema $(ls $R/checkpoints_regression_ema/*.mdlus | sort -V | tail -1)"
  ls $R/crop_visits_rank0.npz >/dev/null 2>&1 && say "REG crop visits $R/crop_visits_rank0.npz: $(grep -h 'crop visits written' $S/reg_p1_b${B}_L$L.log | tail -1 | sed 's/.*crop visits written: //')"
  LOG=$S/reg_p2_b${B}_L$L.log; : > $LOG
  CD_CAP="timeout --signal=TERM --kill-after=60 300" cd_srun regression $R $LOG training.hp.batch_size_per_gpu=$B training.hp.total_batch_size=$(( B * GPUS )) \
    training.perf.songunet_checkpoint_level=$L training.hp.training_duration=$(( NR * 2 )) training.io.print_progress_freq=$(( B * GPUS * 2 )) \
    training.io.validation_freq=$(( NR * 2 )) training.io.save_checkpoint_freq=$(( NR * 2 )) $COMMON; rc=$?
  if grep -q "Training Completed" $LOG && grep -q "Loaded model state dictionary" $LOG && grep -q "Loaded optimizer state dictionary" $LOG && grep -q "EMA loaded from" $LOG; then
    say "REG_RESUME_OK: model, Adam and EMA reloaded ($(grep -h -oE 'EMA loaded from [^ ]+' $LOG | head -1)), trained to $(( NR * 2 )) samples"
  else say "REG_RESUME_FAIL rc=$rc: $(grep -cE 'Loaded model state dictionary' $LOG) model loads, $(grep -c 'Loaded optimizer' $LOG) optimizer, $(grep -c 'EMA loaded' $LOG) EMA; $(grep -E 'Error|error' $LOG | tail -2 | tr '\n' ' ' | cut -c1-400)"; fi
  REG_CKPT=$(ls $R/checkpoints_regression/*.mdlus | sort -V | tail -1)
fi

# 3. diffusion with the smoke's regression checkpoint
if [[ -z "$REG_CKPT" ]]; then
  [[ -n "${CD_REG_INIT:-}" ]] && REG_CKPT=$CD_REG_INIT && say "diffusion uses CD_REG_INIT (the regression smoke gave no checkpoint)"
fi
if [[ -n "$REG_CKPT" ]]; then
  export CD_REG_CKPT=$REG_CKPT; say "diffusion with CD_REG_CKPT=$CD_REG_CKPT"
  DB=$(python -c "import yaml;print(yaml.safe_load(open('$CD_CONF_DIR/config_training_e2v4${CD_CONF_SUFFIX}_diffusion.yaml'))['training']['hp']['batch_size_per_gpu'])")
  DT=$(( DB * GPUS )); DFIT=""
  for L in 1 2 3; do
    D=$S/diff_L$L; LOG=$S/diff_p1_L$L.log; : > $LOG
    CD_CAP="timeout --signal=TERM --kill-after=60 420" cd_srun diffusion $D $LOG training.hp.total_batch_size=$DT training.perf.songunet_checkpoint_level=$L \
      training.hp.training_duration=$ND training.io.print_progress_freq=$(( DT * 2 )) training.io.validation_freq=$ND training.io.save_checkpoint_freq=$ND $COMMON; rc=$?
    if grep -q "Training Completed" $LOG; then DFIT=$L; say "DIFF part1 PASS bpg $DB level $L total batch $DT: $(last_tick $LOG)"; summ "DIFF (s per sample, 4 patches)" $LOG $NODES; break; fi
    if is_oom $LOG; then say "DIFF level $L OOM"; continue; fi
    say "DIFF part1 FAIL level $L rc=$rc (not OOM): $(grep -E 'Error|error' $LOG | tail -3 | tr '\n' ' ' | cut -c1-600)"; break
  done
  if [[ -n "$DFIT" ]]; then
    (( NODES == 1 )) && echo "$DB $DFIT" > $CD/state/FIT_diffusion && say "FIT_diffusion = $DB $DFIT (bpg level)"
    ls $D/checkpoints_patched_diffusion_ema/*.mdlus >/dev/null 2>&1 && say "DIFF SAVE_OK raw $(ls $D/checkpoints_patched_diffusion/*.mdlus | sort -V | tail -1) ema $(ls $D/checkpoints_patched_diffusion_ema/*.mdlus | sort -V | tail -1)"
    LOG=$S/diff_p2_L$DFIT.log; : > $LOG
    CD_CAP="timeout --signal=TERM --kill-after=60 300" cd_srun diffusion $D $LOG training.hp.total_batch_size=$DT training.perf.songunet_checkpoint_level=$DFIT \
      training.hp.training_duration=$(( ND * 2 )) training.io.print_progress_freq=$(( DT * 2 )) training.io.validation_freq=$(( ND * 2 )) training.io.save_checkpoint_freq=$(( ND * 2 )) $COMMON; rc=$?
    if grep -q "Training Completed" $LOG && grep -q "Loaded model state dictionary" $LOG && grep -q "EMA loaded from" $LOG; then
      say "DIFF_RESUME_OK: model, Adam and EMA reloaded, trained to $(( ND * 2 )) samples"
    else say "DIFF_RESUME_FAIL rc=$rc: $(grep -E 'Error|error' $LOG | tail -2 | tr '\n' ' ' | cut -c1-400)"; fi
  fi
else say "DIFF skipped: no regression checkpoint"; fi
say "SMOKE_END $(date -u) logs $S"
