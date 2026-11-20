#!/bin/bash
# CorrDiff on Jupiter: memory and speed smokes on 1 node (4 GH200), 2,560 samples each (10 optimizer steps of 256).
# Data gate first (login fill, verify). Regression: checkpoint level 0, then 1, 2, 3 at batch_size_per_gpu 1; the first
# level that trains is written to state/REG_LEVEL (the training chain starts from it). Diffusion: level 1,
# max_patch_per_gpu 4, regression checkpoint = the regression smoke's .mdlus. Summary: state/SMOKE_SUMMARY.txt.
#SBATCH --job-name=cd_smoke
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/logs/%x_%j.out
set -uo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/kit/cd_env_jupiter.sh
python -c "import physicsnemo, torch; print(physicsnemo.__version__, torch.cuda.is_available())"
source $CD_KIT/cd_data_gate_jupiter.sh; source $CD_KIT/cd_train_jupiter_body.sh
echo "smoke start $(date -u) host $(hostname) repo $(cat $CD/PHYSICSNEMO_REV)"
cd_data_gate || exit 1
S=$CD/smoke; mkdir -p $S; SUM=$CD/state/SMOKE_SUMMARY.txt; echo "smoke job $SLURM_JOB_ID $(date -u)" >> $SUM
SM="training.hp.training_duration=2560 training.io.print_progress_freq=256 training.io.validation_freq=1280 training.io.save_checkpoint_freq=2560"
LV=""
for L in 0 1 2 3; do
  LOG=$S/reg_L$L.log; : > $LOG
  cd_torchrun regression $S/reg_L$L $LOG $SM training.perf.songunet_checkpoint_level=$L training.hp.batch_size_per_gpu=1; rc=$?
  if (( rc == 0 )) && grep -q "Training Completed" $LOG; then
    echo "REG L$L PASS: $(last_tick $LOG)" | tee -a $SUM; LV=$L; break
  fi
  if is_oom $LOG; then echo "REG L$L OOM" | tee -a $SUM; continue; fi
  echo "REG L$L FAIL rc=$rc (not OOM): $(grep -E 'Error|error' $LOG | tail -2 | tr '\n' ' ')" | tee -a $SUM; exit 1
done
[[ -n "$LV" ]] || { echo "no regression level fits" | tee -a $SUM; exit 1; }
[[ -f $CD/state/REG_LEVEL ]] || echo $LV > $CD/state/REG_LEVEL
export CD_REG_CKPT=$(find $S/reg_L$LV -name "*.mdlus" | sort | tail -1)
echo "diffusion smoke with CD_REG_CKPT=$CD_REG_CKPT" | tee -a $SUM
LOG=$S/diff_L1_mp4.log; : > $LOG
cd_torchrun diffusion $S/diff_L1_mp4 $LOG $SM training.perf.songunet_checkpoint_level=1 training.hp.max_patch_per_gpu=4; rc=$?
if (( rc == 0 )) && grep -q "Training Completed" $LOG; then echo "DIFF L1 mp4 PASS: $(last_tick $LOG)" | tee -a $SUM
elif is_oom $LOG; then echo "DIFF L1 mp4 OOM" | tee -a $SUM
else echo "DIFF L1 mp4 FAIL rc=$rc" | tee -a $SUM; fi
echo "SMOKE_END $(date -u)" | tee -a $SUM
