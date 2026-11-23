#!/bin/bash
# CorrDiff on Jupiter, one node, at most 30 min: (1) residual ladder of regression checkpoints (cd_residual_ladder.py, one
# process per GPU, restartable, time-boxed by CD_LADDER_MIN minutes), (2) widening check (cd_widen_regression.py check:
# donor vs widened output on one validation row; the widened file is made first if missing), (3) if CD_DIFF_SMOKE=1, a
# diffusion training smoke of 640 samples with the x diffusion config (time inputs, regression = the widened file) to
# read seconds per sample and peak GPU memory. Booster is OverSubscribe=EXCLUSIVE: a job always gets (and is charged)
# whole nodes, so the header asks for the whole node and uses its 4 GPUs.
#   sbatch --export=ALL,CD_LADDER_CKPTS="<paths or globs>"[,CD_DIFF_SMOKE=1] cd_ladder_jupiter.sbatch
# Optional: CD_LADDER_OUT (default $CD/eval_x/ladder), CD_LADDER_N (48), CD_LADDER_MIN (14), CD_WIDEN_DONOR (default the
# final regression checkpoint), CD_WIDENED (default $CD_KIT/tests_20261123/widened/<donor name>.w5.mdlus).
#SBATCH --job-name=cd_ladder
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/logs/%x_%j.out
set -uo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/kit_20261123/cd_env_jupiter_x.sh
python ~/kernel_guard/kernel_guard.py check || exit 97
python -c "import physicsnemo, torch; print(physicsnemo.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
source $CD_KIT/cd_train_jupiter_body.sh
: "${CD_LADDER_CKPTS:?CD_LADDER_CKPTS=<regression .mdlus paths or globs, space or comma separated>}"
OUT=${CD_LADDER_OUT:-$CD/eval_x/ladder}; mkdir -p $OUT
CFG=$CD_CONF_DIR/config_training_e2v4x_diffusion.yaml
DONOR=${CD_WIDEN_DONOR:-$CD/run/reg/checkpoints_regression/CorrDiffRegressionUNet.0.1000192.mdlus}
WIDE=${CD_WIDENED:-$CD_KIT/tests_20261123/widened/$(basename ${DONOR%.mdlus}).w5.mdlus}
cd $CD_REPO/examples/weather/corrdiff
echo "ladder job $SLURM_JOB_ID $(date -u) host $(hostname): ckpts '$CD_LADDER_CKPTS' -> $OUT; donor $DONOR widened $WIDE"

# (1) residual ladder, 4 shards (one per GPU), time-boxed; parts already written are skipped on a rerun
T0=$SECONDS
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-288} --cpu-bind=none bash -c "
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=\$r timeout ${CD_LADDER_MIN:-14}m python $CD_KIT/cd_residual_ladder.py run --cfg $CFG --out $OUT \
      --ckpts '$CD_LADDER_CKPTS' --n ${CD_LADDER_N:-48} --shard \$r/4 > $OUT/run_${SLURM_JOB_ID}_r\$r.log 2>&1 &
  done; wait"
python $CD_KIT/cd_residual_ladder.py report --out $OUT | tail -3
grep -h "\[ladder\]" $OUT/run_${SLURM_JOB_ID}_r*.log | grep -v "rows read" | head -40
grep -hE "Error|error" $OUT/run_${SLURM_JOB_ID}_r*.log | tail -5
echo "LADDER_STEP_END $(( SECONDS - T0 )) s"

# (2) widening check on one GPU
T0=$SECONDS
[[ -f $WIDE ]] || python $CD_KIT/cd_widen_regression.py widen --donor $DONOR --add 5 --out $WIDE
CUDA_VISIBLE_DEVICES=0 srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-288} --cpu-bind=none \
  python $CD_KIT/cd_widen_regression.py check --donor $DONOR --widened $WIDE --cfg $CFG 2>&1 | grep -E "CHECK_|Error"
echo "WIDEN_CHECK_STEP_END $(( SECONDS - T0 )) s"

# (3) diffusion smoke: 640 samples with the x diffusion config, regression = the widened checkpoint (51 input channels)
if [[ "${CD_DIFF_SMOKE:-0}" == 1 ]]; then
  T0=$SECONDS
  export CD_REG_CKPT=$WIDE
  S=$CD/smoke_x/diff_${SLURM_JOB_ID}; mkdir -p $S; LOG=$S/train.log; : > $LOG
  cd_torchrun diffusion $S $LOG training.hp.training_duration=640 training.io.print_progress_freq=160 \
    training.io.save_checkpoint_freq=640 training.io.validation_freq=640; rc=$?
  echo "DIFF_SMOKE rc=$rc $( grep -q 'Training Completed' $LOG && echo COMPLETED || echo NOT_COMPLETED ): $(last_tick $LOG)"
  grep -h "sec_per_sample" $LOG | sed 's/.*\] - //' | cut -c1-300
  is_oom $LOG && echo "DIFF_SMOKE out of memory"
  echo "DIFF_SMOKE_STEP_END $(( SECONDS - T0 )) s"
fi
echo "LADDER_JOB_END $(date -u)"
