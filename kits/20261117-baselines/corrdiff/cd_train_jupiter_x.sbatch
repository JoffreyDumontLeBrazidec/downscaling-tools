#!/bin/bash
# CorrDiff on Jupiter, second kit copy ($CD/kit_20261123, configs config_training_e2v4x_*.yaml): training chain segment.
# Copy of cd_train_jupiter.sbatch with these changes: state folder state/chain_<mode>_x, run folder run/<reg|diff>_x
# (never chain_<mode> or run/<reg|diff>); default ONE node (give --nodes on the command line), and the successor asks for
# the node count and time limit of the running segment; CD_MAX_SEG defaults to 1 (no successor unless asked);
# validation_freq is the config's unless CD_VAL_FREQ is set; the batch must divide evenly over the GPUs (else train.py
# loads more samples per update than it counts); when the run folder already holds a model file at the start (a widened
# regression from cd_widen_regression.py, or a resume), the segment stops if the log does not show it was loaded (train.py
# swallows load errors and would train from random weights).
# Kept from the original: PhysicsNeMo's own resume from the newest checkpoint of the run folder; the successor is submitted
# once this segment logs its first progress line (afterany on itself) unless state/chain_<mode>_x/DONE or STOP exists or
# CD_MAX_SEG segments were started; checkpoint level from state/chain_<mode>_x/LEVEL else state/REG_LEVEL (regression) /
# 1 (diffusion), the next level tried in the same allocation on an out-of-memory failure before the first tick (this
# overrides the config's training.perf.songunet_checkpoint_level).
#   sbatch --nodes=N --time=HH:MM:SS --export=ALL,MODE=regression[,CD_MAX_SEG=K] cd_train_jupiter_x.sbatch
#   sbatch --nodes=N --time=HH:MM:SS --export=ALL,MODE=diffusion,CD_REG_CKPT=<regression .mdlus>[,CD_MAX_SEG=K] cd_train_jupiter_x.sbatch
#   stop the chain: touch $CD/state/chain_<mode>_x/STOP
#SBATCH --job-name=cd_x
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/logs/%x_%j.out
set -uo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/kit_20261123/cd_env_jupiter_x.sh
python ~/kernel_guard/kernel_guard.py check || exit 97
python -c "import physicsnemo, torch; print(physicsnemo.__version__, torch.cuda.is_available())"
source $CD_KIT/cd_data_gate_jupiter.sh; source $CD_KIT/cd_train_jupiter_body.sh
MODE=${MODE:-regression}; [[ $MODE == regression ]] && TAG=reg || TAG=diff
ST=$CD/state/chain_${MODE}_x; RUN=$CD/run/${TAG}_x; mkdir -p $ST $RUN
MAX_SEG=${CD_MAX_SEG:-1}
NODES=$SLURM_JOB_NUM_NODES; TLIM=$(squeue -h -j $SLURM_JOB_ID -o %l)
CFG=$CD_CONF_DIR/config_training_e2v4${CD_CONF_SUFFIX}_${MODE}.yaml
echo "segment start $(date -u) job $SLURM_JOB_ID host $(hostname) MODE=$MODE nodes=$NODES time=$TLIM run=$RUN config=$CFG repo $(cat $CD/PHYSICSNEMO_REV)"
[[ -f $CFG ]] || { echo "FATAL $CFG missing (make_cd_configs.py --name-suffix $CD_CONF_SUFFIX --out-conf-dir $CD_CONF_DIR)"; exit 1; }
[[ -f $ST/DONE ]] && { echo "CHAIN_DONE: $(cat $ST/DONE)"; exit 0; }
[[ -f $ST/STOP ]] && { echo "CHAIN_STOP present"; exit 0; }
[[ $MODE == diffusion ]] && { : "${CD_REG_CKPT:?CD_REG_CKPT=<regression .mdlus>}"; [[ -f $CD_REG_CKPT ]] || { echo "FATAL $CD_REG_CKPT missing"; exit 1; }; }
# total_batch_size must equal batch_size_per_gpu x accumulation rounds x GPUs (train.py counts total_batch_size per update)
python - "$CFG" $(( NODES * 4 )) <<'EOF' || exit 1
import sys, yaml
hp = yaml.safe_load(open(sys.argv[1]))["training"]["hp"]; g = int(sys.argv[2])
b, p = int(hp["total_batch_size"]), int(hp["batch_size_per_gpu"])
ok = b % (p * g) == 0
print(f"batch check: total_batch_size {b}, batch_size_per_gpu {p}, {g} GPUs -> "
      + (f"{b // (p * g)} accumulation rounds" if ok else "FATAL: not a multiple; train.py would load more samples per update than it counts"))
sys.exit(0 if ok else 1)
EOF
SEG=$(( $(cat $ST/SEGMENTS 2>/dev/null || echo 0) + 1 )); echo $SEG > $ST/SEGMENTS
echo "$SLURM_JOB_ID seg $SEG start $(date -u) nodes $NODES time $TLIM" >> $ST/jobs.txt
cd_data_gate || exit 1
CKD=$RUN/checkpoints_$([[ $MODE == regression ]] && echo regression || echo patched_diffusion)
HAD_MODEL=$(ls $CKD/*.mdlus 2>/dev/null | wc -l)
echo "model files in $CKD at start: $HAD_MODEL"
L0=$(cat $ST/LEVEL 2>/dev/null || cat $CD/state/REG_LEVEL 2>/dev/null || echo 0)
[[ $MODE == diffusion ]] && L0=$(cat $ST/LEVEL 2>/dev/null || echo 1)
VAL=(); [[ -n "${CD_VAL_FREQ:-}" ]] && VAL=(training.io.validation_freq=$CD_VAL_FREQ)
for L in $(seq $L0 3); do
  echo $L > $ST/LEVEL
  LOG=$CD/logs/${TAG}_x_seg${SEG}_${SLURM_JOB_ID}_L$L.log; : > $LOG
  cd_torchrun $MODE $RUN $LOG training.perf.songunet_checkpoint_level=$L ${VAL[@]+"${VAL[@]}"} &
  TP=$!; SUBMITTED=0
  # the successor is submitted only once this segment logs its first progress line (no cascade of failing segments)
  while kill -0 $TP 2>/dev/null; do
    if (( SUBMITTED == 0 )) && grep -q sec_per_sample $LOG; then
      SUBMITTED=1; echo "first tick $(date -u): $(last_tick $LOG)"
      if (( HAD_MODEL > 0 )) && ! grep -q "Loaded model state dictionary" $LOG; then
        echo "FATAL: $CKD held a model file but the log shows no 'Loaded model state dictionary' (train.py swallows load" \
             "errors and trains from random weights); stopping, no successor"
        touch $ST/STOP; exit 3   # the end of the batch script ends the job and its srun step
      fi
      if (( SEG < MAX_SEG )) && [[ ! -f $ST/STOP && ! -f $ST/DONE ]]; then
        NEXT=$(sbatch --parsable --nodes=$NODES --time=$TLIM --dependency=afterany:$SLURM_JOB_ID --export=ALL,MODE=$MODE $CD_KIT/cd_train_jupiter_x.sbatch)
        echo "successor $NEXT (afterany:$SLURM_JOB_ID, nodes $NODES, time $TLIM) seg $((SEG+1)) $(date -u)" | tee -a $ST/jobs.txt
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
