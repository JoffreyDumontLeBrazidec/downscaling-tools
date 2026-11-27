#!/bin/bash
# CorrDiff Europe (E2 box) on MN5: one training chain segment (regression on random 448 crops, or patched diffusion),
# configs config_training_e2v4m5_<mode>.yaml in $CD/conf. Modelled on cd_train_jupiter_x.sbatch, with:
# - state $CD/state/chain_<mode>_m5, run $CD/run/<reg|diff>_m5; nodes and time from the command line (default 1 node,
#   12 h); the successor asks for the same node count; CD_MAX_SEG (default 1) segments at most.
# - GPU-hour ceiling CD_GPUH_CEIL (default: regression 150, diffusion 700 H100-h), summed over every job id of the
#   chain in state/chain_<mode>_m5/jobs.txt (sacct ElapsedRaw x GPUs, cd_gpuh.py; failed segments count). Before
#   training: if the chain has reached the ceiling, write CEILING_REACHED and stop; if this segment's time limit
#   would pass it, training is cut by `timeout` at the remaining budget. Before submitting a successor: the remaining
#   budget after this segment's full time limit sets the successor's time limit (at most this one's); under 1 h
#   left -> no successor, CEILING_REACHED written.
# - regression start: CD_REG_INIT = a widened regression .mdlus (cd_widen_regression.py widen output) copied as
#   CorrDiffRegressionUNet.0.0.mdlus into an EMPTY run folder (training starts at 0 samples, fresh Adam, EMA from these
#   weights); unset = from scratch (fallback). A run folder that already holds a model file resumes and ignores it.
# - batch: no gradient accumulation (total_batch_size = batch_size_per_gpu x GPUs). Regression: batch_size_per_gpu and
#   checkpoint level from $CD/state/FIT_regression ("BPG LEVEL", written by the smoke) else the config's, total batch
#   overridden to BPG x GPUs (constant learning rate, no sample-based schedule); an out-of-memory failure (at any point, resuming
#   from the newest checkpoint) steps down the ladder 8:0 4:0 4:1 2:1 2:2 1:2 1:3 (bpg:level) in the same allocation. Diffusion:
#   the config's batch (its schedule is scaled to it) must equal batch_size_per_gpu x GPUs, else the segment stops;
#   out of memory steps the checkpoint level up (1, 2, 3).
# - load guards: a run folder holding a model file must show "Loaded model state dictionary" (train.py swallows load
#   errors); EMA: train_cd.py stops on a resume without the EMA file (EMA_RESUME_MISSING).
#   sbatch --nodes=N --time=HH:MM:SS --export=ALL,MODE=regression[,CD_REG_INIT=<widened .mdlus>][,CD_MAX_SEG=K][,CD_GPUH_CEIL=X] cd_train_mn5.sbatch
#   sbatch --nodes=N --time=HH:MM:SS --export=ALL,MODE=diffusion,CD_REG_CKPT=<regression .mdlus>[,CD_MAX_SEG=K][,CD_GPUH_CEIL=X] cd_train_mn5.sbatch
#   stop the chain: touch $CD/state/chain_<mode>_m5/STOP
#SBATCH --job-name=cd_m5
#SBATCH --account=ehpc01
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5/logs/%x_%j.out
set -uo pipefail
ulimit -c 0
source /gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5/kit/cd_env_mn5.sh
source $CD_KIT/cd_train_mn5_body.sh
MODE=${MODE:-regression}; [[ $MODE == regression ]] && TAG=reg || TAG=diff
ST=$CD/state/chain_${MODE}_m5; RUN=$CD/run/${TAG}_m5; mkdir -p $ST $RUN
MAX_SEG=${CD_MAX_SEG:-1}
CEIL=${CD_GPUH_CEIL:-$([[ $MODE == regression ]] && echo 150 || echo 700)}
NODES=$SLURM_JOB_NUM_NODES; GPUS=$(( NODES * 4 ))
TLIM=$(squeue -h -j $SLURM_JOB_ID -o %l); TLIM_S=$(tlim_s $TLIM)
CFG=$CD_CONF_DIR/config_training_e2v4${CD_CONF_SUFFIX}_${MODE}.yaml
echo "segment start $(date -u) job $SLURM_JOB_ID host $(hostname) MODE=$MODE nodes=$NODES time=$TLIM run=$RUN config=$CFG ceiling=$CEIL H100-h"
python -c "import physicsnemo, torch; print('physicsnemo', physicsnemo.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
cd_code_ok || exit 1
[[ -f $CFG ]] || { echo "FATAL $CFG missing (make_cd_configs.py --name-suffix $CD_CONF_SUFFIX --out-conf-dir $CD_CONF_DIR)"; exit 1; }
[[ -f $ST/DONE ]] && { echo "CHAIN_DONE: $(cat $ST/DONE)"; exit 0; }
[[ -f $ST/STOP ]] && { echo "CHAIN_STOP present"; exit 0; }
[[ -f $ST/CEILING_REACHED ]] && { echo "CEILING_REACHED: $(cat $ST/CEILING_REACHED)"; exit 0; }
cd_data_ok || exit 1
[[ $MODE == diffusion ]] && { : "${CD_REG_CKPT:?CD_REG_CKPT=<regression .mdlus>}"; [[ -f $CD_REG_CKPT ]] || { echo "FATAL $CD_REG_CKPT missing"; exit 1; }; }

# GPU-hour ceiling, before training
USED=$(python $CD_KIT/cd_gpuh.py used --jobs $ST/jobs.txt --exclude $SLURM_JOB_ID --verbose) || { echo "FATAL: sacct accounting failed"; exit 1; }
REM_S=$(python -c "print(int(max(0.0, ($CEIL - $USED) * 3600 / $GPUS)))")
echo "ceiling: chain used $USED H100-h before this segment, ceiling $CEIL, budget left for this segment ${REM_S} s on $GPUS GPUs (time limit $TLIM_S s)"
if (( REM_S < 600 )); then echo "ceiling reached: $USED of $CEIL H100-h used, job $SLURM_JOB_ID $(date -u)" | tee $ST/CEILING_REACHED; exit 0; fi
(( REM_S < TLIM_S )) && echo "this segment is cut at $(( REM_S - 60 )) s by the ceiling"

SEG=$(( $(cat $ST/SEGMENTS 2>/dev/null || echo 0) + 1 )); echo $SEG > $ST/SEGMENTS
echo "$SLURM_JOB_ID seg $SEG start $(date -u) nodes $NODES time $TLIM" >> $ST/jobs.txt
CKD=$RUN/checkpoints_$([[ $MODE == regression ]] && echo regression || echo patched_diffusion)
mkdir -p $CKD
if [[ $MODE == regression && -n "${CD_REG_INIT:-}" ]]; then
  if ls $CKD/*.mdlus >/dev/null 2>&1; then echo "CD_REG_INIT ignored: $CKD already holds model files (resume)"
  else [[ -f $CD_REG_INIT ]] || { echo "FATAL CD_REG_INIT $CD_REG_INIT missing"; exit 1; }
       cp $CD_REG_INIT $CKD/CorrDiffRegressionUNet.0.0.mdlus && echo "start weights: $CD_REG_INIT -> $CKD/CorrDiffRegressionUNet.0.0.mdlus" | tee -a $ST/jobs.txt; fi
elif [[ $MODE == regression ]] && ! ls $CKD/*.mdlus >/dev/null 2>&1; then echo "regression from scratch (CD_REG_INIT unset, empty run folder)"; fi
HAD_MODEL=$(ls $CKD/*.mdlus 2>/dev/null | wc -l)
echo "model files in $CKD at start: $HAD_MODEL"

# batch / memory ladder
if [[ $MODE == regression ]]; then
  LADDER=(8:0 4:0 4:1 2:1 2:2 1:2 1:3)
  read -r B0 L0 < <(cat $ST/FIT 2>/dev/null || cat $CD/state/FIT_regression 2>/dev/null || python - "$CFG" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))["training"]
print(c["hp"]["batch_size_per_gpu"], c["perf"]["songunet_checkpoint_level"])
EOF
)
  START=$(for i in "${!LADDER[@]}"; do [[ ${LADDER[$i]} == "$B0:$L0" ]] && { echo $i; break; }; done)
  [[ -n "$START" ]] || { LADDER=("$B0:$L0" "${LADDER[@]}"); START=0; }
else
  python - "$CFG" $GPUS <<'EOF' || exit 1
import sys, yaml
hp = yaml.safe_load(open(sys.argv[1]))["training"]["hp"]; g = int(sys.argv[2])
b, p = int(hp["total_batch_size"]), int(hp["batch_size_per_gpu"])
ok = b == p * g
print(f"batch check: total_batch_size {b}, batch_size_per_gpu {p}, {g} GPUs -> " + ("no accumulation, OK" if ok else
      "FATAL: total_batch_size must equal batch_size_per_gpu x GPUs (no accumulation); regenerate the configs for this node count"))
sys.exit(0 if ok else 1)
EOF
  B=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['training']['hp']['batch_size_per_gpu'])")
  L1=$(cat $ST/FIT 2>/dev/null | awk '{print $2}'); L1=${L1:-$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['training']['perf']['songunet_checkpoint_level'])")}
  LADDER=(); for l in $(seq $L1 3); do LADDER+=("$B:$l"); done; START=0
fi
VAL=(); [[ -n "${CD_VAL_FREQ:-}" ]] && VAL=(training.io.validation_freq=$CD_VAL_FREQ)

SUBMITTED=0
for (( i=START; i<${#LADDER[@]}; i++ )); do
  HAD_MODEL=$(ls $CKD/*.mdlus 2>/dev/null | wc -l)   # a later rung resumes from what an earlier one saved
  BPG=${LADDER[$i]%:*}; L=${LADDER[$i]#*:}; echo "$BPG $L" > $ST/FIT
  OV=(training.perf.songunet_checkpoint_level=$L)
  [[ $MODE == regression ]] && OV+=(training.hp.batch_size_per_gpu=$BPG training.hp.total_batch_size=$(( BPG * GPUS )))
  LOG=$CD/logs/${TAG}_m5_seg${SEG}_${SLURM_JOB_ID}_b${BPG}_L$L.log; : > $LOG
  echo "attempt batch_size_per_gpu $BPG level $L (total batch $(( BPG * GPUS ))) log $LOG"
  CAP=""; (( REM_S < TLIM_S )) && CAP="timeout --signal=TERM --kill-after=120 $(( REM_S - SECONDS - 60 ))"   # left of the cut
  CD_CAP="$CAP" cd_srun $MODE $RUN $LOG "${OV[@]}" ${VAL[@]+"${VAL[@]}"} &
  TP=$!; TICKED=0
  # the successor is submitted only once this segment logs its first progress line (no cascade of failing segments)
  while kill -0 $TP 2>/dev/null; do
    if (( TICKED == 0 )) && grep -q sec_per_sample $LOG; then
      TICKED=1; echo "first tick $(date -u): $(last_tick $LOG)"
      if (( HAD_MODEL > 0 )) && ! grep -q "Loaded model state dictionary" $LOG; then
        echo "FATAL: $CKD held a model file but the log shows no 'Loaded model state dictionary' (train.py swallows load" \
             "errors and trains from random weights); stopping, no successor"
        touch $ST/STOP; exit 3   # the end of the batch script ends the job and its srun step
      fi
      if (( SUBMITTED == 0 && SEG < MAX_SEG )) && [[ ! -f $ST/STOP && ! -f $ST/DONE ]]; then
        SUBMITTED=1
        OTHERS=$(python $CD_KIT/cd_gpuh.py used --jobs $ST/jobs.txt --exclude $SLURM_JOB_ID) || OTHERS=""
        [[ $OTHERS =~ ^[0-9.]+$ ]] || { echo "sacct accounting failed: no successor submitted; STOP written"; touch $ST/STOP; OTHERS=1e9; }
        NEXT_S=$(python -c "r=($CEIL-$OTHERS)*3600/$GPUS-$TLIM_S; print(int(min($TLIM_S, max(0, r))))")
        if (( NEXT_S < 3600 )); then
          echo "no successor: after this segment's full time limit the chain would have used $OTHERS + $(( TLIM_S * GPUS / 3600 )) of $CEIL H100-h (under 1 h left) $(date -u)" | tee $ST/CEILING_REACHED
        else
          NT=$(printf '%d:%02d:%02d' $(( NEXT_S / 3600 )) $(( NEXT_S % 3600 / 60 )) $(( NEXT_S % 60 )))
          UNSET=(); for v in $(compgen -e); do [[ $v == SLURM_* && $v != SLURM_CONF ]] && UNSET+=(-u $v); done
          NEXT=$(env "${UNSET[@]}" sbatch --parsable --nodes=$NODES --time=$NT --dependency=afterany:$SLURM_JOB_ID \
                 --export=ALL,MODE=$MODE,CD_MAX_SEG=$MAX_SEG,CD_GPUH_CEIL=$CEIL${CD_REG_CKPT:+,CD_REG_CKPT=$CD_REG_CKPT}${CD_VAL_FREQ:+,CD_VAL_FREQ=$CD_VAL_FREQ} \
                 $CD_KIT/cd_train_mn5.sbatch)
          echo "successor $NEXT (afterany:$SLURM_JOB_ID, nodes $NODES, time $NT) seg $((SEG+1)) $(date -u)" | tee -a $ST/jobs.txt
        fi
      fi
    fi
    sleep 30
  done
  wait $TP; rc=$?
  if grep -q "Training Completed" $LOG; then echo "Training Completed job $SLURM_JOB_ID $(date -u) $(last_tick $LOG)" > $ST/DONE; cat $ST/DONE; exit 0; fi
  grep -q EMA_RESUME_MISSING $LOG && { echo "FATAL: EMA_RESUME_MISSING (see $LOG); STOP written"; touch $ST/STOP; exit 4; }
  # out of memory (before or after the first tick, e.g. at the first full-box validation): next rung in this
  # allocation, resumed from the newest checkpoint; $ST/FIT then holds the smaller rung for the successors
  if is_oom $LOG; then echo "bpg $BPG level $L out of memory ($(grep -c sec_per_sample $LOG) ticks), next rung"; continue; fi
  (( rc == 124 )) && echo "segment cut by the GPU-hour ceiling $(date -u)" | tee $ST/CEILING_REACHED
  echo "segment end rc=$rc $(date -u) last tick: $(last_tick $LOG) | $(tick_stats $LOG)"; exit $rc
done
echo "FATAL: no batch / checkpoint level fits"; exit 1
