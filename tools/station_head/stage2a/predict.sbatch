#!/bin/bash --login
#SBATCH --job-name=sh2a_predict
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%A_%a.out
# Stage 2a inference: the frozen branch U checkpoint at step 100,000 over the 2026
# AIFS ENS version 2 calendar, one array task per group of initialisations.
#
# This is se_predict.sbatch of the fine-scale short evaluation with one structural
# change. That script called eval.cli predict, which builds an srun command and
# runs eval.predict.main. Our bundles carry no truth field, because the station
# head is scored against station reports and not against the analysis grid, and
# eval.cli has no way to pass the harness's own prediction-only escape hatch,
# --allow-missing-target-unsafe, down to the predictor. So this script builds the
# same srun command itself, with the same sampler configuration that the lane
# _hres_shorteval_ac carries and that job 33412497 used on 2026-09-05 for this
# very checkpoint, and adds that one flag. Nothing in the harness is modified.
#
# Usage:
#   sbatch --array=0-27 --export=ALL,SPLIT=validation,NSLICES=28 predict.sbatch
set -uo pipefail
SPLIT=${SPLIT:?validation or training}
NSLICES=${NSLICES:?number of array tasks}
SLICE=${SLURM_ARRAY_TASK_ID:-0}
CKPT=${CKPT:-/home/ecm5702/perm/checkpoints-jupiter-20260905/ffccc33c0a13450c8ad0a69ed3432945/anemoi-by_epoch-epoch_050-step_100000.ckpt}
ACT=${ACT:-/home/ecm5702/hpcperm/sandbox/20260902-hres-local-branch/activate.sh}
CASES=/home/ecm5702/agent-work/20260909-station-head-adapter/notes/cases_firstcut.csv
BUNDLE_ROOT=/home/ecm5702/scratch/eval/station_head_2a/bundles
OUT_ROOT=/home/ecm5702/scratch/eval/station_head_2a/predictions
# the sampler block of lane _hres_shorteval_ac, copied verbatim from the effective
# config that eval.cli wrote for this checkpoint on 2026-09-05
SAMPLER='{"schedule_type": "experimental_piecewise", "num_steps": 30, "sigma_max": 100000.0, "sigma_transition": 10.0, "sigma_min": 0.03, "high_schedule_type": "exponential", "low_schedule_type": "karras", "num_steps_high": 10, "num_steps_low": 20, "rho": 7.0, "sampler": "heun", "S_churn": 2.5, "S_min": 0.75, "S_max": 100000.0, "S_noise": 1.05}'

module load ecmwf-toolbox
export DATA_DIR=/home/mlx/ai-ml/datasets/ DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# the predict.env block of lane _hres_shorteval_ac, which eval.cli would normally
# push into the environment: chunking off (the edges sharding path pays a speed
# tax for chunking and needs none) and the expandable allocator, without which
# rank 0 runs out of memory on a 40 GB A100 during the first halo exchange.
export ANEMOI_INFERENCE_NUM_CHUNKS=1 ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
source "$ACT" || { echo "venv activation FAILED: $ACT"; exit 3; }
PY=$(command -v python)

INIT_LIST=$("$PY" - "$CASES" "$SPLIT" "$SLICE" "$NSLICES" <<'PYEOF'
import csv, sys
cases, split, sl, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = list(csv.DictReader(open(cases)))
inits = sorted({(r["date"] + ("%02d" % int(r["time"])), r["n_members"])
                for r in rows if r["split"] == split})
print(" ".join(a + ":" + b for a, b in inits[sl::n]))
PYEOF
)
cd /home/ecm5702/dev/downscaling-tools
echo "host=$(hostname) start=$(date -Is) dstools=$(git rev-parse --short HEAD) split=$SPLIT slice=$SLICE/$NSLICES"
echo "checkpoint=$CKPT"
echo "initialisations: $INIT_LIST"

rc_all=0
for tok in $INIT_LIST; do
  INIT=${tok%%:*}
  NMEM=${tok##*:}
  DATE=${INIT:0:8}
  MEMBERS=$(seq -s, 1 "$NMEM")
  BDIR=$BUNDLE_ROOT/$INIT
  ODIR=$OUT_ROOT/$INIT/predictions
  mkdir -p "$ODIR"
  for STEP in 6 12; do
    TARGET="$ODIR/predictions_${DATE}_step$(printf %03d "$STEP").nc"
    if [ -s "$TARGET" ]; then
      echo "=== $INIT step $STEP already done, skipping"
      continue
    fi
    echo "=== $INIT step $STEP members=$MEMBERS bundles=$BDIR"
    t0=$(date +%s)
    env -u PYTHONPATH srun --ntasks 4 --ntasks-per-node 4 "$PY" -m eval.predict.main \
      --name-ckpt "$CKPT" \
      --out-dir "$ODIR" \
      --members "$MEMBERS" \
      --steps "$STEP" \
      --dates "$DATE" \
      --input-root "$BDIR" \
      --allow-existing-out-dir \
      --allow-missing-target-unsafe \
      --num-gpus-per-model 4 \
      --extra-args-json "$SAMPLER"
    rc=$?
    echo "PREDICT init=$INIT step=$STEP rc=$rc wall_s=$(( $(date +%s) - t0 ))"
    [ "$rc" -ne 0 ] && rc_all=$rc
  done
done
echo "SH2A predict split=$SPLIT slice=$SLICE rc=$rc_all"
exit $rc_all
