#!/bin/bash
# o2560 Europe-box read (2 Nov 2026): EMA-10000 predictions of one row at one lead, 12 dates (1-12 Nov 2024) x 10 members,
# rule-31 seeds, 30-step churn-off sampler, fp32 + TF32, one node of 4 GH200 with the model sharded over 4 ranks (the full
# o2560 box does not fit one GH200 in inference: smoke 2645083 ran out of memory at 88 GiB; the noise is one box-wide field
# per draw whatever the number of ranks, GlobalNoise in the predictor). Usage: sbatch --job-name=o25p_<row>_L<lead> predict.sbatch <row> <lead>
# Skips files already complete; exit 7 if the checkpoint file is missing. Log ends O2560_PREDICT_DONE.
#SBATCH --job-name=o25p
#SBATCH --account=e-sta-destine
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs/%x-%j.out
set -eo pipefail
ulimit -c 0
KITK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/kit
row=${1:?row}; lead=${2:?lead}; L=$(printf %03d $((10#$lead)))
source $KITK/rows.sh "$row" || exit 7
source $KITK/env.sh
export MODE=predict OMP_NUM_THREADS=8 OUTDIR=$OUT/pred/${row}_L$L CASES=$(cases_for $lead)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1); export MASTER_ADDR MASTER_PORT=29594
echo "[o25p] row=$row lead=$lead step=$CKPT_STEP ckpt=$CKPT_FILE code=$CODE_ROOT@$(git -C $CODE_ROOT rev-parse --short HEAD) outdir=$OUTDIR ranks=$SLURM_NTASKS chunks=$ANEMOI_INFERENCE_NUM_CHUNKS $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 30 > $KIT/logs/gpumem_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv 2>/dev/null &
cd /tmp
rc=0
srun --kill-on-bad-exit=1 --label bash -c 'export CUDA_VISIBLE_DEVICES=0,1,2,3; exec python '"$KITK"'/o2560_box_predict.py' || rc=$?
n=$(ls $OUTDIR/predictions_*_step$L.nc 2>/dev/null | wc -l || true)
echo "[o25p] rc=$rc $row L$L holds $n of 12 files; gpu peak (nvidia-smi MiB) $(sort -t, -k3 -n $KIT/logs/gpumem_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv | tail -1) $(date -u +%FT%TZ)"
[[ $rc == 0 && $n == 12 ]] || { echo "[o25p] FAIL $row L$L"; exit 1; }
