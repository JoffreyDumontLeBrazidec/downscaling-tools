#!/bin/bash --login
#SBATCH --job-name=sh2a_predict_proof
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
# The one case that proves the stage 2a bundle route: initialisation
# 12 May 2026 12 UTC, lead 6 hours, members 1 and 2, through the frozen branch U
# checkpoint at step 100,000 in the sandbox that ran it on 2026-09-05.
set -uo pipefail
CKPT=/home/ecm5702/perm/checkpoints-jupiter-20260905/ffccc33c0a13450c8ad0a69ed3432945/anemoi-by_epoch-epoch_050-step_100000.ckpt
ACT=/home/ecm5702/hpcperm/sandbox/20260902-hres-local-branch/activate.sh
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
source "$ACT" || exit 3
PY=$(command -v python)
cd /home/ecm5702/dev/downscaling-tools
echo "host=$(hostname) start=$(date -Is) dstools=$(git rev-parse --short HEAD)"
t0=$(date +%s)
env -u PYTHONPATH srun --ntasks 4 --ntasks-per-node 4 "$PY" -m eval.predict.main \
  --name-ckpt "$CKPT" \
  --out-dir /home/ecm5702/scratch/eval/station_head_2a/predictions/2026051212/predictions \
  --members 1,2 --steps 6 --dates 20260512 \
  --input-root /home/ecm5702/scratch/eval/station_head_2a/bundles/2026051212 \
  --allow-existing-out-dir --allow-missing-target-unsafe \
  --num-gpus-per-model 4 --extra-args-json "$SAMPLER"
rc=$?
echo "PROOF predict rc=$rc wall_s=$(( $(date +%s) - t0 ))"
exit $rc
