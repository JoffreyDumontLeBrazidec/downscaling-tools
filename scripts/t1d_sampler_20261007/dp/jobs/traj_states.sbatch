#!/bin/bash --login
#SBATCH --job-name=t1d_traj
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# T1d dense-trajectory diagnostic (docs: 20261007_T1d_sampler_beat_1p2M). One bundle (date, lead, member 01),
# several seeds, the dense churn-off Heun schedule, the 1.2M parent 551dfd1e step 400k.
#   ROUTE=box (default): regional Franklin-Idalia cut graph (10-40N, 100-58W, halo 1), ONE A100, as the
#                        tc_o320_o1280 lane's local_scope; about 1.1 s per denoiser call.
#   ROUTE=global       : the global graph sharded over 4 A100 (submit with --ntasks-per-node=4
#                        --gpus-per-node=4 --mem=0); about 3 s per call. Fallback only.
# Usage (see LAUNCH.md; submit_traj.sh builds the calls):
#   T1D_SB=<sandbox> sbatch --output=<log> traj_states.sbatch DATE STEP NLEV OUTDIR SEED [SEED ...]
set -uo pipefail
DATE="${1:?date YYYYMMDD}"; STEP="${2:?lead, zero-padded e.g. 024}"; NLEV="${3:?dense levels: 12 (smoke) or 240}"
OUTDIR="${4:?output dir}"; shift 4; SEEDS=("$@"); [[ ${#SEEDS[@]} -gt 0 ]] || { echo "FATAL no seeds"; exit 2; }
ROUTE="${ROUTE:-box}"
SB="${T1D_SB:?export T1D_SB=<the experiment sandbox folder>}"
DT=$SB/downscaling-tools
CKPT=/home/ecm5702/perm/checkpoints/o320_o1280/551dfd1eb2a649d49f099acd30b59483/inference-anemoi-by_step-epoch_171-step_400000.ckpt
# default = the stage-A input root (T1d t1d_env.sh T1D_INPUT_ROOT), the bundles the box screen reads
BUNDLES="${T1D_BUNDLES:-/ec/res4/scratch/ecm5702/eval/o320_o1280/manual_731d203a_pristine_20260818/bundles_with_y}"
SCOPE='{"mode":"bbox","cut_graph":true,"hidden_halo_hops":1,"label":"franklin_idalia_full250_box","lat_min":10.0,"lat_max":40.0,"lon_min":-100.0,"lon_max":-58.0}'
SCHED=$DT/scripts/t1d_sampler_20261007/dp/schedules/dense${NLEV}.json
SPJ='{"sampler":"heun","S_churn":0.0,"S_noise":1.0}'
# storm search window: 5-6 deg inside the cut region, so the 500 km disc never touches its edge
WINDOW="15,35,266,296"
[[ -f "$CKPT" ]] || { echo "FATAL missing checkpoint $CKPT"; exit 1; }
[[ -f "$SCHED" ]] || { echo "FATAL missing schedule $SCHED"; exit 1; }
NSJ=$(cat "$SCHED")
module load ecmwf-toolbox 2>/dev/null || true
source "$SB/activate.sh" || { echo "sandbox activation FAILED"; exit 3; }
python -c "import anemoi.models.samplers.diffusion_samplers as d, torch; assert 'custom' in d.NOISE_SCHEDULERS, 'not the patched fork'; print('RUNTIME', d.__file__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | sed "s/^/GPU /"
export DATA_DIR=/home/mlx/ai-ml/datasets/ DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ANEMOI_INFERENCE_NUM_CHUNKS=1 ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" CODEX_UNIFIED_SHARD_STRATEGY=edges
export TORCH_COMPILE_DISABLE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ANEMOI_GRAPH_TRANSFORMER_IMPL=triton HYDRA_FULL_ERROR=1
export TMPDIR=/tmp; mkdir -p /tmp/$USER/hydra
cd "$DT" || exit 5
ARGS=(--checkpoint "$CKPT" --output-dir "$OUTDIR"
      --bundle-dir "$BUNDLES" --dates "$DATE" --members 01 --steps "$STEP"
      --mode trajectory --lockin --save-trajectory-states
      --num-steps "$NLEV" --seeds "${SEEDS[@]}"
      --eye-radius-km 500 --auto-window "$WINDOW" --precision fp32
      --noise-scheduler-json "$NSJ" --sampler-params-json "$SPJ")
echo "host=$(hostname) start=$(date -Is) route=$ROUTE dstools=$(git rev-parse --short HEAD) core=$(git -C $SB/code/anemoi-core rev-parse --short HEAD) ckpt=$CKPT date=$DATE step=$STEP member=01 seeds=${SEEDS[*]} nlev=$NLEV calls_per_seed=$((2*NLEV-1)) out=$OUTDIR bundles=$BUNDLES"
mkdir -p "$OUTDIR"
t0=$(date +%s)
if [[ "$ROUTE" == box ]]; then
  python -m interp trajectory "${ARGS[@]}" --local-scope-json "$SCOPE"
  rc=$?
elif [[ "$ROUTE" == global ]]; then
  srun --ntasks 4 --ntasks-per-node 4 python -m interp trajectory "${ARGS[@]}"
  rc=$?
else
  echo "FATAL ROUTE=$ROUTE"; exit 2
fi
wall=$(( $(date +%s) - t0 ))
echo "T1D_TRAJ route=$ROUTE date=$DATE step=$STEP seeds=${SEEDS[*]} nlev=$NLEV rc=$rc wall_s=$wall s_per_call=$(python -c "print(round($wall/(${#SEEDS[@]}*(2*$NLEV-1)),3))")"
ls -la "$OUTDIR"
exit $rc
