# Sourced by the T1d diagnostic GPU jobs (traj_states.sbatch, verify.sbatch). The runtime is the one stage A's gate G1
# validates: campaign settings from ../t1d_env.sh (T1D_SANDBOX = the `exp new t1d-stagea` folder, T1D_CORE_SHA,
# T1D_HOST, T1D_RUNTIME, T1D_INPUT_ROOT), all passed into the job by submit_traj.sh's --export list (Atos sets
# SBATCH_EXPORT=NONE, so nothing else from the submit shell arrives), and the same activation as
# se_tc_predict_t1d.sbatch's sandbox runtime (t1d_activate_sandbox):
#   default (T1D_RUNTIME=venv): the sandbox's own uv venv for this node, .venv-x86_64 on AC (A100) or .venv-aarch64 on
#     AG (GH200, owner's option 1 of 2026-10-07), through its guarded activate.sh, + samplers/HEAD checks
#   T1D_RUNTIME=overlay (AG opt-in): certified arm venv + PYTHONPATH onto the sandbox code + import guard
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" \
  || { echo "FATAL cannot source t1d_env.sh"; exit 2; }
HOST="${T1D_HOST:-ac}"; ARCH=$(uname -m)
case "$HOST:$ARCH" in ac:x86_64|ag:aarch64) ;; *) echo "FATAL T1D_HOST=$HOST on a $ARCH node (ac = x86_64 A100, ag = aarch64 GH200)"; exit 3 ;; esac
module load ecmwf-toolbox 2>/dev/null || true
t1d_activate_sandbox || exit $?
python -c "import anemoi.models.samplers.diffusion_samplers as d, torch; assert 'custom' in d.NOISE_SCHEDULERS, 'not the patched fork'; print('RUNTIME', d.__file__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | sed "s/^/GPU /"
export DATA_DIR=/home/mlx/ai-ml/datasets/ DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ANEMOI_INFERENCE_NUM_CHUNKS=1 ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" CODEX_UNIFIED_SHARD_STRATEGY=edges
export TORCH_COMPILE_DISABLE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HYDRA_FULL_ERROR=1
export TMPDIR=/tmp; mkdir -p /tmp/$USER/hydra
# diagnostic constants (shared by every T1d diagnostic job)
T1D_DIAG_CKPT=/home/ecm5702/perm/checkpoints/o320_o1280/551dfd1eb2a649d49f099acd30b59483/inference-anemoi-by_step-epoch_171-step_400000.ckpt
T1D_DIAG_SCOPE='{"mode":"bbox","cut_graph":true,"hidden_halo_hops":1,"label":"franklin_idalia_full250_box","lat_min":10.0,"lat_max":40.0,"lon_min":-100.0,"lon_max":-58.0}'
T1D_DIAG_SPJ='{"sampler":"heun","S_churn":0.0,"S_noise":1.0}'
T1D_DIAG_BUNDLES="${T1D_BUNDLES:-$T1D_INPUT_ROOT}"
[[ -f "$T1D_DIAG_CKPT" ]] || { echo "FATAL missing checkpoint $T1D_DIAG_CKPT"; exit 1; }
# Idalia search window of a bundle (one table: dp/common.py IDALIA_WINDOWS); override with T1D_WINDOW. A job receives
# it as T1D_WINDOW_COLON (lat0:lat1:lon0:lon1; sbatch --export splits on commas), converted back here.
[[ -n "${T1D_WINDOW_COLON:-}" ]] && export T1D_WINDOW="${T1D_WINDOW_COLON//:/,}"
t1d_window() { (cd "$T1D_CODE" && python -m scripts.t1d_sampler_20261007.dp.locate_storms --print-window "$1" "$2"); }
