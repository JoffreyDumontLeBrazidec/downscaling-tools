# Sourced by the T1d diagnostic GPU jobs (traj_states.sbatch, verify.sbatch). The runtime is the one stage A's gate G1
# validates: campaign settings from ../t1d_env.sh (T1D_SANDBOX = the `exp new t1d-stagea` folder, T1D_CORE_SHA,
# T1D_HOST, T1D_CERT_VENV_AG, T1D_INPUT_ROOT), and the same activation as se_tc_predict_t1d.sbatch's sandbox runtime:
#   T1D_HOST=ac (default): Atos AC, A100, x86_64 -> the sandbox's own uv venv via its guarded activate.sh
#   T1D_HOST=ag          : Atos AG, GH200, aarch64 -> certified arm venv + PYTHONPATH overlay of the sandbox code
#                          + import guard (the runbook's AG rule; the uv layer is not validated on aarch64)
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" \
  || { echo "FATAL cannot source t1d_env.sh"; exit 2; }
HOST="${T1D_HOST:-ac}"; ARCH=$(uname -m)
case "$HOST:$ARCH" in ac:x86_64|ag:aarch64) ;; *) echo "FATAL T1D_HOST=$HOST on a $ARCH node (ac = x86_64 A100, ag = aarch64 GH200)"; exit 3 ;; esac
module load ecmwf-toolbox 2>/dev/null || true
case "$HOST" in
  ac)
    set +u; source "$T1D_SANDBOX/activate.sh" || { echo "FATAL sandbox activation failed ($T1D_SANDBOX)"; exit 3; }; set -u
    CORE_DIR=$T1D_SANDBOX/code/anemoi-core ;;
  ag)
    set +u; unset PYTHONPATH; source "$T1D_CERT_VENV_AG/bin/activate" || exit 3; set -u
    SBC=$T1D_SANDBOX/code
    export PYTHONPATH="$SBC/anemoi-core/training/src:$SBC/anemoi-core/models/src:$SBC/anemoi-core/graphs/src:$SBC/anemoi-inference/src"
    python - "$SBC" <<'PYGUARD' || { echo "FATAL GUARD: the overlay did not win"; exit 4; }
import importlib, os, sys
root = os.path.realpath(sys.argv[1]); bad = []
for name in ("anemoi.models", "anemoi.training", "anemoi.graphs", "anemoi.inference", "anemoi.models.samplers.diffusion_samplers"):
    p = os.path.realpath(importlib.import_module(name).__file__)
    (bad.append if not p.startswith(root + os.sep) else print)(f"  GUARD {name} -> {p}")
if bad: print("\n".join(bad)); sys.exit(1)
PYGUARD
    CORE_DIR=$SBC/anemoi-core ;;
esac
[[ "$(git -C "$CORE_DIR" rev-parse HEAD)" == "$T1D_CORE_SHA" ]] || { echo "FATAL sandbox anemoi-core is not $T1D_CORE_SHA"; exit 3; }
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
# Idalia search window of a bundle (one table: dp/common.py IDALIA_WINDOWS); override with T1D_WINDOW
t1d_window() { (cd "$T1D_CODE" && python -m scripts.t1d_sampler_20261007.dp.locate_storms --print-window "$1" "$2"); }
