#!/bin/bash --login
#SBATCH --job-name=t1d_predict
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ecm5702/work-t1d-20261007/logs/%x_%j.out
# Campaign T1d stage A (2026-10-07): one Franklin/Idalia box prediction run of the 1.2M parent (551dfd1e step 400k)
# on ONE A100 of Atos AC (qos ng = resource_profiles.predict of tc_o320_o1280), fp32, cut graph.
# Written new because the stage 3 script se_tc_predict_s3.sbatch was lost; the executor may diff it against the Mac
# copy in /Users/ecm5702/agent-work/20260930-fewstep-sampler/ (expected differences: checkpoint from the lane,
# runtime switch, leads 24,120, the resolver, the probe always on in the sandbox).
# The command is the one eval.cli predict builds (eval/cli/predict.py): python -m eval.predict.main --name-ckpt <BASE
# ckpt> --out-dir --members --steps --dates --input-root --allow-existing-out-dir --num-gpus-per-model 1
# --extra-args-json <resolved predict.sampler> --local-scope-json <predict.local_scope>; one GPU, so no srun (as
# eval.cli predict). It is called directly because tc_o320_o1280 allows predict on atos_ag only.
#
#   sbatch se_tc_predict_t1d.sbatch ARM LANE SEED RR RUNTIME [DATES] [STEPS] [MEMBERS] [PROBE]
#     SEED 756 = ANEMOI_BASE_SEED unset (default 756); 757 = export ANEMOI_BASE_SEED=757 (runtime base 757000)
#     RUNTIME sandbox = the fresh `exp new` sandbox with anemoi-core 27391c1; certified = the certified venv (gate G1 only)
#     T1D_HOST (passed by the submit script in --export, Atos has SBATCH_EXPORT=NONE; default ac): ac = Atos AC, A100,
#       x86_64; ag = Atos AG, GH200, aarch64. sandbox = the sandbox's own uv venv .venv-<arch> through its guarded
#       activate.sh + the checks of t1d_activate_sandbox (T1D_RUNTIME=overlay = AG opt-in: certified arm venv + PYTHONPATH
#       onto the sandbox code); certified = ~/dev/.ds-260612 (ac) or ~/dev/.ds-ag-260616 (ag). The submit script passes
#       the AG resources on the sbatch command line (1 GH200, 32 CPUs, 120G), overriding the AC header below.
#     PROBE 1 (default in the sandbox) = eval.predict.main through common/probe_sampler.py (prints the applied schedule
#       and the denoiser calls; the wrappers return the originals' results unchanged)
set -uo pipefail
ARM="${1:?arm}"; LANE="${2:?lane}"; SEED="${3:?756|757}"; RR="${4:?run root}"; RUNTIME="${5:?sandbox|certified}"
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
DATES="${6:-$T1D_DATES}"; STEPS="${7:-$T1D_STEPS}"; MEMBERS="${8:-$T1D_MEMBERS}"
PROBE="${9:-$([[ $RUNTIME == sandbox ]] && echo 1 || echo 0)}"
case "$SEED" in 756) unset ANEMOI_BASE_SEED ;; 757) export ANEMOI_BASE_SEED=757 ;; *) echo "FATAL seed $SEED"; exit 1 ;; esac
HOST="${T1D_HOST:-ac}"; ARCH=$(uname -m)
case "$HOST:$ARCH" in ac:x86_64|ag:aarch64) ;; *) echo "FATAL T1D_HOST=$HOST on a $ARCH node (ac = x86_64 A100, ag = aarch64 GH200)"; exit 3 ;; esac
PRED=$RR/predictions; META=$RR/meta
if [[ -n "$(ls -A "$PRED" 2>/dev/null)" ]]; then echo "FATAL $PRED is not empty"; exit 1; fi
mkdir -p "$PRED" "$META"
module load ecmwf-toolbox 2>/dev/null || true
[[ "$RUNTIME" == certified && "$ARM" != p12m_pw30_c0 && "$ARM" != p12m_c0_pw16_s1k ]] && { echo "FATAL the certified venv has no custom schedule; $ARM runs in the sandbox only"; exit 1; }
case "$HOST:$RUNTIME" in
  ac:sandbox|ag:sandbox)   # the sandbox's own uv venv (.venv-x86_64 / .venv-aarch64) via its guarded activate.sh,
                           # or T1D_RUNTIME=overlay on AG (opt-in); see t1d_activate_sandbox in t1d_env.sh
    t1d_activate_sandbox || exit $? ;;
  ac:certified)
    set +u; unset PYTHONPATH; source "$T1D_CERT_VENV/bin/activate" || exit 3; set -u
    CORE_DIR=$(python -c "import anemoi.models, os; print(os.path.dirname(anemoi.models.__file__))") ;;
  ag:certified)
    set +u; unset PYTHONPATH; source "$T1D_CERT_VENV_AG/bin/activate" || exit 3; set -u
    CORE_DIR=$(python -c "import anemoi.models, os; print(os.path.dirname(anemoi.models.__file__))") ;;
  *) echo "FATAL runtime $RUNTIME"; exit 1 ;;
esac
export DATA_DIR=/home/mlx/ai-ml/datasets/ DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/ OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/ INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ANEMOI_INFERENCE_NUM_CHUNKS=1 ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=1 ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCH_COMPILE_DISABLE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 HYDRA_FULL_ERROR=1
export TMPDIR=/tmp; mkdir -p /tmp/$USER/hydra
cd "$T1D_CODE" || exit 3
python "$T1D_S/resolve_lane_t1d.py" "$LANE" --outdir "$META" --check-files || exit 4
SAMPLER=$(cat "$META/$LANE.sampler.json"); SCOPE=$(cat "$META/$LANE.scope.json"); CKPT=$(cat "$META/$LANE.ckpt")
python -c "import anemoi.models.samplers.diffusion_samplers as d, torch; print('RUNTIME', d.__file__, 'custom' in d.NOISE_SCHEDULERS, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed "s/^/GPU /"
ENTRY=(-m eval.predict.main); [[ "$PROBE" == 1 ]] && ENTRY=("$T1D_S/common/probe_sampler.py")
echo "host=$(hostname) arch=$(uname -m) start=$(date -Is) dstools=$(git rev-parse --short HEAD) core=$(t1d_git_head "$CORE_DIR" 2>/dev/null | cut -c1-9 || true) runtime=$RUNTIME t1d_runtime=$T1D_RUNTIME cluster=$HOST arm=$ARM lane=$LANE ckpt=$CKPT dates=$DATES steps=$STEPS members=$MEMBERS precision=fp32 ANEMOI_BASE_SEED=${ANEMOI_BASE_SEED:-unset} probe=$PROBE input=$T1D_INPUT_ROOT rr=$RR"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 10 > "$META/nvsmi_${SLURM_JOB_ID:-local}.csv" & SIDE=$!
t0=$(date +%s)
python "${ENTRY[@]}" --name-ckpt "$CKPT" --out-dir "$PRED" --members "$MEMBERS" --steps "$STEPS" --dates "$DATES" \
  --input-root "$T1D_INPUT_ROOT" --allow-existing-out-dir --num-gpus-per-model 1 --precision fp32 \
  --extra-args-json "$SAMPLER" --local-scope-json "$SCOPE"
rc=$?
wall=$(( $(date +%s) - t0 )); kill $SIDE 2>/dev/null
nd=$(( $(echo "$DATES" | tr ',' '\n' | wc -l) * $(echo "$STEPS" | tr ',' '\n' | wc -l) * $(echo "$MEMBERS" | tr ',' '\n' | wc -l) ))
peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$META/nvsmi_${SLURM_JOB_ID:-local}.csv")
nf=$(ls "$PRED"/predictions_*.nc 2>/dev/null | wc -l)
echo "SE_TC_T1D arm=$ARM seed=$SEED runtime=$RUNTIME cluster=$HOST rc=$rc files=$nf wall_s=$wall draws=$nd s_per_draw=$(awk -v w=$wall -v n=$nd 'BEGIN{printf "%.1f", w/n}') peak_mem_mib=$peak"
exit $rc
