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
# T1d dense-trajectory diagnostic (docs: 20261007_T1d_sampler_beat_1p2M). One bundle (date, lead, member 01), several
# seeds, the dense churn-off Heun schedule, the 1.2M parent 551dfd1e step 400k, the Idalia box.
# Runtime: jobs/_runtime.sh = stage A's sandbox runtime (t1d_env.sh, sandbox t1d-stagea, gate G1).
#   T1D_HOST=ac (default): Atos AC, one A100 (this header). T1D_HOST=ag: Atos AG, one GH200; submit_traj.sh passes the
#                          AG resources (t1d_host_opts: 32 CPUs, 120G) on the sbatch line.
#   ROUTE=box (default)  : regional Franklin-Idalia cut graph on one GPU, about 1.1 s per denoiser call on A100.
#   ROUTE=global (AC only, fallback): global graph sharded over 4 A100 (submit with --ntasks-per-node=4
#                          --gpus-per-node=4 --mem=0), about 3 s per call.
# Usage (submit_traj.sh builds the calls):  sbatch --output=<log> traj_states.sbatch DATE STEP NLEV OUTDIR SEED [SEED ...]
set -uo pipefail
DATE="${1:?date YYYYMMDD}"; STEP="${2:?lead, zero-padded e.g. 024}"; NLEV="${3:?dense levels: 12 (smoke) or 240}"
OUTDIR="${4:?output dir}"; shift 4; SEEDS=("$@"); [[ ${#SEEDS[@]} -gt 0 ]] || { echo "FATAL no seeds"; exit 2; }
ROUTE="${ROUTE:-box}"
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/dp/jobs/_runtime.sh" || exit 2
[[ "$ROUTE" == global && "$HOST" != ac ]] && { echo "FATAL ROUTE=global is AC only"; exit 2; }
SCHED=$T1D_CODE/scripts/t1d_sampler_20261007/dp/schedules/dense${NLEV}.json
[[ -f "$SCHED" ]] || { echo "FATAL missing schedule $SCHED"; exit 1; }
NSJ=$(cat "$SCHED")
WINDOW=$(t1d_window "$DATE" "$STEP") || { echo "FATAL no Idalia window for $DATE/$STEP"; exit 2; }
cd "$T1D_CODE" || exit 5
ARGS=(--checkpoint "$T1D_DIAG_CKPT" --output-dir "$OUTDIR"
      --bundle-dir "$T1D_DIAG_BUNDLES" --dates "$DATE" --members 01 --steps "$STEP"
      --mode trajectory --lockin --save-trajectory-states
      --num-steps "$NLEV" --seeds "${SEEDS[@]}"
      --eye-radius-km 500 --auto-window "$WINDOW" --precision fp32
      --noise-scheduler-json "$NSJ" --sampler-params-json "$T1D_DIAG_SPJ")
echo "host=$(hostname) arch=$(uname -m) start=$(date -Is) cluster=$HOST route=$ROUTE dstools=$(git rev-parse --short HEAD) core=$(t1d_git_head "$CORE_DIR" | cut -c1-9) ckpt=$T1D_DIAG_CKPT date=$DATE step=$STEP member=01 seeds=${SEEDS[*]} nlev=$NLEV calls_per_seed=$((2*NLEV-1)) window=$WINDOW out=$OUTDIR bundles=$T1D_DIAG_BUNDLES"
mkdir -p "$OUTDIR"
t0=$(date +%s)
if [[ "$ROUTE" == box ]]; then
  python -m interp trajectory "${ARGS[@]}" --local-scope-json "$T1D_DIAG_SCOPE"
  rc=$?
elif [[ "$ROUTE" == global ]]; then
  srun --ntasks 4 --ntasks-per-node 4 python -m interp trajectory "${ARGS[@]}"
  rc=$?
else
  echo "FATAL ROUTE=$ROUTE"; exit 2
fi
wall=$(( $(date +%s) - t0 ))
echo "T1D_TRAJ cluster=$HOST route=$ROUTE date=$DATE step=$STEP seeds=${SEEDS[*]} nlev=$NLEV rc=$rc wall_s=$wall s_per_call_incl_load=$(python -c "print(round($wall/(${#SEEDS[@]}*(2*$NLEV-1)),3))")"
ls -la "$OUTDIR"
exit $rc
