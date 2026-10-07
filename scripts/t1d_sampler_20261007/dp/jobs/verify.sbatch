#!/bin/bash --login
#SBATCH --job-name=t1d_verify
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# T1d batch 2: verification draws. One bundle, the same seeds as the dense run, every candidate schedule of the
# schedule file (verify_schedules.py make), one draw per (schedule, seed), the model loaded ONCE (in-process cache).
# Same runtime, box, cut graph and sampler as traj_states.sbatch; only the schedule changes.
#   sbatch --output=<log> verify.sbatch DATE STEP OUTROOT SCHEDFILE SEED [SEED ...]
set -uo pipefail
DATE="${1:?}"; STEP="${2:?}"; OUTROOT="${3:?}"; SCHEDS="${4:?}"; shift 4; SEEDS=("$@")
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/dp/jobs/_runtime.sh" || exit 2
WINDOW=$(t1d_window "$DATE" "$STEP") || exit 2
cd "$T1D_CODE" || exit 5
echo "host=$(hostname) arch=$(uname -m) start=$(date -Is) cluster=$HOST dstools=$(git rev-parse --short HEAD) core=$(t1d_git_head "$CORE_DIR" | cut -c1-9) date=$DATE step=$STEP seeds=${SEEDS[*]} window=$WINDOW scheds=$SCHEDS"
t0=$(date +%s)
python -m scripts.t1d_sampler_20261007.dp.verify_schedules run --schedules "$SCHEDS" \
  --checkpoint "$T1D_DIAG_CKPT" --bundle-dir "$T1D_DIAG_BUNDLES" --date "$DATE" --step "$STEP" --seeds "${SEEDS[@]}" \
  --window "$WINDOW" --scope "$T1D_DIAG_SCOPE" --sampler-params "$T1D_DIAG_SPJ" --out-root "$OUTROOT"
rc=$?
echo "T1D_VERIFY cluster=$HOST date=$DATE step=$STEP rc=$rc wall_s=$(( $(date +%s) - t0 ))"
exit $rc
