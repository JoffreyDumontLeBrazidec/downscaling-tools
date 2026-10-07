#!/bin/bash --login
#SBATCH --job-name=t1d_dp
#SBATCH --qos=nf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# T1d CPU stage (Atos AC, certified x86 venv as every stage-A CPU job): cost matrices -> DP schedules -> lock-in
# sigma50 -> verification schedule file -> docs bundle staging. With VERIFY=1 also the batch-2 analysis.
#   source <worktree>/scripts/t1d_sampler_20261007/t1d_env.sh; sbatch "$(t1d_export)" --output=<output root>/logs/%x_%j.out cpu_dp.sbatch <output root> [VERIFY 0|1]
# The output root and VERIFY are positional: Atos sets SBATCH_EXPORT=NONE, so T1D_OUT/VERIFY exported in the submit
# shell never reach the job (the environment fallbacks stay for an interactive run).
set -euo pipefail
OUT="${1:-${T1D_OUT:?output root: first argument}}"; VERIFY="${2:-${VERIFY:-0}}"
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh"
set +u; unset PYTHONPATH; source "$T1D_CERT_VENV/bin/activate"; set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$T1D_CODE"
echo "host=$(hostname) start=$(date -Is) dstools=$(git rev-parse --short HEAD) verify=${VERIFY:-0}"
M=scripts.t1d_sampler_20261007.dp
if [[ "${VERIFY:-0}" != 1 ]]; then
  python -m $M.cost_matrix --inputs "$OUT/d2023*_l*/trajectory_states_s*.npz" --out-dir "$OUT/cost" --workers 4
  python -m $M.dp_schedule --cost-dir "$OUT/cost" --out-dir "$OUT/dp"
  python -m $M.lockin_read --inputs "$OUT/d2023*_l*/trajectory.json" --out "$OUT/lockin/lockin_sigma50.json"
  python -m $M.lockin_read --from-states --inputs "$OUT/d2023*_l*/trajectory_states_s*.npz" \
    --out "$OUT/lockin/lockin_sigma50_from_states.json"
  python -m $M.verify_schedules make --dp-json "$OUT/dp/dp_schedules.json" --out "$OUT/verify/schedules_verify.json"
else
  python -m $M.verify_schedules analyze --verify-root "$OUT/verify" --dense-root "$OUT" \
    --dp-json "$OUT/dp/dp_schedules.json" --out "$OUT/verify/verify_table"
fi
# stage the docs bundle (small files only; the npz states and full cost matrices stay here)
B="$OUT/bundle_for_docs/diag"; mkdir -p "$B"/{cost,dp,lockin,logs,verify}
cp "$OUT/cost/cost_manifest.json" "$B/cost/"
cp "$OUT/dp/bundle/"cost_C_*.npz "$B/cost/"
cp "$OUT/dp/dp_schedules.json" "$OUT/dp/dp_summary.md" "$OUT/dp/c0_30_running_cost.json" "$B/dp/"
cp -r "$OUT/dp/schedules" "$B/dp/"
cp "$OUT/lockin/"*.json "$B/lockin/"
cp "$OUT/verify/schedules_verify.json" "$B/verify/" 2>/dev/null || true
cp "$OUT/verify/verify_table".{md,json} "$B/verify/" 2>/dev/null || true
cp "$OUT/jobs.txt" "$B/" 2>/dev/null || true
for f in "$OUT"/logs/*.out; do tail -n 60 "$f" > "$B/logs/$(basename "$f")"; done
du -sh "$B"; find "$B" -type f | sort
echo "T1D_DP done $(date -Is)"
