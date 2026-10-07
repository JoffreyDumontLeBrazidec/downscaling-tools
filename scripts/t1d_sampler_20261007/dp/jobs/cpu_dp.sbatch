#!/bin/bash --login
#SBATCH --job-name=t1d_dp
#SBATCH --qos=nf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# T1d CPU stage: cost matrices -> DP schedules -> lock-in sigma50 -> docs bundle staging.
#   T1D_SB=<sandbox> T1D_OUT=<output root> sbatch --output=$T1D_OUT/logs/%x_%j.out cpu_dp.sbatch
set -euo pipefail
SB="${T1D_SB:?}"; OUT="${T1D_OUT:?}"
source "$SB/activate.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$SB/downscaling-tools"
echo "host=$(hostname) start=$(date -Is) dstools=$(git rev-parse --short HEAD)"
python -m scripts.t1d_sampler_20261007.dp.cost_matrix \
  --inputs "$OUT/d2023*_l*/trajectory_states_s*.npz" --out-dir "$OUT/cost" --workers 4
python -m scripts.t1d_sampler_20261007.dp.dp_schedule --cost-dir "$OUT/cost" --out-dir "$OUT/dp"
python -m scripts.t1d_sampler_20261007.dp.lockin_read \
  --inputs "$OUT/d2023*_l*/trajectory.json" --out "$OUT/lockin/lockin_sigma50.json"
python -m scripts.t1d_sampler_20261007.dp.lockin_read --from-states \
  --inputs "$OUT/d2023*_l*/trajectory_states_s*.npz" --out "$OUT/lockin/lockin_sigma50_from_states.json"
# stage the docs bundle (small files only; the npz states and full cost matrices stay here)
B="$OUT/bundle_for_docs/diag"; rm -rf "$B"; mkdir -p "$B"/{cost,dp,lockin,logs}
cp "$OUT/cost/cost_manifest.json" "$B/cost/"
cp "$OUT/dp/bundle/"cost_C_*.npz "$B/cost/"
cp "$OUT/dp/dp_schedules.json" "$OUT/dp/dp_summary.md" "$OUT/dp/c0_30_running_cost.json" "$B/dp/"
cp -r "$OUT/dp/schedules" "$B/dp/"
cp "$OUT/lockin/"*.json "$B/lockin/"
cp "$OUT/jobs.txt" "$B/" 2>/dev/null || true
for f in "$OUT"/logs/*.out; do tail -n 60 "$f" > "$B/logs/$(basename "$f")"; done
du -sh "$B"; find "$B" -type f | sort
echo "T1D_DP done $(date -Is)"
