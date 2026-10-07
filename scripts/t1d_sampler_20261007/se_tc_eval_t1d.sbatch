#!/bin/bash --login
#SBATCH --job-name=t1d_eval
#SBATCH --qos=nf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/work-t1d-20261007/logs/%x_%j.out
# Campaign T1d stage A (2026-10-07): all six stage-1 evaluators on the 100-draw box files of one run (5 dates x leads
# 24 and 120 x 10 members = 10 files): --only tc,texture,wind_extremes,probabilistic,surface,shape (stage 3 ran
# --only tc and lost the peak-wind row of wind_extremes). The stage-1 command line (T1 bundle README), venv .ds-260612,
# host atos_ac, from the T1d worktree. Refuses an existing output dir and fewer than 10 prediction files.
#   sbatch se_tc_eval_t1d.sbatch <lane> <run root>      (writes <run root>/eval_stageA)
set -uo pipefail
LANE="${1:?lane}"; RR="${2:?run root}"
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
PRED=$RR/predictions; OUTD=$RR/eval_stageA
[[ -e "$OUTD" ]] && { echo "FATAL output dir exists: $OUTD"; exit 1; }
n=$(ls "$PRED"/predictions_*.nc 2>/dev/null | wc -l)
echo "host=$(hostname -s) start=$(date -Is) lane=$LANE pred=$PRED files=$n/10 out=$OUTD only=$T1D_ONLY"
[[ "$n" -ge 10 ]] || { echo "FATAL expected 10 files"; exit 1; }
cd "$T1D_CODE" || exit 3
set +u; unset PYTHONPATH; source "$T1D_CERT_VENV/bin/activate" || exit 3; set -u
echo "dstools=$(git rev-parse --short HEAD)"
python -m eval.cli evaluate --lane "$LANE" --host atos_ac --predictions-dir "$PRED" --only "$T1D_ONLY" \
  --steps $T1D_STEPS --output-dir "$OUTD"
rc=$?
for ev in ${T1D_ONLY//,/ }; do [[ -s "$OUTD/evaluators/$ev/metrics.json" ]] && echo "  $ev metrics.json OK" || { echo "  $ev metrics.json MISSING"; [[ $rc == 0 ]] && rc=5; }; done
echo "T1D_EVAL lane=$LANE rc=$rc"
exit $rc
