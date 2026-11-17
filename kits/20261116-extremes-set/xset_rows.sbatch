#!/bin/bash
#SBATCH --job-name=xset_rows
#SBATCH --qos=nf
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/%x-%j.out
# A set other than XSET-1 (S25, ...): Atos drops the caller's environment (SBATCH_EXPORT=NONE and the ECMWF sbatch
# wrapper), so exported XSET_* never arrive. Put them in xset_set.env in the directory you submit from (17 Nov 2026).
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/xset_set.env" ]] && source "$SLURM_SUBMIT_DIR/xset_set.env"
echo "[xset] set work ${XSET_WORK:-XSET-1 default} lists ${XSET_LISTS:-default} cache ${XSET_CACHE:-default}"
# Evaluation points, then the 9 km ensemble row files (with the input row and IMERG inside) and the global tail
# histograms of the given start dates. Needs the input stores (xset_build_in), the GRIB of the 9 km ensemble and the
# station cache holding these dates. Usage: sbatch --job-name=xset_rows_<i> xset_rows.sbatch <yyyymmdd> [...]
set -o pipefail; ulimit -c 0; module load python3 ecmwf-toolbox >/dev/null 2>&1
A=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}
K=$A/downscaling-tools/kits/20261116-extremes-set
DS=/ec/ai/project/ai-ml/datasets
GOUT=$DS/downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-24h-12h-v4.zarr
GIN=$DS/downscaling-od-cf-eefh-0001-mars-o320-2003-2023-24h-12h-v4.zarr
args=""; for d in "$@"; do args="$args --date $d"; done
echo "[rows] $args $(date -u +%FT%TZ)"
L=${XSET_LISTS:-$K/lists}
python3 $K/xset_points.py --cache ${XSET_CACHE:-$A/cache/xset1_v1} --tc-cases $L/tc_cases.csv --rain-events $L/rain_events.csv \
  --gout $GOUT --gin $GIN --out $A/points $args || { echo "[rows] POINTS_FAIL $(date -u +%FT%TZ)"; exit 1; }
python3 $K/xset_rows_date.py --gout $GOUT --points $A/points --grib $A/grib --stores $A/stores --out $A $args \
  || { echo "[rows] ROWS_FAIL $(date -u +%FT%TZ)"; exit 1; }
du -sh $A/rows/ENS9 $A/hist $A/points | tr '\n' ' '; echo
echo "[rows] JOB_DONE $(date -u +%FT%TZ)"
