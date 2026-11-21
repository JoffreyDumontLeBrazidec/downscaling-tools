#!/bin/bash
#SBATCH --job-name=xset_score
#SBATCH --qos=nf
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/%x-%j.out
# A set other than XSET-1 (S25, ...): Atos drops the caller's environment (SBATCH_EXPORT=NONE and the ECMWF sbatch
# wrapper), so exported XSET_* never arrive. Put them in xset_set.env in the directory you submit from (17 Nov 2026).
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/xset_set.env" ]] && source "$SLURM_SUBMIT_DIR/xset_set.env"
echo "[xset] set work ${XSET_WORK:-XSET-1 default} lists ${XSET_LISTS:-default} cache ${XSET_CACHE:-default}"
# Scores of one row of the extremes set on Atos: stations (obs_points, per model cycle), cyclones against IBTrACS and
# IMERG discs. Row files in A/rows/<LABEL>. With "refs" as second argument the cyclone and IMERG scores of the INPUT row
# (native x) and of the single 9 km field CF9 are written too: use it on the ENS9 row only.
# Usage: sbatch --job-name=xset_score_<LABEL> xset_score.sbatch <LABEL> [refs|screen]
# "screen" (SCREEN-TC row, 21 Nov 2026): cyclone cases only (--tc-only); no stations (5-member station scores are not
# trusted, audit note s8), no IMERG, no tail check. Read it with xset_screen_read.py, never with xset_table.py.
# Another set (17 Nov 2026): export XSET_WORK, XSET_LISTS, XSET_CACHE, XSET_DATES first (defaults = XSET-1) and pass -o.
set -o pipefail; ulimit -c 0
A=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}
T=$A/downscaling-tools; K=$T/kits/20261116-extremes-set
module load python3 vtb ecmwf-toolbox >/dev/null 2>&1
export TMPDIR=$A/tmp/$SLURM_JOB_ID PYTHONPATH=$T:$K; mkdir -p "$TMPDIR" $A/scores
lab=${1:?row label}; refs=""; [[ ${2:-} == refs ]] && refs="--with-refs"
screen=0; [[ ${2:-} == screen ]] && screen=1
[[ $screen == 1 && $lab != *_SCRTC ]] && { echo "[score] FATAL screen scoring of $lab: a SCREEN-TC row ends in _SCRTC"; exit 2; }
n=$(ls $A/rows/$lab/predictions_*_step*.nc 2>/dev/null | wc -l)
echo "[score] $lab: $n files, $(date -u +%FT%TZ)"
cd $T
if [[ $screen == 1 ]]; then
  python3 $K/xset_score_events.py --pred $A/rows/$lab --label $lab --tc-cases ${XSET_LISTS:-$K/lists}/tc_cases.csv --tc-only \
    --out $A/scores/$lab || { echo "[score] EVENTS_FAIL $lab"; exit 1; }
  echo "$lab files=$n screen=TC job=$SLURM_JOB_ID kit=$(git -C $T rev-parse --short HEAD) $(date -u +%FT%TZ)" > $A/scores/$lab/SCORE_DONE
  echo "[score] SCORE_DONE $lab SCREEN-TC (cyclones only) $(date -u +%FT%TZ)"; exit 0
fi
python3 $K/xset_score_events.py --pred $A/rows/$lab --label $lab --tc-cases ${XSET_LISTS:-$K/lists}/tc_cases.csv \
  --rain-events ${XSET_LISTS:-$K/lists}/rain_events.csv --out $A/scores/$lab $refs || { echo "[score] EVENTS_FAIL $lab"; exit 1; }
echo "[score] events done $(date -u +%FT%TZ)"
python3 $K/xset_score_stations.py --pred $A/rows/$lab --label $lab --cache ${XSET_CACHE:-$A/cache/xset1_v1} --out $A/scores/$lab \
  --dates-csv ${XSET_DATES:-$K/lists/dates_xset1.csv} \
  --overwrite || { echo "[score] STATIONS_FAIL $lab"; exit 1; }
# tails of the row against the 9 km control and members at the evaluation points (not fatal; read with the table)
python3 $K/xset_check_cf_vs_pf.py $A/rows/$lab > $A/scores/$lab/tail_check.txt 2>&1 || echo "[score] tail check failed (not fatal)"
echo "$lab files=$n job=$SLURM_JOB_ID kit=$(git -C $T rev-parse --short HEAD) $(date -u +%FT%TZ)" > $A/scores/$lab/SCORE_DONE
echo "[score] SCORE_DONE $lab $(date -u +%FT%TZ)"
