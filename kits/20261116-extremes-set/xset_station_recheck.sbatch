#!/bin/bash
#SBATCH --job-name=xset_strk
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/%x-%j.out
# Station scores of a row again on the present station cache, into A/scores_recheck/<LABEL> (A/scores/<LABEL> is not
# touched), then the comparison with the row's earlier scores (STATIONS_SAME or STATIONS_DIFFERENT with a per-date
# summary). Written 17 Nov 2026 after another set's jobs merged 30 dates into the XSET-1 cache and fetched three shared
# dates again (20250920, 20250923, 20250927). XSET-1 only: no xset_set.env is read here.
# Usage: sbatch --job-name=xset_strk_<LABEL> xset_station_recheck.sbatch <LABEL>
set -o pipefail; ulimit -c 0
A=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set
T=$A/downscaling-tools; K=$T/kits/20261116-extremes-set
module load python3 vtb ecmwf-toolbox >/dev/null 2>&1
export TMPDIR=$A/tmp/$SLURM_JOB_ID PYTHONPATH=$T:$K; mkdir -p "$TMPDIR" $A/scores_recheck
lab=${1:?row label}; O=$A/scores_recheck/$lab
echo "[strk] $lab $(date -u +%FT%TZ) kit=$(git -C $T rev-parse --short HEAD) cache=$A/cache/xset1_v1"
cd $T
python3 $K/xset_score_stations.py --pred $A/rows/$lab --label $lab --cache $A/cache/xset1_v1 --out $O \
  --dates-csv $K/lists/dates_xset1.csv --overwrite || { echo "[strk] STATIONS_FAIL $lab"; exit 1; }
python3 $K/xset_compare_scores.py $A/scores/$lab/obs_points/scores_per_date.csv $O/obs_points/scores_per_date.csv
echo "[strk] RECHECK_DONE $lab compare_rc=$? $(date -u +%FT%TZ)"
