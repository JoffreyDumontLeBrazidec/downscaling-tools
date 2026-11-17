#!/bin/bash
#SBATCH --job-name=xset_tcmem
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/%x-%j.out
# Per-member cyclone file (tc_members.csv) of a row that was scored before 17 Nov 2026 (rows scored later have it in
# A/scores/<LABEL>). Cyclone cases only, written to A/scores_members/<LABEL>: A/scores/<LABEL> is not touched. The
# tc_cases_scored.csv written beside it must equal the one of A/scores/<LABEL> (printed: TC_SAME or TC_DIFFERENT).
# Usage: sbatch --job-name=xset_tcmem_<LABEL> xset_tc_members.sbatch <LABEL> [refs]      (refs on the ENS9 row only)
set -o pipefail; ulimit -c 0
A=${XSET_WORK:-/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set}
T=$A/downscaling-tools; K=$T/kits/20261116-extremes-set
module load python3 vtb ecmwf-toolbox >/dev/null 2>&1
export TMPDIR=$A/tmp/$SLURM_JOB_ID PYTHONPATH=$T:$K; mkdir -p "$TMPDIR" $A/scores_members
lab=${1:?row label}; refs=""; [[ ${2:-} == refs ]] && refs="--with-refs"
O=$A/scores_members/$lab
echo "[tcmem] $lab $(date -u +%FT%TZ) kit=$(git -C $T rev-parse --short HEAD)"
cd $T
python3 $K/xset_score_events.py --pred $A/rows/$lab --label $lab --tc-cases ${XSET_LISTS:-$K/lists}/tc_cases.csv \
  --out $O --tc-only $refs || { echo "[tcmem] FAIL $lab"; exit 1; }
if cmp -s $O/tc_cases_scored.csv $A/scores/$lab/tc_cases_scored.csv; then echo "[tcmem] TC_SAME $lab"; else echo "[tcmem] TC_DIFFERENT $lab"; fi
echo "[tcmem] TCMEM_DONE $lab lines=$(wc -l < $O/tc_members.csv) $(date -u +%FT%TZ)"
