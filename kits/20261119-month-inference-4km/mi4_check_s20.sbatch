#!/bin/bash
#SBATCH --job-name=mi4_chk20
#SBATCH --qos=nf
#SBATCH --account=ecaifs
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261119-mi4k-s26/logs/%x-%j.out
ulimit -c 0; module load python3 ecmwf-toolbox >/dev/null 2>&1
K=/ec/res4/scratch/ecm5702/agent-work/20261119-mi4k-s26/dt/kits/20261119-month-inference-4km
source $K/mi4_set.sh
python3 $K/mi4_check_s20.py --meta $MI4_META --date ${1:-20250808} --lead 24 --lead 48 > $MI4_A/state/check_s20_${1:-20250808}.txt 2>&1; rc=$?
cat $MI4_A/state/check_s20_${1:-20250808}.txt; echo "[chk] rc=$rc"; exit $rc
