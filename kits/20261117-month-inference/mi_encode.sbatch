#!/bin/bash
#SBATCH --job-name=mi_enc
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference/logs/%x-%j.out
# MI-S25: GRIB for quaver of one row and the given dates (needs rows/<ROW>/<date>.ok from mi_copy_from_jupiter.sh).
# Usage: sbatch --job-name=mi_enc_<row>_<i> mi_encode.sbatch <ROW> <EXPVER label> <yyyymmdd> [...]
ulimit -c 0; module load python3 ecmwf-toolbox >/dev/null 2>&1
A=/ec/res4/scratch/ecm5702/agent-work/20261117-month-inference
row=${1:?row}; ev=${2:?expver}; shift 2
mkdir -p $A/grib_out/$row
rc=0
for d in "$@"; do
  [[ -f $A/rows/$row/$d.ok ]] || { echo "[enc] FAIL $d not copied"; rc=1; continue; }
  [[ -s $A/grib_out/$row/${row}_$d.grib ]] && { echo "have $d"; continue; }
  python3 $A/downscaling-tools/kits/20261117-month-inference/mi_encode_grib.py --rowdir $A/rows/$row --date $d \
    --template $A/grib/tmpl_o1280.grib --out $A/grib_out/$row/${row}_$d.grib --expver $ev || { echo "[enc] FAIL $d"; rc=1; }
done
echo "[enc] JOB_DONE rc=$rc $(date -u +%FT%TZ)"; exit $rc
