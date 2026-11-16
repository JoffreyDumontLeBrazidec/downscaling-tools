#!/bin/bash
#SBATCH --job-name=xset_cache
#SBATCH --qos=nf
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/logs/%x-%j.out
# Station cache of the extremes set (STVL, global, +12..+72 h, 2t 10ff msl and 12 h tp, quaver limits and analysis check) for
# the start dates of a csv with a "date" column, then the height column of the eefo O320 orography when its GRIB exists.
# A second call with more dates merges into the same cache. HPCPERM is full (16 Nov 2026): the cache lives on scratch.
# Usage: sbatch xset_station_cache.sbatch <dates csv>
set -eo pipefail; ulimit -c 0
A=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set
T=$A/downscaling-tools
C=$A/cache/xset1_v1
module load python3 vtb ecmwf-toolbox
export TMPDIR=$A/tmp/$SLURM_JOB_ID; mkdir -p "$TMPDIR" "$C"
dates=$(tail -n +2 "${1:?dates csv}" | cut -d, -f1 | paste -sd,)
echo "[cache] dates $dates $(date -u +%FT%TZ)"
cd $T
python3 -m eval.cli obs fetch --dates "$dates" --leads 12,24,36,48,60,72 --vars 2t,10ff,msl,tp --tp-periods 12 \
  --box -90,90,-180,180 --out $C
for o in $A/grib/orography_O320_eefo_*.grib; do
  [[ -f $o ]] || continue
  n=$(basename $o .grib); n=${n#orography_}; n=$(echo $n | tr 'A-Z' 'a-z')
  python3 -m eval.cli obs fetch --out $C --add-height $n=$o
done
head -c 600 $C/cache_meta.json; echo
for v in 2t 10ff msl tp; do echo "$v rows $(wc -l < $C/obs_$v.csv) header $(head -1 $C/obs_$v.csv)"; done
echo "[cache] CACHE_DONE $(date -u +%FT%TZ)"
