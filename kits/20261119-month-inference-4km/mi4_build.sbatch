#!/bin/bash
#SBATCH --job-name=mi4_bin
#SBATCH --qos=nf
#SBATCH --account=ecaifs
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=/ec/res4/scratch/ecm5702/agent-work/20261119-mi4k-s26/logs/%x-%j.out
# MI4-S26 global input store of the given start dates. After a date's store passes its check, its temporary GRIB (pl_gNN, sfx)
# is deleted and the kept native GRIB (sfq: 10u 10v 2d 2t msl; tp: tp cp) moves to keep/. Usage: sbatch mi4_build.sbatch <d> [...]
ulimit -c 0; module load python3 ecmwf-toolbox >/dev/null 2>&1
K=/ec/res4/scratch/ecm5702/agent-work/20261119-mi4k-s26/dt/kits/20261119-month-inference-4km
source $K/mi4_set.sh
[[ -f $MI4_A/state/STOP ]] && { echo "[bin] STOP file, nothing done"; exit 0; }
echo "[bin] host $(hostname) git $(git -C $K rev-parse --short HEAD) meta $MI4_META out $MI4_A/stores dates $* $(date -u +%FT%TZ)"
for d in "$@"; do
  /usr/bin/time -v python3 $K/mi4_build_in.py --meta $MI4_META --grib $MI4_A/grib --out $MI4_A/stores --date $d --threads 8 \
    2> >(grep -E "Maximum resident|Elapsed|Error|error|Traceback|assert" >&2) || { echo "[bin] BUILD_FAIL $d $(date -u +%FT%TZ)"; exit 1; }
  if [[ -f $MI4_A/stores/check/in_$d.done ]]; then
    mkdir -p $MI4_A/keep
    for n in sfq tp; do [[ -f $MI4_A/grib/${n}_$d.grib ]] && mv $MI4_A/grib/${n}_$d.grib $MI4_A/grib/${n}_$d.grib.ok $MI4_A/keep/; done
    rm -f $MI4_A/grib/pl_g??_$d.grib $MI4_A/grib/pl_g??_$d.grib.ok $MI4_A/grib/sfx_$d.grib $MI4_A/grib/sfx_$d.grib.ok
    echo "[bin] GRIB_CLEANED $d: kept $(ls $MI4_A/keep/*_$d.grib | wc -l) files, store $(du -sb --apparent-size $MI4_A/stores/in/$d.zarr | cut -f1) B"
  fi
done
echo "[bin] JOB_DONE $(date -u +%FT%TZ)"
