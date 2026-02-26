#!/bin/bash
#SBATCH --job-name=spectra_j24v
#SBATCH --qos=nf
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/home/ecm5702/perm/eval/j24v/logs/spectra_j24v_%j.out

set -euo pipefail
mkdir -p /home/ecm5702/perm/eval/j24v/logs

module unload ecmwf-toolbox || true
module load ifs

expid="j24v"
thegrid="o320"
thedate="20230826/to/20230827/by/1"
thestep="144"
outbase="/home/ecm5702/perm/ai_spectra/${expid}"
mkdir -p "${outbase}/grb" "${outbase}/spectral_harmonics" "${outbase}/spectra"

params=("2t:sfc" "10u:sfc" "10v:sfc" "sp:sfc" "t:850" "z:500")
for param_level in "${params[@]}"; do
  param="${param_level%%:*}"
  level="${param_level##*:}"
  outdir="${outbase}/grb/${param}_${level}"
  mkdir -p "${outdir}"
  if [ "${level}" = "sfc" ]; then levtype="sfc"; else levtype="pl"; fi
  if [ "${level}" = "sfc" ]; then
mars <<MARS
retrieve,
 type=pf,
 date=${thedate},
 levtype=${levtype},
 param=${param},
 step=${thestep},
 time=0000,
 number=1/2,
 grid=${thegrid},
 stream=enfo,
 class=rd,
 expver=${expid},
 area=89/0/-89/360,
 target="${outdir}/[expver]_[date]_[step]_[number]_nopoles.grb"
MARS
  else
mars <<MARS
retrieve,
 type=pf,
 date=${thedate},
 levtype=${levtype},
 level=${level},
 param=${param},
 step=${thestep},
 time=0000,
 number=1/2,
 grid=${thegrid},
 stream=enfo,
 class=rd,
 expver=${expid},
 area=89/0/-89/360,
 target="${outdir}/[expver]_[date]_[step]_[number]_nopoles.grb"
MARS
  fi
done

module unload ecmwf-toolbox || true
module load eclib
module load pifsenv
module load ifs
export DR_HOOK_ASSERT_MPI_INITIALIZED=0

indir="${outbase}/grb"
wdir="${outbase}/spectral_harmonics"
mkdir -p "${wdir}"

param_dirs=("2t_sfc" "10u_sfc" "10v_sfc" "sp_sfc" "t_850" "z_500")
for param_dir in "${param_dirs[@]}"; do
  mkdir -p "${wdir}/${param_dir}"
  for xfile in ${indir}/${param_dir}/${expid}_*.grb; do
    [ -f "${xfile}" ] || continue
    bname=$(basename "${xfile}")
    gptosp.ser -l -g "${xfile}" -S "${wdir}/${param_dir}/${bname}_sh"
  done
done

module unload ifs || true
module load ecmwf-toolbox
source /home/ecm5702/dev/.ds-dyn/bin/activate
python /home/ecm5702/dev/post_prepml/spectra/spectra_ml/individual_files/compute_spectra-3.py --expver ${expid}
