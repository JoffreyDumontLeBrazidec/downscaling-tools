#!/bin/bash --login
#SBATCH --job-name=t1d_spectra_v3_box
#SBATCH --qos=nf
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ecm5702/work-t1d-20261007/logs/%x_%j.out
# Campaign T1d stage A (2026-10-07): v3 amplitude spectra on the Franklin/Idalia box (the stage 3 spectra_s3 pattern):
# common/plot_sampler_texture_v3.py --skip-maps --spectra-method linear, leads 24 and 120, every member of every file
# (100 draws per run). The truth (o1280 target) and the interpolated input are computed by the script as references
# next to the five runs; the first arm (p12m_pw30_c0 seed 756) sets the (date, lead) files every arm must have.
set -uo pipefail
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
OUT=$T1D_W/stageA/spectra_v3; mkdir -p $OUT
ARMS=(); for spec in "p12m_pw30_c0_s756 p12m_pw30_c0 756" "p12m_pw30_c0_s757 p12m_pw30_c0 757" "p12m_c0_pw16_s1k p12m_c0_pw16_s1k 756" "p12m_st2 p12m_st2 756" "p12m_st4 p12m_st4 756"; do
  read -r NAME ARM SEED <<<"$spec"; ARMS+=(--arm "$NAME=$(t1d_root $ARM $SEED)/predictions"); done
set +u; source "$T1D_CERT_VENV/bin/activate" || exit 3; set -u
env -u PYTHONPATH python $T1D_S/common/plot_sampler_texture_v3.py "${ARMS[@]}" --out $OUT --skip-maps --spectra-method linear \
  --spectra-leads 24,120 2>&1 | tee -a $OUT/run.log
echo "T1D_SPECTRA RC=${PIPESTATUS[0]}"
exit ${PIPESTATUS[0]}
