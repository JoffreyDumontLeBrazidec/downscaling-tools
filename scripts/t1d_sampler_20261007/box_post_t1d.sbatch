#!/bin/bash --login
#SBATCH --job-name=t1d_tc_intensity
#SBATCH --qos=nf
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ecm5702/work-t1d-20261007/logs/%x_%j.out
# Campaign T1d stage A (2026-10-07): per-draw storm intensity (event-box wind max and msl min per date, lead and member)
# of the five stage A runs, with its pooled checks against the tc evaluator and the summary (the stage 3 box_post_s3
# pattern; tc_intensity_t1d.py takes the runs as arguments instead of a hard-coded table).
set -uo pipefail
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
N=$T1D_W/stageA/notes; mkdir -p $N
set +u; source "$T1D_CERT_VENV/bin/activate" || exit 3; set -u
A=(); for spec in "p12m_pw30_c0_s756 p12m_pw30_c0 756" "p12m_pw30_c0_r757 p12m_pw30_c0 757" "p12m_c0_pw16_s1k p12m_c0_pw16_s1k 756" "p12m_st2 p12m_st2 756" "p12m_st4 p12m_st4 756"; do
  read -r NAME ARM SEED <<<"$spec"; A+=("$NAME=$(t1d_root $ARM $SEED)"); done
python $T1D_S/tc_intensity_t1d.py --out $N/tc_intensity_stageA.tsv --eval-subdir eval_stageA "${A[@]}" > $N/tc_intensity_stageA_checks.txt || { cat $N/tc_intensity_stageA_checks.txt; exit 1; }
python $T1D_S/common/tc_intensity_summary.py $N/tc_intensity_stageA.tsv p12m_pw30_c0_s756 p12m_pw30_c0_r757 p12m_c0_pw16_s1k p12m_st2 p12m_st4 > $N/tc_intensity_stageA_summary.txt
rc=$?
cat $N/tc_intensity_stageA_checks.txt $N/tc_intensity_stageA_summary.txt
echo "T1D_TC_INTENSITY rc=$rc"
exit $rc
