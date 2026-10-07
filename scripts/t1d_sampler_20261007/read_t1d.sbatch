#!/bin/bash --login
#SBATCH --job-name=t1d_read_stageA
#SBATCH --qos=nf
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/ecm5702/work-t1d-20261007/logs/%x_%j.out
# Campaign T1d stage A (2026-10-07): the stage-1 reading rule of T1 section 6 (scripts/fewstep_sampler_20260930/
# read_stage1.py, unchanged): baseline p12m_pw30_c0 at base seed 756; ONE seed replicate, the same lane at 757000, sets
# the band (1.5 x its deviation, never below the floors fair CRPS 1 %, nMSE 1 %, spread 5 %, texture ratio 0.03,
# grain 0.05, retention 0.02); arms p12m_c0_pw16_s1k, p12m_st2, p12m_st4. The replicate is also listed as an arm
# (p12m_pw30_c0_r757), which shows how a null arm reads against the band it set (inside by construction).
set -uo pipefail
source "${T1D_CODE:-/home/ecm5702/work-t1d-20261007/code/downscaling-tools}/scripts/t1d_sampler_20261007/t1d_env.sh" || exit 2
set +u; source "$T1D_CERT_VENV/bin/activate" || exit 3; set -u
B=$(t1d_root p12m_pw30_c0 756)/eval_stageA; R=$(t1d_root p12m_pw30_c0 757)/eval_stageA
python $T1D_CODE/scripts/fewstep_sampler_20260930/read_stage1.py --baseline $B --seed $R \
  --arm p12m_c0_pw16_s1k=$(t1d_root p12m_c0_pw16_s1k)/eval_stageA --arm p12m_st2=$(t1d_root p12m_st2)/eval_stageA \
  --arm p12m_st4=$(t1d_root p12m_st4)/eval_stageA --arm p12m_pw30_c0_r757=$R \
  --calls p12m_c0_pw16_s1k=31 --calls p12m_st2=35 --calls p12m_st4=39 --calls p12m_pw30_c0_r757=59 \
  --out $T1D_W/stageA/read_stageA
rc=$?; echo "T1D_READ rc=$rc"; exit $rc
