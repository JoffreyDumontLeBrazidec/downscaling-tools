#!/bin/bash
# Extremes set XSET-1: gate before any row. The out-of-store path (member store + computed forcings, no dataset sample in the
# batch) must reproduce the 4 Nov member read bit for bit: GL512 237k EMA-2000, 20230101 +24 h, members 1-2, E2 box, against
# outputs/ens_member_gl512_20261104/pred/GL512_237e2_m_L024/predictions_20230101_step024.nc. Writes test/GL512_237k/XSET_TEST_PASS.
#SBATCH --job-name=xset_test
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/logs/%x-%j.out
set -o pipefail; ulimit -c 0
K=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261116-extremes-set/downscaling-tools/kits/20261116-extremes-set/jupiter
export MEMBERS=2
export MEMBER_IN_STORE=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261104-ens-member-gl512/stores/gl_in_lres_o320_pf.zarr
export TEST_REF=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/ens_member_gl512_20261104/pred/GL512_237e2_m_L024/predictions_20230101_step024.nc
bash $K/xset_run.sh GL512_237k test 20230101:24 29641 || { echo "[xset] XSET_TEST_FAIL $(date -u +%FT%TZ)"; exit 1; }
echo "[xset] XSET_TEST_JOB_DONE $(date -u +%FT%TZ)"
