#!/bin/bash
# Regression baseline REG of the paper (17 Nov 2026, downscaling-tools kits/20261117-baselines/reg): GL512 450k EMA-2000 warm
# start, training_approach deterministic (sigma 5e5 = MSE regression), 10 outputs, lr 1e-4 constant after 1k warm-up,
# dynamic 4,000 km caps + 800 km halo, 2 nodes x 4 GH200, one GPU per model (global batch 8), EMA [2000, 5000], 30,000 steps.
# Chain: 12 h segments, same file for every segment (tools/chain); the fresh segment trains 200 steps and runs the gate
# (reg_accept.sh) first; failure -> <chain root>/SMOKE_FAILED.txt, exit 1, no successor.
# Login-node checks: CFG_ONLY=1 bash <this file>; DRY_RUN=1 bash <this file>
#SBATCH --job-name=reg_gl512_450k_det
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --account=gkpdm
#SBATCH -p booster
#SBATCH --time=12:00:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline/logs/%x_%j.out
#SBATCH --error=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline/logs/%x_%j.err
EXP=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline
export LL_MODE=crop LL_RADIUS=4000 LL_CONFIG=reg_gl512_450k_det LL_TAG=reg_gl512_450k_det LL_LR=1.0e-4 LL_FREEZE=true LL_EMA=2000,5000 LL_END_STEP=30000 LL_INCHAIN_SMOKE=1 LL_NODES=2 LL_GPN=4 LL_CHAIN_NEXT=1 LL_SEED=756 LL_ACCEPT=0 LL_COMPILE=1
export LL_CKPT_ROOT=$EXP/chains/reg_gl512_450k_det
export LL_WARM_FILE=$EXP/warm/gl512_step450000_ema2000_warm.ckpt
export LL_WARM_SHA=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' $EXP/warm/WARM_SHA256)
export LL_EXTRA=
export ANEMOI_ENCODER_TRAP_LOG=1 H9_DATAWAIT_SYNC=1
ulimit -c 0
exec bash $EXP/jobs/reg_job.sh
