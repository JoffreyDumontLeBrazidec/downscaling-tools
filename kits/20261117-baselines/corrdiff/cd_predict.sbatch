#!/bin/bash
# CorrDiff baseline inference on one evaluation set: SET=xset1 or s25 (inputs in $CD/sets/$SET/in, outputs in .../out).
#   sbatch --export=ALL,SET=xset1,CD_REG_CKPT=<reg .mdlus>,CD_RES_CKPT=<diffusion .mdlus> cd_predict.sbatch
#SBATCH --job-name=cd_predict
#SBATCH --account=DestE_340_26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/logs/%x_%j.out
set -euo pipefail
source /leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff/kit/cd_env.sh
: "${SET:?}" "${CD_REG_CKPT:?}" "${CD_RES_CKPT:?}"
cd $CD_REPO/examples/weather/corrdiff
torchrun --standalone --nproc_per_node=4 $CD_KIT/cd_predict.py --cfg conf/config_training_e2v4_diffusion.yaml \
  --reg $CD_REG_CKPT --res $CD_RES_CKPT --inputs $CD/sets/$SET/in --out $CD/sets/$SET/out
ls $CD/sets/$SET/out/cdout_*.npz | wc -l
