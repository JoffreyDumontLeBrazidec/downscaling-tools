# CorrDiff baseline environment on Leonardo (source it). Built once on a login node (SETUP in README.md).
export CD=/leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261117-corrdiff
export CD_VENV=$CD/venv-physicsnemo-2.2.2
export CD_REPO=$CD/physicsnemo                     # git checkout of NVIDIA/physicsnemo at tag v2.2.2 (072465a1)
export CD_KIT=$CD/kit                              # this folder (downscaling-tools kits/20261117-baselines/corrdiff)
export CD_DATA=$CD/data/e2v4_box_0p0694.zarr
export E2V4=/leonardo_work/DestE_340_26/users/jdumontl/data/o320_o1280_europe_e2_v4
export E2=/leonardo_work/DestE_340_26/users/jdumontl/data/o320_o1280_europe_e2
export HF_HUB_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=8
module load python/3.11 cuda/12.2 2>/dev/null || true
source $CD_VENV/bin/activate
