# CorrDiff baseline environment on Jupiter for the second kit copy $CD/kit_20261123 (source it). Same as cd_env_jupiter.sh
# except CD_KIT and the configs: config_training_e2v4x_*.yaml from $CD_KIT/conf (make_cd_configs.py --name-suffix x
# --out-conf-dir $CD_KIT/conf), passed to train.py with --config-dir by cd_train_jupiter_body.sh.
export CD=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter
export CD_VENV=$CD/venv-physicsnemo-2.2.2           # python 3.11 aarch64, torch 2.10.0+cu128, nvidia-physicsnemo 2.2.2
export CD_REPO=$CD/physicsnemo                      # NVIDIA/physicsnemo tag v2.2.2 (072465a1)
export CD_KIT=$CD/kit_20261123                      # downscaling-tools branch claude/corrdiff-lowcompute-20261123
export CD_CONF_DIR=$CD_KIT/conf CD_CONF_SUFFIX=x
export CD_DATA=$CD/data/e2v4_box_0p0694.zarr
export E2V4=/e/data1/jureap-data/ecmwf/users/jdumont/datasets/o320_o1280_europe_e2_v4
export E2=/e/data1/jureap-data/ecmwf/users/jdumont/datasets/o320_o1280_europe_e2
export HF_HUB_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=8
# keep every cache off HOME (80k file limit)
export XDG_CACHE_HOME=$CD/cache WANDB_DIR=$CD/cache/wandb NUMBA_CACHE_DIR=$CD/cache/numba MPLCONFIGDIR=$CD/cache/mpl \
       TRITON_CACHE_DIR=$CD/cache/triton TORCHINDUCTOR_CACHE_DIR=$CD/cache/inductor
# TMPDIR short and node-local: AF_UNIX socket paths are limited to 108 chars; $CD/tmp killed smoke 3023869 (20 Nov)
export TMPDIR=/tmp/${USER}_cd_${SLURM_JOB_ID:-x}
mkdir -p $XDG_CACHE_HOME $WANDB_DIR $NUMBA_CACHE_DIR $MPLCONFIGDIR $TMPDIR 2>/dev/null
source $CD_VENV/bin/activate
# GitPython: no git binary on compute nodes; reg seg1 3023870 died "ImportError: Bad git executable" (20 Nov 19:02 UTC)
export GIT_PYTHON_REFRESH=quiet
