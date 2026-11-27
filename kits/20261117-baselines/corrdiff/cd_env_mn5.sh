# CorrDiff Europe (E2 box) baseline environment on MareNostrum 5 (source it; login node or job). Code + venv on
# /gpfs/projects ($CD), the lat-lon export on /gpfs/scratch ($CDD). Configs config_training_e2v4m5_*.yaml in $CD/conf
# (make_cd_configs.py --name-suffix m5 --out-conf-dir $CD/conf), passed to train.py with --config-dir by
# cd_train_mn5_body.sh. Same layout as cd_env_jupiter_x.sh.
export CD=/gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5
export CDD=/gpfs/scratch/ehpc01/jdumontl/sandbox/20261127-corrdiff-mn5/data
export CD_VENV=$CD/venv                             # python 3.11 x86_64, torch 2.10.0 (cu128), nvidia-physicsnemo 2.2.2 (build_venv_mn5.sh)
export CD_REPO=$CD/physicsnemo                      # NVIDIA/physicsnemo tag v2.2.2 (072465a1)
export CD_KIT=$CD/kit                               # downscaling-tools branch claude/corrdiff-mn5-20261127 (SHA in $CD_KIT/KIT_SHA)
export CD_CONF_DIR=$CD/conf CD_CONF_SUFFIX=m5
export CD_DATA=$CDD/e2v4_box_0p0694.zarr
export E2V4=/gpfs/scratch/ehpc01/jdumontl/datasets/o320_o1280_europe_e2_v4
export E2=/gpfs/scratch/ehpc01/jdumontl/datasets/o320_o1280_europe_e2
export HF_HUB_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# MN5 ~/.bashrc loads PYTHON modules (PYTHONHOME / PYTHONPATH) and an old venv: neutralise them (as ~/dev/runtimes/uv.env)
unset PYTHONHOME PYTHONPATH
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)"; unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
fi
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME=$CD/cache WANDB_DIR=$CD/cache/wandb NUMBA_CACHE_DIR=$CD/cache/numba MPLCONFIGDIR=$CD/cache/mpl \
       TRITON_CACHE_DIR=$CD/cache/triton TORCHINDUCTOR_CACHE_DIR=$CD/cache/inductor
# TMPDIR short and node-local: AF_UNIX socket paths are limited to 108 chars (Jupiter smoke 3023869, 20 Nov)
export TMPDIR=/tmp/${USER}_cd_${SLURM_JOB_ID:-x}
mkdir -p $XDG_CACHE_HOME $WANDB_DIR $NUMBA_CACHE_DIR $MPLCONFIGDIR $TMPDIR 2>/dev/null
source $CD_VENV/bin/activate
# GitPython: no git binary guaranteed on compute nodes (Jupiter reg seg1 3023870 "Bad git executable", 20 Nov)
export GIT_PYTHON_REFRESH=quiet
