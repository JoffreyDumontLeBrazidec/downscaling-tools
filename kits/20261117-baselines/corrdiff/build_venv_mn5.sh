#!/bin/bash
# Build the CorrDiff venv on the MN5 login node (alogin1: pypi reachable, compute nodes have no internet; login processes
# are killed past 300 CPU-s, uv stays well below). Python 3.11, torch 2.10.0 (pypi x86_64 build = CUDA 12.8; MN5 driver
# 595 / CUDA 13.2 runs it), nvidia-physicsnemo 2.2.2, the v2.2.2 corrdiff example requirements, every version pinned to
# the Leonardo venv (venv-physicsnemo-2.2.2, freeze of 27 Nov = LEONARDO_FREEZE). Then the git_hash patch
# (cd_patch_version_check.py) and an import check.
#   bash build_venv_mn5.sh            (CDM from cd_env_mn5.sh; writes $CDM/venv, cache $CDM/.uv-cache)
set -uo pipefail
CDM=${CDM:-/gpfs/projects/ehpc01/ecm800825/sandbox/20261127-corrdiff-mn5}
V=$CDM/venv; KIT=${CD_KIT:-$CDM/kit}
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV   # MN5 ~/.bashrc PYTHON modules (as ~/dev/runtimes/uv.env)
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh || exit 1; }
export UV_CACHE_DIR=$CDM/.uv-cache UV_PYTHON_INSTALL_DIR=$CDM/.uv-python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=0
echo "start $(date -u) $(uv --version) host $(hostname)"
[[ -x $V/bin/python ]] || uv venv $V --python 3.11 || { echo VENV_FAIL; exit 1; }
FREEZE=$KIT/LEONARDO_FREEZE
uv pip install --python $V/bin/python -c $FREEZE "torch==2.10.0" "torchvision==0.25.0" "nvidia-physicsnemo==2.2.2" pyyaml \
  -r $CDM/physicsnemo/examples/weather/corrdiff/requirements.txt || { echo PIP_FAIL; exit 1; }
$V/bin/python $KIT/cd_patch_version_check.py || { echo PATCH_FAIL; exit 1; }
GIT_PYTHON_REFRESH=quiet $V/bin/python -c "import torch, physicsnemo, zarr, hydra, numba; from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet; print('IMPORT_OK', physicsnemo.__version__, torch.__version__, 'cuda', torch.version.cuda, 'zarr', zarr.__version__, 'numba', numba.__version__)" || { echo IMPORT_FAIL; exit 1; }
uv pip freeze --python $V/bin/python > $CDM/state/venv_freeze.txt
echo "BUILD_DONE $(date -u)"
