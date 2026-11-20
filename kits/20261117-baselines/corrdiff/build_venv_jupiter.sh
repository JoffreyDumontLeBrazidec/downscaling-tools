#!/bin/bash
# Build the CorrDiff venv on the Jupiter login node (uv, cache on /e/data1, python 3.11, torch cu128 aarch64).
set -uo pipefail
JCD=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter
V=$JCD/venv-physicsnemo-2.2.2
source $HOME/dev/runtimes/uv.env
echo "start $(date -u) $(uv --version)"
uv venv $V --python 3.11 || exit 1
uv pip install --python $V/bin/python "torch==2.10.*" "torchvision==0.25.*" --index-url https://download.pytorch.org/whl/cu128 || { echo TORCH_FAIL; exit 1; }
uv pip install --python $V/bin/python "nvidia-physicsnemo==2.2.2" -r $JCD/physicsnemo/examples/weather/corrdiff/requirements.txt pyyaml \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || { echo REST_FAIL; exit 1; }
$V/bin/python -c "import torch, physicsnemo, zarr, hydra; print('IMPORT_OK', physicsnemo.__version__, torch.__version__, torch.version.cuda, zarr.__version__)"
site=$($V/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
$V/bin/python ~/kernel_guard/kernel_guard.py install-hook "$site"; echo "hook_rc=$?"
env -u PYTHONPATH $V/bin/python ~/kernel_guard/kernel_guard.py check; echo "check_rc=$?"
echo "BUILD_DONE $(date -u)"
