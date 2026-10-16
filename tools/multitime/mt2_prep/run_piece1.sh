#!/bin/bash
# Piece 1: Vera's datamodule on the E2 stores through the lane fork anemoi-datasets (cheap path, no port).
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2
PY=/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/.venv-x86_64/bin/python
C=$S/code/anemoi-core
export PYTHONPATH=$C/models/src:$C/training/src:$C/graphs/src:/home/ecm5702/dev/anemoi-datasets/src
export OUTPUT=$S/runs/check_datamodule HYDRA_FULL_ERROR=1 ANEMOI_BASE_SEED=756
mkdir -p $OUTPUT
cd $OUTPUT
date -u
$PY $S/scripts/check_mt2_datamodule.py --config-dir $S/configs --config-name mt2_e2_20k --json $S/logs/piece1_datamodule.json
echo rc=$?
date -u
