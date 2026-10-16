# sourced by the piece-3 step scripts: the Vera x86 interpreter on the sandbox code and the lane fork anemoi-datasets
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2
W=$S/checkpoints/warm
PY=/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/.venv-x86_64/bin/python
C=$S/code/anemoi-core
export PYTHONPATH=$C/models/src:$C/training/src:$C/graphs/src:/home/ecm5702/dev/anemoi-datasets/src
export HYDRA_FULL_ERROR=1 ANEMOI_BASE_SEED=756 ANEMOI_GRAPHS_FORCE_CPU=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8
