#!/bin/bash
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2
PY=/home/ecm5702/hpcperm/sandbox/20260917-vera-spatial-downscaler/.venv-x86_64/bin/python
C=$S/code/anemoi-core
export PYTHONPATH=$C/models/src:$C/training/src:$C/graphs/src
date -u
$PY $S/scripts/make_vera_graph.py --stem /home/ecm5702/hpcperm/lanes/o320_o1280_europe_e2/graph_m600_ringmask_knn13/europe_e2_m600rmknn13_graph --out $S/assets/graph/europe_e2_m600rmknn13_vera_fused.pt
echo rc=$?
date -u
