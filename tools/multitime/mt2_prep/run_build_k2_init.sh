#!/bin/bash
# Piece 3(c) step 1: Vera's two-slot model from the final mt2 config, freshly initialised (seed 756): the container for tiling.
source /home/ecm5702/scratch/sandbox/20261016-multitime-e2/scripts/vera_env.sh
export OUTPUT=$S/runs/build_k2_init; mkdir -p $OUTPUT; cd $OUTPUT; date -u
$PY $S/scripts/build_vera_model.py --config-dir $S/configs --config-name mt2_e2_20k \
  --container $W/stage_c0_vera_k2_init.ckpt --keys $W/stage_c0_vera_k2_init.keys.json
echo rc=$?; date -u
