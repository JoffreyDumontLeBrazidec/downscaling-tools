#!/bin/bash
# Piece 3(b) step 1: Vera's single-slot model for the E2 layout, freshly initialised (seed 756): the container for the
# transplant and the randomly initialised contrast of the forward check.
source /home/ecm5702/scratch/sandbox/20261016-multitime-e2/scripts/vera_env.sh
export OUTPUT=$S/runs/build_k1_init; mkdir -p $OUTPUT; cd $OUTPUT; date -u
$PY $S/scripts/build_vera_model.py --config-dir $S/configs --config-name mt2_e2_20k \
  --override 'task.input_offsets=[0h]' --override 'task.output_offsets=[0h]' \
  --container $W/stage_b0_vera_k1_init.ckpt --interface $W/stage_b0_vera_k1_init.interface.pt --keys $W/stage_b0_vera_k1_init.keys.json
echo rc=$?; date -u
