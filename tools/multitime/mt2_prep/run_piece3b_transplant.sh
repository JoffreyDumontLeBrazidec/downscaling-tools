#!/bin/bash
# Piece 3(b): transplant stage a into Vera's single-slot container, then load it through the branch's warm start and verify.
source /home/ecm5702/scratch/sandbox/20261016-multitime-e2/scripts/vera_env.sh
date -u
$PY $S/scripts/transplant_to_vera.py --ours $W/stage_a_do0_donor70.ckpt --container $W/stage_b0_vera_k1_init.ckpt \
  --out $W/stage_b_vera_k1_transplanted.ckpt || { echo rc=1; exit 1; }
export OUTPUT=$S/runs/build_k1_warm; mkdir -p $OUTPUT; cd $OUTPUT
$PY $S/scripts/build_vera_model.py --config-dir $S/configs --config-name mt2_e2_20k \
  --override 'task.input_offsets=[0h]' --override 'task.output_offsets=[0h]' \
  --warm-start $W/stage_b_vera_k1_transplanted.ckpt --verify --interface $W/stage_b_vera_k1_transplanted.interface.pt
echo rc=$?; date -u
