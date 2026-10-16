#!/bin/bash
# Piece 3(c): tile stage b into the two-slot container (average mode), then load it through the branch's warm start and verify.
source /home/ecm5702/scratch/sandbox/20261016-multitime-e2/scripts/vera_env.sh
date -u
$PY $S/code/downscaling-tools/tools/multitime/tile_warm_start.py --single $W/stage_b_vera_k1_transplanted.ckpt \
  --container $W/stage_c0_vera_k2_init.ckpt --widths 68,11,70 --mode average --out $W/stage_c_vera_k2_tiled_average.ckpt || { echo rc=1; exit 1; }
export OUTPUT=$S/runs/build_k2_warm; mkdir -p $OUTPUT; cd $OUTPUT
$PY $S/scripts/build_vera_model.py --config-dir $S/configs --config-name mt2_e2_20k \
  --warm-start $W/stage_c_vera_k2_tiled_average.ckpt --verify --interface $W/stage_c_vera_k2_tiled_average.interface.pt
echo rc=$?; date -u
