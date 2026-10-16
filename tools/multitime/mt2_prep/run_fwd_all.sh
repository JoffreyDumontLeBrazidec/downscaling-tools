#!/bin/bash
# Piece 3 forwards on the login-node CPU, one at a time (float32, pyg attention, 16 inference chunks on both codes).
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2; W=$S/checkpoints/warm; R=$S/scripts/run_fwd.sh
$R ours $W/stage_a_do0_donor70.interface.pt a_ours_do0_donor70_cpu ours > $S/logs/piece3a_fwd_ours_cpu.log 2>&1
$R vera $W/stage_b_vera_k1_transplanted.interface.pt b_vera_k1_transplanted_cpu > $S/logs/piece3b_fwd_vera_cpu.log 2>&1
$R vera $W/stage_b0_vera_k1_init.interface.pt b0_vera_k1_init_contrast_cpu > $S/logs/piece3b_fwd_contrast_cpu.log 2>&1
$R vera2 $W/stage_c_vera_k2_tiled_average.interface.pt c_vera_k2_tiled_average_cpu > $S/logs/piece3c_fwd_vera2_cpu.log 2>&1
source $S/scripts/vera_env.sh; cd /tmp
$PY $S/scripts/warm_start_fwdcheck.py compare --ref $W/fwd_a_ours_do0_donor70_cpu.pt --other $W/fwd_b_vera_k1_transplanted_cpu.pt \
  --other $W/fwd_b0_vera_k1_init_contrast_cpu.pt --other $W/fwd_c_vera_k2_tiled_average_cpu.pt --json $W/fwdcheck_summary_cpu.json
echo compare_rc=$?
