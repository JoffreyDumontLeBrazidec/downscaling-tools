#!/bin/bash
# mt2_e2_20k warm start, piece 3 forward checks on ONE A100 (2026-10-16). Runs the denoiser at sigma = 1 on the saved fixed
# input checkpoints/warm/fwd_input_seed20261016.pt for: (a) our materialised donor (do0 venv), (b) Vera's single-slot network
# with the transplanted weights, the randomly initialised single-slot network (contrast), (c) Vera's two-slot network with
# the tiled weights and the same state in both slots; then compares (b), contrast and (c) with (a). float32, TF32 off.
# Nothing is trained. (The login-node CPU run was stopped by the login node's memory limit.)
#SBATCH --job-name=mt2_warm_fwdcheck
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:59:00
#SBATCH --output=/home/ecm5702/scratch/sandbox/20261016-multitime-e2/logs/%x_%j.out
#SBATCH --error=/home/ecm5702/scratch/sandbox/20261016-multitime-e2/logs/%x_%j.out
set -uo pipefail
ulimit -c 0
S=/home/ecm5702/scratch/sandbox/20261016-multitime-e2; W=$S/checkpoints/warm
F=$S/scripts/warm_start_fwdcheck.py; IN=$W/fwd_input_seed20261016.pt
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TRITON_CACHE_DIR=/tmp/$USER/triton_${SLURM_JOB_ID:-local} OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date -u
( source /ec/res4/scratch/ecm5702/sandbox/20261007-pl25-ac/activate.sh || exit 3
  python $F run --side ours --interface $W/stage_a_do0_donor70.interface.pt --input $IN --out $W/fwd_a_ours_do0_donor70.pt --device cuda ); echo "OURS_RC=$?"
( source $S/scripts/vera_env.sh
  $PY $F run --side vera --interface $W/stage_b_vera_k1_transplanted.interface.pt --input $IN --out $W/fwd_b_vera_k1_transplanted.pt --device cuda; echo "B_RC=$?"
  $PY $F run --side vera --interface $W/stage_b0_vera_k1_init.interface.pt --input $IN --out $W/fwd_b0_vera_k1_init_contrast.pt --device cuda; echo "B0_RC=$?"
  $PY $F run --side vera2 --interface $W/stage_c_vera_k2_tiled_average.interface.pt --input $IN --out $W/fwd_c_vera_k2_tiled_average.pt --device cuda; echo "C_RC=$?"
  $PY $F compare --ref $W/fwd_a_ours_do0_donor70.pt --other $W/fwd_b_vera_k1_transplanted.pt --other $W/fwd_b0_vera_k1_init_contrast.pt \
     --other $W/fwd_c_vera_k2_tiled_average.pt --json $W/fwdcheck_summary.json; echo "COMPARE_RC=$?" )
date -u
echo FWDCHECK_DONE
