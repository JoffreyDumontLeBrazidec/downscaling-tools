#!/bin/bash
# Atos AC, mt2_e2_20k (2026-10-16): multi-in multi-out diffusion downscaler, two time slots 12 h apart, o320 -> o1280
# on the E2-v4 box, Vera Gahlen's branch (fork anemoi-core claude/project-thread-n1zaq3, 4a81ecb43), 20,000 steps.
# Chain method (downscaling-tools tools/chain): the SAME file for every segment; the start is read from the chain root
# /home/ecm5702/scratch/sandbox/20261016-multitime-e2/checkpoints/mt2_e2_20k (CHAIN.json and the run's last.ckpt). The
# first segment loads checkpoints/warm/stage_c_vera_k2_tiled_average.ckpt weights only; later ones resume from disk.
# Each segment queues its successor (chain_next, afterany). Declare the chain once before the first submission:
#   python3 /home/ecm5702/scratch/sandbox/20261016-multitime-e2/code/downscaling-tools/tools/chain/chain_state.py init \
#       --root /home/ecm5702/scratch/sandbox/20261016-multitime-e2/checkpoints/mt2_e2_20k --note "mt2_e2_20k from stage_c"
# 2 nodes x 4 A100-40GB, batch 1 per GPU = global batch 8. Body: jobs/mt2_job.sh.
# Preview without training: DRY_RUN=1 bash <this file>
#SBATCH --job-name=mt2_e2_20k
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/home/ecm5702/scratch/sandbox/20261016-multitime-e2/logs/%x_%j.out
#SBATCH --error=/home/ecm5702/scratch/sandbox/20261016-multitime-e2/logs/%x_%j.err
export MT_KIND=chain MT_TAG=mt2_e2_20k MT_END_STEP=20000 MT_RUN_NAME=mt2_e2_20k
exec bash /home/ecm5702/scratch/sandbox/20261016-multitime-e2/jobs/mt2_job.sh
