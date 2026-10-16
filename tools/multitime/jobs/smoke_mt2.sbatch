#!/bin/bash
# Atos AC, mt2_e2_20k SMOKE (2026-10-16): Vera Gahlen's branch (fork anemoi-core claude/project-thread-n1zaq3,
# 4a81ecb43), two time slots 12 h apart, E2-v4 stores, warm checkpoint checkpoints/warm/stage_c_vera_k2_tiled_average.ckpt
# loaded weights only. 150 steps, 20 training batches per epoch (validation every 20 steps on the config's 144
# batches), one checkpoint at step 150. 2 nodes x 4 A100-40GB, batch 1 per GPU = global batch 8. Reports peak GPU
# memory (torch.cuda.max_memory_allocated per rank) and seconds per step ([peakmem], [timing] lines).
# Body: jobs/mt2_job.sh. Preview without training: DRY_RUN=1 bash <this file>
#SBATCH --job-name=smoke_mt2
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/ecm5702/scratch/sandbox/20261016-multitime-e2/logs/%x_%j.out
#SBATCH --error=/home/ecm5702/scratch/sandbox/20261016-multitime-e2/logs/%x_%j.err
export MT_KIND=smoke MT_TAG=smoke MT_END_STEP=150 MT_RUN_NAME=smoke_mt2_e2
export MT_EXTRA="dataloader.limit_batches.training=20 diagnostics.checkpoint.every_n_train_steps.save_frequency=150"
exec bash /home/ecm5702/scratch/sandbox/20261016-multitime-e2/jobs/mt2_job.sh
