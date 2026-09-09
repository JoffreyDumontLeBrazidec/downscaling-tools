#!/bin/bash --login
#SBATCH --job-name=sh2a_head
#SBATCH --partition=gpu
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
# Train the station head for one target variable on one GPU and score it against
# its three controls. The work itself is in train_score_body.sh, which the CPU
# variant train_score_cpu.sbatch runs as well.
#
#   sbatch --export=ALL,TARGET=2t,TAG=firstcut,RUN_PREFIX=firstcut_20260909 train_score.sbatch
bash /home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a/train_score_body.sh
