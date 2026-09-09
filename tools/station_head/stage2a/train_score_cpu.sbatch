#!/bin/bash --login
#SBATCH --job-name=sh2a_head_cpu
#SBATCH --qos=nf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ecm5702/agent-work/20260909-station-head-adapter/logs/%x_%j.out
# The same station-head run as train_score.sbatch, on the nf queue and without a
# GPU. The head is small enough that this is only a few times slower, and it is
# the way to run while the inference arrays hold every GPU node slot the account
# is allowed.
#
#   sbatch --export=ALL,TARGET=2t,TAG=firstcut,RUN_PREFIX=firstcut_20260909 train_score_cpu.sbatch
bash /home/ecm5702/dev/downscaling-tools-station-head/tools/station_head/stage2a/train_score_body.sh
