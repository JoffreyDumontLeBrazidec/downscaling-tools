#!/bin/bash
# Queue-wait probe (28 Nov 2026): same shape as a training segment (2 node(s) x 4 GPUs, 24 h requested), runs 30 s.
# Read: sacct -n -X --name=qwait -S 2026-11-28 -o JobID,JobName%20,NNodes,Timelimit,Submit,Start,End -P
#SBATCH --job-name=qwait_n2_t24
#SBATCH --account=DestE_340_26
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
date -u; hostname; sleep 30; date -u
