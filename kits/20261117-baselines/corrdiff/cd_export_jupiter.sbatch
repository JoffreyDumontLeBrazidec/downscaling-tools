#!/bin/bash
# CorrDiff on Jupiter: export fill + stats + verify on one booster node (fallback if the login-node fill is cut;
# the smoke and training jobs run the same gate at their start). Submit after `cd_export.py init`.
#SBATCH --job-name=cd_export
#SBATCH --account=gkpdm
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/logs/%x_%j.out
set -uo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261120-corrdiff-jupiter/kit/cd_env_jupiter.sh
source $CD_KIT/cd_data_gate_jupiter.sh
cd_data_gate
