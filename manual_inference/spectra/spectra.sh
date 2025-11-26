#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/outputs/spectra-%j.out


set -ex

source /home/ecm5702/dev/.ds-dyn/bin/activate
python /home/ecm5702/dev/downscaling-tools/manual_inference/spectra/spectra.py