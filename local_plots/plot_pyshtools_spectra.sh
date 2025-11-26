#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/outputs/pyshtools_spectra/pyshtools_spectra%j.out

set -ex

source /home/ecm5702/dev/.ds-dyn/bin/activate


WORKDIR=/home/ecm5702/dev
cd $WORKDIR


# Run inference for each experiment
python /home/ecm5702/dev/downscaling-tools/local_plots/plot_pyshtools_spectra.py