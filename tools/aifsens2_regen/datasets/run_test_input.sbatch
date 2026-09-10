#!/bin/bash
#SBATCH --job-name=aifsv2-early-test-input
#SBATCH --qos=nf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ecm5702/agent-work/20260910-aifsens2-2026-early/notes/test_input_%j.log
set -x
source /home/ecm5702/hpcperm/venvs/pristine-uv-x86_64/bin/activate
export PYTHONPATH=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/anemoi-datasets/src
python -c "import anemoi.datasets as d; print(d.__file__)"
D=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/validation/datasets_test/stores
R=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/runs/early-test/recipes
mkdir -p $D && cd $D
N=downscaling-ai-pf-enfo-rgn2-mars-o320-2026-2026-12h-6h-v1-aifsens2-early-test
anemoi-datasets init $R/aifs_in_early.yaml $N.zarr --overwrite; echo "INIT_RC=$?"
anemoi-datasets load $N.zarr --part 1/1; echo "LOAD_RC=$?"
anemoi-datasets finalise $N.zarr; echo "FINALISE_RC=$?"
anemoi-datasets inspect $N.zarr | head -40
echo TEST_INPUT_DONE
