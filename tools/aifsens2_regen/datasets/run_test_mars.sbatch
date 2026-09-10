#!/bin/bash
#SBATCH --job-name=aifsv2-early-test-mars
#SBATCH --qos=nf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ecm5702/agent-work/20260910-aifsens2-2026-early/notes/test_mars_%j.log
set -x
source /home/ecm5702/hpcperm/venvs/pristine-uv-x86_64/bin/activate
export PYTHONPATH=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/anemoi-datasets/src
export MIR_COEFFICIENT_CACHE=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/mir_cache
export MIR_LEGENDRE_LOADER=file-io
mkdir -p $MIR_COEFFICIENT_CACHE
D=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/validation/datasets_test/stores
R=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/runs/early-test/recipes
cd $D
for REC in an_target_early forcings_early; do
  N=$(sed -n "s/^name: //p" $R/$REC.yaml)
  anemoi-datasets init $R/$REC.yaml $N.zarr --overwrite; echo "INIT_RC_$REC=$?"
  anemoi-datasets load $N.zarr --part 1/1; echo "LOAD_RC_$REC=$?"
  anemoi-datasets finalise $N.zarr; echo "FINALISE_RC_$REC=$?"
  anemoi-datasets inspect $N.zarr | head -25
done
echo TEST_MARS_DONE
