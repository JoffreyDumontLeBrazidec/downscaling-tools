#!/bin/bash
#SBATCH --job-name=aifsv2-early-test-verify
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ecm5702/agent-work/20260910-aifsens2-2026-early/notes/test_verify_%j.log
set -x
source /home/ecm5702/hpcperm/venvs/pristine-uv-x86_64/bin/activate
export PYTHONPATH=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/anemoi-datasets/src
S=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build
T=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/validation/datasets_test
echo "########## VERIFY_EARLY ##########"
python $S/verify_early.py --dir $T/stores --suffix=-test --expect-dates 4 \
    --grib-dir $T/derived_o320 --nan-sample 4 --allow-identical-members --no-boundary
echo "VERIFY_EARLY_RC=$?"
echo "########## COMBINED_VIEW (combined, tiny early store) ##########"
python $S/test_combined_view.py --mode combined --early-dir $T/stores --early-suffix=-test \
    --expect-early 4 --expect-starts 225 --no-boundary
echo "COMBINED_RC=$?"
echo "########## COMBINED_VIEW (summer only) ##########"
python $S/test_combined_view.py --mode summer-only
echo "SUMMER_ONLY_RC=$?"
echo TEST_VERIFY_DONE
