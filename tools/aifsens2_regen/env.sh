#!/bin/bash
# Common environment for every stage of the AIFS ENS version 2 regeneration.
#
# Sourced by each sbatch template.  It provides the eccodes and mir command
# line tools from the ecmwf-toolbox module, and the pinned Python environment
# whose recipe is committed next to this file in env/.
#
# The same interpreter is used for the CPU stages and for the GPU stage.  The
# CPU stages only need the standard library and the eccodes command line tools,
# but using one interpreter everywhere removes a class of "works in one job,
# fails in another" problems.
#
# AIFSENS2_TOOLS must point at the directory that CONTAINS the aifsens2_regen
# package, not at the package itself, because every stage is invoked as
# "python -m aifsens2_regen.<stage>".  Running it that way also keeps the
# package directory off sys.path, which matters here: this package contains a
# module named calendar.py, and if the package directory were on sys.path it
# would shadow the standard library's calendar module for every library that
# imports it.

set -uo pipefail

module load ecmwf-toolbox 2>/dev/null

export AIFSENS2_VENV="${AIFSENS2_VENV:-/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/regen-env/.venv-x86_64}"
export AIFSENS2_TOOLS="${AIFSENS2_TOOLS:-/home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools}"
export AIFSENS2_ROOT="${AIFSENS2_ROOT:-/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910}"

source "$AIFSENS2_VENV/bin/activate"
cd "$AIFSENS2_TOOLS"

echo "host          : $(hostname)"
echo "slurm job     : ${SLURM_JOB_ID:-none} ${SLURM_ARRAY_TASK_ID:+task $SLURM_ARRAY_TASK_ID}"
echo "python        : $(python -V 2>&1)"
echo "tools         : $AIFSENS2_TOOLS"
echo "data root     : $AIFSENS2_ROOT"
