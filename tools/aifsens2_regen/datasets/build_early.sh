#!/bin/bash
# Build the three early-2026 (January to 12 May) AIFS ENS version 2 training stores.
#
# The chain is the usual one: init (three stores) -> load (a SLURM job array, several starts per
# array task) -> finalise + inspect + verify. It mirrors build_full.sh of the summer build
# (/home/ecm5702/hpcperm/sandbox/20260905-aifs-analysis-target/build_full.sh), with three
# differences: the fork is the NEW worktree, the destination is a plain Lustre directory on
# scratch (so no "lfs setstripe" pool command is needed and /ec/ai is not used), and the whole
# thing is parameterised so that a block subset (for example January alone, or the two-start
# fixture) can be built into a separately named store.
#
# Usage:
#   ./build_early.sh                                  # the full early period, real store names
#   ./build_early.sh --suffix -jan --start "2026-01-01 00:00:00" --end "2026-01-31 12:00:00" \
#                    --nparts 6
#   ./build_early.sh --suffix -test --start ... --end ... --grib-dir /path/to/derived_o320 \
#                    --dest /path/to/test/area --nparts 1 --dry-run
#
# Options:
#   --suffix S      append S to the three store names (use it for every subset build)
#   --start S       override the recipe start date (quoted, "YYYY-MM-DD HH:MM:SS")
#   --end S         override the recipe end date
#   --origin S      override fake_forecasts_origin (default: keep the recipe value)
#   --grib-dir D    override the directory holding the O320 GRIB files of the input store
#   --dest D        destination directory for the .zarr stores
#   --nparts N      number of load array tasks (default 27, about 10 starts each)
#   --concurrency N how many array tasks run at once (default 14)
#   --recipes R     space separated list of recipe stems to build (default: all three)
#   --members M     comma separated MARS ensemble members for the run copy (fixtures only)
#   --dry-run       write the sbatch files and the run recipes, but do not submit
#   --tag T         name of the run directory under runs/ (default: early<suffix>)
set -euo pipefail

S=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build
FORK=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/anemoi-datasets
VENV=/home/ecm5702/hpcperm/venvs/pristine-uv-x86_64
DEST=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/training
SUFFIX=""
START=""
END=""
ORIGIN=""
GRIBDIR=""
NPARTS=27
CONC=14
RECIPES="aifs_in_early an_target_early forcings_early"
MEMBERS=""
DRYRUN=0
TAGOPT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --suffix) SUFFIX="$2"; shift 2;;
    --suffix=*) SUFFIX="${1#*=}"; shift;;
    --members) MEMBERS="$2"; shift 2;;
    --members=*) MEMBERS="${1#*=}"; shift;;
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --origin) ORIGIN="$2"; shift 2;;
    --grib-dir) GRIBDIR="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --nparts) NPARTS="$2"; shift 2;;
    --concurrency) CONC="$2"; shift 2;;
    --recipes) RECIPES="$2"; shift 2;;
    --dry-run) DRYRUN=1; shift;;
    --tag) TAGOPT="$2"; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done

TAG="${TAGOPT:-early${SUFFIX}}"
RUN=$S/runs/$TAG
LOGS=$RUN/logs
CACHE=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/cache/$TAG
mkdir -p "$RUN/recipes" "$LOGS" "$DEST" "$CACHE" "$S/mir_cache"

# Build the run copies of the recipes, applying whatever overrides were asked for. With no
# option at all the copies are byte-identical to the recipes in $S/recipes.
for R in $RECIPES; do
  SRC=$S/recipes/$R.yaml
  DST=$RUN/recipes/$R.yaml
  cp "$SRC" "$DST"
  [ -n "$SUFFIX" ] && sed -i "s|^\(name: .*\)$|\1$SUFFIX|" "$DST"
  [ -n "$START" ]  && sed -i "s|^\(  start: \).*$|\1$START|" "$DST"
  [ -n "$END" ]    && sed -i "s|^\(  end: \).*$|\1$END|" "$DST"
  [ -n "$ORIGIN" ] && sed -i "s|^\(  fake_forecasts_origin: \).*$|\1$ORIGIN|" "$DST"
  [ -n "$GRIBDIR" ] && sed -i "s|^\(      path: \).*/\([^/]*\)$|\1${GRIBDIR}/\2|" "$DST"
  # --members restricts the MARS ensemble to the listed members in the run copy only. It is meant
  # for fixtures, so that a test build stays a small tape retrieval instead of fifty members.
  [ -n "$MEMBERS" ] && sed -i "s|^\\( *number: \\).*$|\\1[$MEMBERS]|" "$DST"
  echo "recipe -> $DST : $(grep -m1 '^name:' "$DST")"
done

# The store name is read out of the run recipe with sed, so that the generated sbatch files
# carry no quoting surprises.
NAME_CMD='sed -n "s/^name: //p"'

cat > "$RUN/init.sbatch" <<EOS
#!/bin/bash
#SBATCH --job-name=aifsv2-$TAG-init
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=$LOGS/%x_%j.out
set -uo pipefail
source $VENV/bin/activate
export PYTHONPATH=$FORK/src:\${PYTHONPATH:-}
python -c "import anemoi.datasets as d; print(d.__file__)" | grep -q 20260910-aifsens2-2026-early || { echo FATAL new fork worktree not resolved; exit 42; }
cd $DEST
for R in $RECIPES; do
  N=\$($NAME_CMD $RUN/recipes/\$R.yaml)
  [ -e \$N.zarr ] && { echo "FATAL: \$N.zarr already exists"; exit 43; }
  anemoi-datasets init $RUN/recipes/\$R.yaml \$N.zarr --overwrite || { echo "INIT FAILED \$R"; exit 1; }
  echo "INIT OK \$N.zarr"
done
echo EARLY_INIT_RC=0
EOS

cat > "$RUN/load.sbatch" <<EOS
#!/bin/bash
#SBATCH --job-name=aifsv2-$TAG-load
#SBATCH --qos=nf
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=1-$NPARTS%$CONC
#SBATCH --output=$LOGS/%x_%A_%a.out
set -uo pipefail
source $VENV/bin/activate
export PYTHONPATH=$FORK/src:\${PYTHONPATH:-}
python -c "import anemoi.datasets as d; print(d.__file__)" | grep -q 20260910-aifsens2-2026-early || { echo FATAL new fork worktree not resolved; exit 42; }
export MIR_COEFFICIENT_CACHE=$S/mir_cache
export MIR_LEGENDRE_LOADER=file-io
mkdir -p \$MIR_COEFFICIENT_CACHE
cd $DEST
for R in $RECIPES; do
  N=\$($NAME_CMD $RUN/recipes/\$R.yaml)
  echo "=== LOAD \$N part \$SLURM_ARRAY_TASK_ID/$NPARTS \$(date -u +%FT%TZ)"
  anemoi-datasets load \$N.zarr --part \$SLURM_ARRAY_TASK_ID/$NPARTS --cache $CACHE/part_\$SLURM_ARRAY_TASK_ID || { echo "LOAD FAILED \$R part \$SLURM_ARRAY_TASK_ID"; exit 1; }
done
mkdir -p $CACHE/_done_aside
mv $CACHE/part_\$SLURM_ARRAY_TASK_ID $CACHE/_done_aside/part_\${SLURM_ARRAY_TASK_ID}_\$(date +%s) 2>/dev/null
echo "EARLY_LOAD_PART_\${SLURM_ARRAY_TASK_ID}_RC=0"
EOS

cat > "$RUN/finalise.sbatch" <<EOS
#!/bin/bash
#SBATCH --job-name=aifsv2-$TAG-finalise
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=$LOGS/%x_%j.out
set -uo pipefail
source $VENV/bin/activate
export PYTHONPATH=$FORK/src:\${PYTHONPATH:-}
cd $DEST
for R in $RECIPES; do
  N=\$($NAME_CMD $RUN/recipes/\$R.yaml)
  anemoi-datasets finalise \$N.zarr || { echo "FINALISE FAILED \$R"; exit 1; }
  anemoi-datasets cleanup \$N.zarr || true
  echo "=== INSPECT \$N"
  anemoi-datasets inspect \$N.zarr | head -30
done
python $S/verify_early.py --dir $DEST --suffix="$SUFFIX"
echo "EARLY_VERIFY_RC=\$?"
EOS

if [ $DRYRUN -eq 1 ]; then
  echo "DRY RUN: sbatch files written under $RUN, nothing submitted"
  exit 0
fi
i=$(sbatch --parsable "$RUN/init.sbatch")
l=$(sbatch --parsable --dependency=afterok:$i "$RUN/load.sbatch")
f=$(sbatch --parsable --dependency=afterok:$l "$RUN/finalise.sbatch")
echo "SUBMITTED tag=$TAG init=$i load=$l finalise=$f  logs=$LOGS"
