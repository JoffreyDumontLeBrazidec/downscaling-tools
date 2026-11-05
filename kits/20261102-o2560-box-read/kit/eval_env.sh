# o2560 box read: CPU scoring environment. downscaling-tools at main 4e659c3 (rule 32 v2 judge) cloned into the kit (dstools/).
set +e +u
source /e/software/default/configs/modules.sh 2>/dev/null
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3 git/2.45.1 >/dev/null 2>&1
set -eo pipefail
source /e/data1/jureap-data/ecmwf/users/jdumont/venvs/.ds/bin/activate
KIT=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read
DST=$KIT/dstools/downscaling-tools
DST_PIN=4e659c3
[[ "$(git -C $DST rev-parse --short=7 HEAD)" == "$DST_PIN" ]] || { echo "FATAL downscaling-tools not at $DST_PIN"; return 1; }
OUT=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102
export PYTHONPATH="$DST:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
export EVAL_ROOT=$OUT/_eval_root
mkdir -p "$EVAL_ROOT"
