#!/bin/bash
# Extremes set XSET-1: define one model row on Jupiter = one export of one model. Written once; a row never changes.
# Usage: bash xset_define_row.sh <ROW> <MODEL> <STEP> [EMA_TAG]
#   ROW      new row name (letters, digits, underscore), e.g. FS242_freeze
#   MODEL    a model file jupiter/models/<MODEL>.env (run directory, config, code tree and pins)
#   STEP     training step of the export
#   EMA_TAG  which weights are read [ema2000]
# Optional environment: EMA_FILE and RAW_FILE (the two export files, when they are not under RUN_DIR with the usual names),
# XSET_SCHED_JSON and XSET_SAMPLER_JSON (a sampler other than the 30-step standard; recorded in every file).
# Writes XOUT/rowdefs/<ROW>.env with the files, the sha256 and the modification time of the weights. Defining the same
# row again with the same export is a no-op; with another export it is refused (take a new row name).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
row=${1:?row}; model=${2:?model}; step=${3:?step}; tag=${4:-ema2000}
[[ $row =~ ^[A-Za-z0-9_]+$ && $model =~ ^[A-Za-z0-9_]+$ && $step =~ ^[0-9]+$ && $tag =~ ^[A-Za-z0-9_]+$ ]] || { echo "FATAL bad argument"; exit 2; }
CK=/e/data1/jureap-data/ecmwf/users/jdumont/outputs
DEVR=/e/home/jusers/dumontlebrazidec1/jupiter/dev
XOUT=${XSET_JOUT:-$CK/xset1_20261116}
case "$row" in GL512_500k|GL512_450k|GL512_237k|REG_GL512_30k|REG_GL512_20k) echo "ROW_DEFINED $row (first row, fixed in xset_runs.sh)"; exit 0 ;; esac
[[ $row != ENS9 && $row != INPUT && $row != CF9 ]] || { echo "FATAL '$row' is a reference row"; exit 2; }
[[ -f $HERE/models/$model.env ]] || { echo "FATAL no model file $HERE/models/$model.env (copy models/TEMPLATE.env)"; exit 3; }
source $HERE/models/$model.env
: "${RUN_DIR:?model file: RUN_DIR}"
S6=$(printf %06d "$step")
pick() {  # what, explicit file, pattern
  if [[ -n "$2" ]]; then [[ -f "$2" ]] || { echo "FATAL $1 file $2 not found" >&2; exit 4; }; echo "$2"; return; fi
  local n; n=$(ls $3 2>/dev/null | wc -l)
  [[ $n == 1 ]] || { echo "FATAL $n $1 files match $3 (want 1; give the file explicitly)" >&2; exit 4; }
  ls $3
}
ema=$(pick weights "${EMA_FILE:-}" "$RUN_DIR/*/inference-${tag}-anemoi-by_step-epoch_*-step_$S6.ckpt")
raw=$(pick "raw by-step" "${RAW_FILE:-}" "$RUN_DIR/*/anemoi-by_step-epoch_*-step_$S6.ckpt")
sha=$(sha256sum "$ema" | cut -d' ' -f1)
new=$(mktemp)
{
  echo "MODEL=$model"; echo "CKPT_STEP=$step"; echo "EMA_TAG=$tag"
  printf 'EMA_FILE=%q\nRAW_FILE=%q\n' "$ema" "$raw"
  echo "EMA_SHA256=$sha"
  [[ -n "${XSET_SCHED_JSON:-}" ]] && printf 'XSET_SCHED_JSON=%q\n' "$XSET_SCHED_JSON"
  [[ -n "${XSET_SAMPLER_JSON:-}" ]] && printf 'XSET_SAMPLER_JSON=%q\n' "$XSET_SAMPLER_JSON"
  true
} > "$new"
mkdir -p $XOUT/rowdefs
def=$XOUT/rowdefs/$row.env
if [[ -f $def ]]; then
  if diff -q <(grep -v '^#' "$def") "$new" >/dev/null; then rm -f "$new"; echo "ROW_DEFINED $row (unchanged) $def"; exit 0; fi
  echo "FATAL row '$row' is already defined with another export:"; diff <(grep -v '^#' "$def") "$new" || true; rm -f "$new"; exit 5
fi
{ cat "$new"; echo "# defined $(date -u +%FT%TZ) kit $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown); weights modified $(date -u -r "$ema" +%FT%TZ), $(stat -c %s "$ema") bytes"; } > "$def"
rm -f "$new"
echo "ROW_DEFINED $row model=$model step=$step $tag sha256=${sha:0:16} $def"
