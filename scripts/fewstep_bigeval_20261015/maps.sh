#!/bin/bash
# zoom_maps figures of the few-step big eval (pre-registered note, section 4.1a).
#
#   PRED_ROOT=... MAP_ROOT=... DT_ROOT=... scripts/fewstep_bigeval_20261015/maps.sh [--run] [--groups "1 2 4 59"]
#
# Default is a dry run that prints the commands; --run executes them (CPU, a few minutes per
# figure; on Atos run it inside an interactive or nf job with the env of score_row.sbatch).
#
# Boxes (member 1, the run and the reference on one colour scale per figure, 10 m wind and 2 m
# temperature, value and fine part):
#   alps   set A, 2022-08-19 + 60 h, --region alps (43-49 N, 4-16 E, drawn widened to 12 deg)
#   storm  set B, Idalia 2023-08-28 + 48 h, 12 x 12 deg centred on the truth's msl minimum inside
#          the Idalia box of the tc evaluator (eval.cli.judge.storm_centre on the reference row's
#          file, member 1; judge's STORM_HALF_DEG = 6)
# One figure group per call count (rows.yaml `group`: 1, 2, 4, 59), each with the reference row.
# The fine part uses --fine-cut-deg 2.0: the O96 input's analogue of the default 0.6 deg, which
# is set for an O320 input (2.0 = 0.6 x 320 / 96), so "fine" is the detail the O96 input cannot
# carry. Titles say O96 / O320 (zoom_maps' defaults name O1280).
#
# Output: $MAP_ROOT/<box>/calls<group>/  (PNGs named {key}_..._init<date>_n001_<tag>_f<step>.png
# plus the zoom_maps manifest).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PRED_ROOT:?PRED_ROOT is required}" "${MAP_ROOT:?MAP_ROOT is required}"
DT_ROOT="${DT_ROOT:-$(cd "$HERE/../.." && pwd)}"
ROWS_FILE="${ROWS_FILE:-$HERE/rows.yaml}"
PY="${PY:-python}"
REF="${REF:-edm_pw30c0}"
MEMBER="${MEMBER:-1}"
RUN=0; CALL_GROUPS="1 2 4 59"; BOXES="alps storm"
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1 ;;
    --groups) CALL_GROUPS="$2"; shift ;;
    --boxes) BOXES="$2"; shift ;;
    -h|--help) sed -n 2,24p "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
export PYTHONPATH="$DT_ROOT:$HERE"
cd "$DT_ROOT"

ALPS_DATE=20220819; ALPS_STEP=60
STORM_EVENT=idalia; STORM_DATE=20230828; STORM_STEP=48; STORM_HALF=6

# predictions directory of a row in a set
pdir() {
  "$PY" -c '
import sys, yaml
from pathlib import Path
import score_row as S
rows_file, label, s, root, psd = sys.argv[1:6]
row = S.load_row(Path(rows_file), label)
print(S.predictions_dir(row, s, Path(root), ("set" + s) if psd == "__default__" else psd))
' "$ROWS_FILE" "$1" "$2" "$PRED_ROOT" "${PRED_SET_DIR-__default__}"
}

# labels of one call-count group, reference first
group_rows() {
  "$PY" -c '
import sys, yaml
from pathlib import Path
rows = yaml.safe_load(Path(sys.argv[1]).read_text())["rows"]
ref, g = sys.argv[2], int(sys.argv[3])
print(" ".join([ref] + [r["label"] for r in rows if int(r.get("group", r["calls"])) == g and r["label"] != ref]))
' "$ROWS_FILE" "$REF" "$1"
}

run_cmd() {
  if [ "$RUN" = 1 ]; then "$@"; else printf '[dry-run]'; printf ' %q' "$@"; echo; fi
}

for box in $BOXES; do
  case "$box" in
    alps)
      set_=A; date=$ALPS_DATE; step=$ALPS_STEP; where=(--region alps) ;;
    storm)
      set_=B; date=$STORM_DATE; step=$STORM_STEP
      refdir="$(pdir "$REF" B)"
      centre="$("$PY" -c '
import sys
from pathlib import Path
from eval.cli.judge import storm_centre
c = storm_centre(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
print("" if c is None else f"{c[0]:.2f} {c[1]:.2f} {c[2]:.1f}")
' "$refdir" "$STORM_DATE" "$STORM_STEP" "$MEMBER" "$STORM_EVENT" 2>/dev/null || true)"
      if [ -z "$centre" ]; then
        echo "storm box: no truth msl minimum in $refdir/predictions_${STORM_DATE}_step$(printf %03d $STORM_STEP).nc;" \
             "storm figures skipped" >&2
        continue
      fi
      read -r clat clon cpmin <<< "$centre"
      echo "storm centre (truth, member $MEMBER): $clat N, $clon E, $cpmin hPa"
      ext=($("$PY" -c "import sys; a,o,h=map(float,sys.argv[1:4]); print(o-h, o+h, a-h, a+h)" "$clat" "$clon" "$STORM_HALF"))
      where=(--extent "${ext[@]}" --region-tag "storm_${STORM_EVENT}") ;;
    *) echo "unknown box $box" >&2; exit 2 ;;
  esac
  for g in $CALL_GROUPS; do
    runs=(); titles=(--title "eefo=Input (EEFO O96)" --title "enfo=Truth (o320)")
    for lab in $(group_rows "$g"); do
      d="$(pdir "$lab" "$set_")"
      if [ ! -f "$d/predictions_${date}_step$(printf %03d "$step").nc" ]; then
        echo "  $box calls$g: $lab has no predictions_${date}_step$(printf %03d "$step").nc in $d; left out" >&2
        continue
      fi
      runs+=(--run "$lab=$d"); titles+=(--title "$lab=$lab · O320")
    done
    [ "${#runs[@]}" -gt 0 ] || { echo "  $box calls$g: no row has the file; skipped" >&2; continue; }
    out="$MAP_ROOT/$box/calls$g"
    [ "$RUN" = 1 ] && mkdir -p "$out"
    for var in wind10m 2t; do
      for field in value fine; do
        run_cmd "$PY" -m eval.cli zoom_maps "${runs[@]}" "${titles[@]}" \
          --date "$date" --step "$step" --member "$MEMBER" "${where[@]}" \
          --variable "$var" --field "$field" --fine-cut-deg 2.0 --output-dir "$out"
      done
    done
  done
done
[ "$RUN" = 1 ] || echo "Dry run: add --run to draw."
