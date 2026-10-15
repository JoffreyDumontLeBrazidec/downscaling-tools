#!/bin/bash
# Submit the scoring jobs of every row of rows.yaml on both case sets (Atos, QOS nf).
#
#   PRED_ROOT=... EVAL_ROOT=... DT_ROOT=... scripts/fewstep_bigeval_20261015/submit_all.sh            # dry run
#   PRED_ROOT=... EVAL_ROOT=... DT_ROOT=... scripts/fewstep_bigeval_20261015/submit_all.sh --submit   # submit
#
# Order: the reference row (edm_pw30c0) is submitted first, one job per set; every other job of a
# set starts after the reference job of that set has ended (afterany), because the reference job
# builds the shared inputs under $EVAL_ROOT/_cache (O96<->O320 matrices, texture kNN, shape box
# caches) and two jobs must not build them at once. One job per row and set: 2 x 20 = 40 jobs.
#
# Options: --submit (default is a dry run that prints the sbatch lines), --sets "A B",
#          --rows "label1 label2" (a subset; the reference row is always included, and its job
#          only skips evaluators already .complete), --reference LABEL (default edm_pw30c0).
# Other environment variables are passed to score_row.sbatch (PRED_SET_DIR, ONLY, SKIP,
# ECMWF_SPECTRA, PY, ROWS_FILE). Rows whose predictions directory is missing are listed and
# not submitted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PRED_ROOT:?PRED_ROOT is required}" "${EVAL_ROOT:?EVAL_ROOT is required}" "${DT_ROOT:?DT_ROOT is required}"
ROWS_FILE="${ROWS_FILE:-$HERE/rows.yaml}"
SUBMIT=0; SETS="A B"; ONLY_ROWS=""; REF=edm_pw30c0
while [ $# -gt 0 ]; do
  case "$1" in
    --submit) SUBMIT=1 ;;
    --sets) SETS="$2"; shift ;;
    --rows) ONLY_ROWS="$2"; shift ;;
    --reference) REF="$2"; shift ;;
    -h|--help) sed -n 2,18p "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
PY="${PY:-python}"
SBATCH_FILE="$DT_ROOT/scripts/fewstep_bigeval_20261015/score_row.sbatch"

# "label set predictions_dir n_files" for every row and set, reference first
plan="$(PYTHONPATH="$HERE" "$PY" -c '
import sys, yaml
from pathlib import Path
import score_row as S
rows_file, pred_root, ref, psd = sys.argv[1:5]
sets = sys.argv[5:]
data = yaml.safe_load(Path(rows_file).read_text())
items = data["rows"] if isinstance(data, dict) else data
items = sorted(items, key=lambda r: r["label"] != ref)
for r in items:
    for s in sets:
        if s not in [str(x) for x in r.get("sets", [])]:
            continue
        d = S.predictions_dir(r, s, Path(pred_root), ("set" + s) if psd == "__default__" else psd)
        print(r["label"], s, d, len(S.prediction_files(d)))
' "$ROWS_FILE" "$PRED_ROOT" "$REF" "${PRED_SET_DIR-__default__}" $SETS)"

declare -A REF_JOB WARNED
missing=()
n=0
while read -r label set dir nfiles; do
  [ -z "$label" ] && continue
  if [ -n "$ONLY_ROWS" ] && [[ " $ONLY_ROWS $REF " != *" $label "* ]]; then continue; fi
  if [ "$nfiles" = 0 ]; then missing+=("$label/$set: $dir"); continue; fi
  dep=()
  if [ "$label" != "$REF" ] && [ -n "${REF_JOB[$set]:-}" ]; then
    dep=(--dependency="afterany:${REF_JOB[$set]}")
  elif [ "$label" != "$REF" ] && [ -z "${WARNED[$set]:-}" ]; then
    echo "WARNING: no reference job for set $set in this submission; its rows start at once" \
         "(fine when $EVAL_ROOT/_cache is already built)" >&2
    WARNED[$set]=1
  fi
  cmd=(sbatch --parsable --job-name="bigeval-$set-$label" "${dep[@]}"
       --export="ALL,ROW=$label,SET=$set,PRED_ROOT=$PRED_ROOT,EVAL_ROOT=$EVAL_ROOT,DT_ROOT=$DT_ROOT,ROWS_FILE=$ROWS_FILE"
       "$SBATCH_FILE")
  if [ "$SUBMIT" = 1 ]; then
    jid="$("${cmd[@]}")"
    echo "submitted $jid  $label $set ($nfiles files)"
  else
    jid="DRY_${set}_${label}"
    printf '[dry-run]'; printf ' %q' "${cmd[@]}"; echo "   # $nfiles files"
  fi
  [ "$label" = "$REF" ] && REF_JOB[$set]="$jid"
  n=$((n + 1))
done <<< "$plan"

if [ "$SUBMIT" = 1 ]; then echo "$n job(s) submitted"; else echo "$n job(s) planned"; fi
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Not submitted, no prediction files (check rows.yaml arm/step/sampler, VERIFY):"
  printf '  %s\n' "${missing[@]}"
fi
[ "$SUBMIT" = 1 ] || echo "Dry run: add --submit to submit."
