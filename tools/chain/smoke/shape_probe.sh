#!/bin/bash
# Which job shape would start first RIGHT NOW? Read-only: `sbatch --test-only` asks the scheduler for the expected
# start of a job without submitting it. One line per shape (nodes x walltime [x QOS]). Use it before choosing the
# shape of a chain, a pilot pool or a sweep (review thread "Review the overlap chains for MN5 and Leonardo").
#   shape_probe.sh <job.sbatch> "<nodes list>" "<walltime list>" ["<qos list>"]
#   shape_probe.sh muon_chain.sbatch "1 2" "06:00:00 12:00:00 24:00:00 72:00:00"
#   shape_probe.sh useq.sbatch "1 2 3" "24:00:00 4-00:00:00" "normal boost_qos_lprod"
# The estimate is for a job with zero age and today's queue; it is pessimistic in absolute terms. Read the ORDER
# and the gaps between shapes, and repeat on 2-3 different days before changing a default.
set -u
JOB="${1:?job file}"; NODES="${2:-1 2}"; WALLS="${3:-06:00:00 24:00:00}"; QOSES="${4:-}"
now="$(date +%s)"
printf '%-8s %-12s %-18s %-20s %s\n' nodes walltime qos est_start wait_h
for q in ${QOSES:-_}; do for n in $NODES; do for w in $WALLS; do
  args=(--test-only -N "$n" -t "$w"); [[ "$q" != _ ]] && args+=(--qos "$q")
  out="$(sbatch "${args[@]}" "$JOB" 2>&1)"
  st="$(sed -n 's/.*to start at \([0-9T:-]*\).*/\1/p' <<< "$out" | head -n 1)"
  if [[ -n "$st" ]]; then
    wait_h="$(awk -v a="$(date -d "$st" +%s)" -v b="$now" 'BEGIN{printf "%.1f", (a-b)/3600}')"
  else st="-"; wait_h="refused: $(tr '\n' ' ' <<< "$out" | cut -c1-90)"; fi
  printf '%-8s %-12s %-18s %-20s %s\n' "$n" "$w" "${q/_/default}" "$st" "$wait_h"
done; done; done
