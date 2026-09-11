#!/bin/bash
#SBATCH --job-name=a2-release
#SBATCH --qos=nf
#SBATCH --time=1-00:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/ecm5702/agent-work/20260910-aifsens2-2026-early/logs/%x_%j.out
#
# Release each block's held assembly, and only that.
#
# This replaces the staging watcher of section 42, which was stopped because it also RESUBMITTED
# retrieval arrays, and it resubmitted them in the monthly form that this archive will not
# schedule (section 47). Submitting is now done by scripts/submit_chunks.sh and by hand. This job
# does one thing: it waits until a block genuinely has every input group and then releases that
# block's assembly, which has been held so that no month can be assembled from partial input.
#
# A group counts as present when the retrieval's own --check passes for it, which verifies the
# field count of the monthly file or of every chunk file. Both the monthly and the chunked paths
# write atomically after verifying the count, so a file that exists is a file that is complete.
#
# It never submits, never cancels and never deletes.
set -uo pipefail

TOOLS=/home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools
RAW=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/ic/raw
SLEEP=600
TIMES="0000 0600 1200 1800"
ATM="sfc pl q con"

declare -A ASM=( [202602]=35425645 [202603]=35425679 [202604]=35425700 [202605]=35426012 )

log() { echo "[$(date -u +%FT%TZ)] $*"; }

# shellcheck disable=SC1091
source "$TOOLS/aifsens2_regen/env.sh" >/dev/null 2>&1
cd "$TOOLS" || exit 1

group_ready() {
    local b=$1 g=$2 t=$3
    # the monthly file, written only after its field count was verified
    [ -f "$RAW/$b/${g}_${t}.grib" ] && return 0
    # otherwise every chunk of that group must be complete
    [ -d "$RAW/$b/${g}_${t}_bychunk" ] || return 1
    python -m aifsens2_regen.retrieve --block "$b" --group "$g" --time "$t" --check >/dev/null 2>&1
}

log "releaser started; it releases held assemblies only, and never submits, cancels or deletes"
for b in "${!ASM[@]}"; do log "  $b assembly ${ASM[$b]}"; done

remaining="202602 202603 202604 202605"
while [ -n "$remaining" ]; do
    still=""
    for b in $remaining; do
        missing=""
        for g in $ATM; do
            for t in $TIMES; do
                group_ready "$b" "$g" "$t" || missing="$missing ${g}_${t}"
            done
        done
        for t in $TIMES; do
            group_ready "$b" wave "$t" || missing="$missing wave_${t}"
        done
        if [ -z "$missing" ]; then
            log "$b COMPLETE: every input group present -> releasing assembly ${ASM[$b]}"
            if scontrol release "${ASM[$b]}"; then
                log "$b assembly ${ASM[$b]} released"
            else
                log "$b COULD NOT release assembly ${ASM[$b]}; a human must do it"
            fi
            continue
        fi
        n=$(echo $missing | wc -w)
        log "$b waiting on $n groups:$missing"
        still="$still $b"
    done
    remaining="$still"
    [ -n "$remaining" ] && sleep $SLEEP
done
log "releaser finished; every block it watched has been released"
