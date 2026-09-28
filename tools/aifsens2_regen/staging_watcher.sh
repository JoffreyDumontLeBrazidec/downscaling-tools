#!/bin/bash
#SBATCH --job-name=a2-staging-watch
#SBATCH --qos=nf
#SBATCH --time=1-00:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/ecm5702/agent-work/20260910-aifsens2-2026-early/logs/%x_%j.out
#
# Keeps the February to May staging honest.
#
# Each block needs twenty grouped input files: sfc, pl, q, con and wave at the four input times.
# A retrieval task killed by the twelve-hour job limit leaves its file absent, and because the
# assemblies depend on the retrieval arrays with afterany they would otherwise run on incomplete
# data and quietly produce a partial month. So the four assemblies are held and this watcher:
#   - releases a block's assembly as soon as that block has all twenty files;
#   - resubmits a block's retrieval arrays when files are missing and no retrieval job of that
#     block is left in the queue. That is safe and cheap: the retrieval skips any request whose
#     target file already holds the expected number of fields;
#   - gives up on a block after MAX_ROUNDS resubmissions, leaving its assembly held and saying so,
#     rather than looping for ever on data the archive cannot serve.
#
# Nothing is deleted and no request is ever substituted by the control or by another member.
#
# Two defects of the first version are fixed here: GROUPS is a special shell variable in bash and
# silently refused the group list, and the check for live retrieval jobs matched on the command
# line, which was unreliable, so every block was resubmitted at once. Live jobs are now tracked by
# explicit job id per block.
set -uo pipefail

RAW=/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/ic/raw
TOOLS=/home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen
MAX_ROUNDS=4
SLEEP=600

GRPS="sfc pl q con wave"
TIMES="0000 0600 1200 1800"

BLOCKS="202602 202603 202604 202605"
declare -A ASM=( [202602]=35425645 [202603]=35425679 [202604]=35425700 [202605]=35426012 )
# Retrieval jobs currently responsible for each block, comma separated. The authority is the file
# $RAW/<block>/.live_jobs, so that jobs submitted by hand from outside this watcher are picked up
# without restarting it. The values below are only the fallback when that file does not exist.
declare -A LIVE=( [202602]=35597022 [202603]="" [202604]=35701524 [202605]=35701526 )

live_ids() {
    local b=$1 f="$RAW/$1/.live_jobs"
    if [ -s "$f" ]; then tr -d " \n" < "$f"; else echo "${LIVE[$b]}"; fi
}

set_live_ids() { printf '%s' "$2" > "$RAW/$1/.live_jobs"; }
declare -A ROUNDS=( [202602]=0 [202603]=0 [202604]=0 [202605]=0 )

log() { echo "[$(date -u +%FT%TZ)] $*"; }

missing_for() {
    local b=$1 miss=""
    for g in $GRPS; do
        for t in $TIMES; do
            [ -f "$RAW/$b/${g}_${t}.grib" ] || miss="$miss ${g}_${t}"
        done
    done
    echo "$miss"
}

live_count() {
    # how many of this block's tracked retrieval jobs are still queued or running
    local ids=$1
    [ -z "$ids" ] && { echo 0; return; }
    squeue -j "$ids" -h -o "%i" 2>/dev/null | wc -l
}

log "watcher started; the four assemblies are held and are released only when a block is complete"
for b in $BLOCKS; do log "  $b assembly ${ASM[$b]} tracked retrieval jobs $(live_ids "$b")"; done

remaining="$BLOCKS"
while [ -n "$remaining" ]; do
    still=""
    for b in $remaining; do
        miss=$(missing_for "$b")
        if [ -z "$miss" ]; then
            log "$b COMPLETE, 20 of 20 groups present -> releasing assembly ${ASM[$b]}"
            scontrol release "${ASM[$b]}" && log "$b assembly ${ASM[$b]} released"
            continue
        fi
        nmiss=$(echo $miss | wc -w)
        nq=$(live_count "$(live_ids "$b")")
        if [ "$nq" -gt 0 ]; then
            log "$b waiting: $nmiss groups missing ($miss ), $nq retrieval jobs still live"
            still="$still $b"
            continue
        fi
        r=${ROUNDS[$b]}
        if [ "$r" -ge "$MAX_ROUNDS" ]; then
            log "$b GIVING UP after $r resubmissions; still missing:$miss"
            log "$b assembly ${ASM[$b]} stays HELD; a human must decide whether these starts are lost"
            continue
        fi
        ROUNDS[$b]=$((r + 1))
        log "$b no live retrieval but $nmiss groups missing ($miss ) -> resubmission round ${ROUNDS[$b]}"
        cd "$TOOLS" || exit 1
        ids=""
        if echo "$miss" | grep -qE "sfc|pl|q_|con"; then
            a=$(sbatch --parsable --time=1-00:00:00 --array=1-16%8 retrieve.sbatch "$b" atm 2>/dev/null) && ids="$a"
            log "$b resubmitted atmospheric array: ${a:-REFUSED (submission limit), will retry next round}"
        fi
        if echo "$miss" | grep -q "wave"; then
            w=$(sbatch --parsable --time=1-00:00:00 --array=1-4%2 retrieve.sbatch "$b" wave 2>/dev/null) && ids="${ids:+$ids,}$w"
            log "$b resubmitted wave array: ${w:-REFUSED (submission limit), will retry next round}"
        fi
        if [ -z "$ids" ]; then
            ROUNDS[$b]=$r   # nothing was actually submitted, so do not spend a round on it
            log "$b nothing could be submitted this round; the round counter is unchanged"
        fi
        set_live_ids "$b" "$ids"
        still="$still $b"
    done
    remaining="$still"
    [ -n "$remaining" ] && sleep $SLEEP
done

log "watcher finished; every block it watched is complete or was given up on"
