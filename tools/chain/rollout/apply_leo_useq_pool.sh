#!/bin/bash
# Put the 6 Leonardo u-screen arms (sandbox 20261127-u-screen, ledger 163) in ONE pilot pool (28 Nov 2026, owner
# "go overlap ... two jobs faster"). Run on the Leonardo login node from a folder holding tools/chain as ./chain:
#   bash apply_leo_useq_pool.sh [--apply]       (without --apply: prints the body diff and the pool file only)
# With --apply:
#  - copies tools/chain into $S/jobs/chain (new folder) and backs up jobs/useq_job.sh -> useq_job.sh.bak_20261128
#  - useq_job.sh: after `CK=$OUTPUT/checkpoint`, if $CK/CHAIN.json exists (pool arms only), source chain_segment.sh,
#    chain_takeover "$CK" and chain_next_auto "$CK" (CHAIN.pool -> queues nothing; the pilots do). Login renders
#    (CFG_ONLY=1) skip it. The body's own resume (latest last.ckpt) is unchanged.
#  - per arm: chain_state.py init --root runs/<ARM>/checkpoint (refuses if checkpoints already exist: then adopt by
#    hand), CHAIN.pool marker; $S/pool/POOL.tsv (root, 50000, jobs/<ARM>.sbatch) and $S/jobs/useq_pilot.sbatch
#    (the arms' own header: 2 nodes x 4 A100, 24 h).
# It does NOT cancel or submit anything. After it: scancel the held afterany successors, sbatch useq_pilot.sbatch x N.
set -eo pipefail
S=/leonardo_work/DestE_340_26/users/jdumontl/sandbox/20261127-u-screen
ARMS="HB UK UW U8 PM P-c"
NEW="$(cd "$(dirname "$0")" && pwd)/chain"
[[ -f "$NEW/chain_pool.py" ]] || { echo "need ./chain (tools/chain of downscaling-tools PR 66)" >&2; exit 2; }
APPLY=0; [[ "${1:-}" == --apply ]] && APPLY=1
BODY=$S/jobs/useq_job.sh
NEWBODY="$(cd "$(dirname "$0")" && pwd)/$(basename "$BODY").new"   # outside the sandbox
python3 "$NEW/../rollout/edit_body.py" "$BODY" --out "$NEWBODY" --insert-after 'CK=$OUTPUT/checkpoint' \
  --insert-text 'if [[ -f "$CK/CHAIN.json" && "${CFG_ONLY:-0}" != 1 ]]; then   # pool arm (28 Nov 2026): one writer per arm\n  export CHAIN_TOOLS=$S/jobs/chain CHAIN_PY=python KERNEL_GUARD_PY=python\n  source "$CHAIN_TOOLS/chain_segment.sh"\n  chain_takeover "$CK"\n  chain_next_auto "$CK"\n  chain_fence "$CK"\nfi'
diff -u "$BODY" "$NEWBODY" || true
for a in $ARMS; do [[ -f $S/jobs/$a.sbatch ]] || { echo "missing $S/jobs/$a.sbatch" >&2; exit 2; }; done
hdr="$(grep '^#SBATCH' $S/jobs/HB.sbatch | grep -v -e '--job-name' -e '--output' -e '--error')"
for a in $ARMS; do
  [[ "$(grep '^#SBATCH' $S/jobs/$a.sbatch | grep -v -e '--job-name' -e '--output' -e '--error')" == "$hdr" ]] \
    || { echo "REFUSED: $a.sbatch asks for other resources than HB.sbatch; a pool needs one shape" >&2; exit 3; }
done
echo "pool shape:"; echo "$hdr"
if [[ $APPLY == 1 ]]; then
  [[ -e $BODY.bak_20261128 ]] || cp -p "$BODY" "$BODY.bak_20261128"
  mkdir -p "$S/jobs/chain" "$S/pool"; cp "$NEW"/*.py "$NEW"/*.sh "$S/jobs/chain/"; cp "$NEW/SHA.txt" "$S/jobs/chain/" 2>/dev/null || true
  : > "$S/pool/POOL.tsv"
  for a in $ARMS; do
    ck=$S/runs/$a/checkpoint
    [[ -f $ck/CHAIN.json ]] || python3 "$S/jobs/chain/chain_state.py" init --root "$ck" --note "u-screen $a, pilot pool 28 Nov"
    echo "pool $S/pool/POOL.tsv" > "$ck/CHAIN.pool"
    printf '%s\t50000\t%s\n' "$ck" "$S/jobs/$a.sbatch" >> "$S/pool/POOL.tsv"
  done
  { echo '#!/bin/bash'
    echo '# u-screen pilot (pool $S/pool/POOL.tsv): trains whichever arm needs a writer. Submit: sbatch jobs/useq_pilot.sbatch'
    echo '#SBATCH --job-name=useq_pilot'
    echo "$hdr"
    echo "#SBATCH --output=$S/logs/%x_%j.out"
    echo "#SBATCH --error=$S/logs/%x_%j.err"
    echo "CHAIN_PY=python3 CHAIN_TOOLS=$S/jobs/chain bash $S/jobs/chain/pilot_job.sh $S/pool/POOL.tsv 0 14400"   # take over a live arm only in its last 4 h
  } > "$S/jobs/useq_pilot.sbatch"
  cp "$NEWBODY" "$BODY"; chmod +x "$BODY"; bash -n "$BODY"
  cat "$S/pool/POOL.tsv"; echo "APPLIED: $BODY, $S/pool/POOL.tsv, $S/jobs/useq_pilot.sbatch"
fi
