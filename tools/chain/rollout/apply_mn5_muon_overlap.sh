#!/bin/bash
# Switch the MN5 Muon FS-250 chain (sandbox 20261125-gl450-muon-e2) to overlap mode (28 Nov 2026, owner "go overlap").
# Run on the MN5 login node from a folder holding this repo's tools/chain as ./chain:  bash apply_mn5_muon_overlap.sh [--apply]
# Without --apply it prints what it would change. With --apply:
#  - backs up jobs/chain -> jobs/chain.bak_20261128 and jobs/muon_job.sh -> jobs/muon_job.sh.bak_20261128 (copies)
#  - copies chain_segment.sh chain_handover.py chain_pool.py pilot_job.sh kernel_guard.py into jobs/chain (keeps the
#    sandbox's chain_state.py, chain_guard_launch.py, store_guard.py: unchanged code paths for the training step)
#  - muon_job.sh: `chain_takeover "$CKPT_ROOT"` before `chain_resolve`, `chain_next` -> `chain_next_auto`
#  - writes <chain root>/CHAIN.overlap = "0 21600": successor queued now, no dependency (earns age); a successor
#    that starts while the writer is younger than 6 h requeues itself and exits (yield rule, review 28 Nov)
set -eo pipefail
EXP=/gpfs/projects/ehpc01/ecm800825/sandbox/20261125-gl450-muon-e2
ROOT=$EXP/runs/checkpoints/gl450_muon_e2_50k
NEW="$(cd "$(dirname "$0")/.." && pwd)/chain"   # kit layout: <kit>/chain, <kit>/rollout
[[ -f "$NEW/chain_handover.py" ]] || { echo "need ./chain (tools/chain of downscaling-tools PR 66)" >&2; exit 2; }
APPLY=0; [[ "${1:-}" == --apply ]] && APPLY=1
BODY=$EXP/jobs/muon_job.sh
NEWBODY="$(cd "$(dirname "$0")" && pwd)/$(basename "$BODY").new"   # outside the sandbox
python3 "$NEW/../rollout/edit_body.py" "$BODY" --takeover-before 'chain_resolve "$CKPT_ROOT" "$LL_END_STEP"' \
    --root-var CKPT_ROOT --next-from 'chain_next "$CKPT_ROOT"' --next-to 'chain_next_auto "$CKPT_ROOT"; chain_fence "$CKPT_ROOT"' \
    --out "$NEWBODY"
diff -u "$BODY" "$NEWBODY" || true
if [[ $APPLY == 1 ]]; then
  [[ -e $EXP/jobs/chain.bak_20261128 ]] || cp -rp "$EXP/jobs/chain" "$EXP/jobs/chain.bak_20261128"
  [[ -e $BODY.bak_20261128 ]] || cp -p "$BODY" "$BODY.bak_20261128"
  for f in chain_segment.sh chain_handover.py chain_pool.py pilot_job.sh kernel_guard.py; do cp "$NEW/$f" "$EXP/jobs/chain/$f"; done
  cat "$NEW/SHA.txt" > "$EXP/jobs/chain/COMMIT.overlap" 2>/dev/null || true
  cp "$NEWBODY" "$BODY"; chmod +x "$BODY"
  echo "0 21600 overlap: begin delay 0 s, min writer age 6 h; MN5 Muon, 28 Nov 2026" > "$ROOT/CHAIN.overlap"
  bash -n "$BODY" && echo "APPLIED: $BODY, $ROOT/CHAIN.overlap"
fi
