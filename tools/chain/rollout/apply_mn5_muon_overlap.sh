#!/bin/bash
# Switch the MN5 Muon FS-250 chain (sandbox 20261125-gl450-muon-e2) to overlap mode (28 Nov 2026, owner "go overlap").
# Run on the MN5 login node from a folder holding this repo's tools/chain as ./chain:  bash apply_mn5_muon_overlap.sh [--apply]
# Without --apply it prints what it would change. With --apply:
#  - backs up jobs/chain -> jobs/chain.bak_20261128 and jobs/muon_job.sh -> jobs/muon_job.sh.bak_20261128 (copies)
#  - copies chain_segment.sh chain_handover.py chain_pool.py pilot_job.sh kernel_guard.py into jobs/chain (keeps the
#    sandbox's chain_state.py, chain_guard_launch.py, store_guard.py: unchanged code paths for the training step)
#  - muon_job.sh: `chain_takeover "$CKPT_ROOT"` before `chain_resolve`, `chain_next` -> `chain_next_auto`
#  - writes <chain root>/CHAIN.overlap = 43200 (successor queued with --begin=now+12h, no dependency)
set -eo pipefail
EXP=/gpfs/projects/ehpc01/ecm800825/sandbox/20261125-gl450-muon-e2
ROOT=$EXP/runs/checkpoints/gl450_muon_e2_50k
NEW="$(cd "$(dirname "$0")" && pwd)/chain"
[[ -f "$NEW/chain_handover.py" ]] || { echo "need ./chain (tools/chain of downscaling-tools PR 66)" >&2; exit 2; }
APPLY=0; [[ "${1:-}" == --apply ]] && APPLY=1
BODY=$EXP/jobs/muon_job.sh
python3 "$NEW/../rollout/edit_body.py" "$BODY" --takeover-before 'chain_resolve "$CKPT_ROOT" "$LL_END_STEP"' \
    --root-var CKPT_ROOT --next-from 'chain_next "$CKPT_ROOT"' --next-to 'chain_next_auto "$CKPT_ROOT"' \
    --out "$BODY.new"
diff -u "$BODY" "$BODY.new" || true
if [[ $APPLY == 1 ]]; then
  [[ -e $EXP/jobs/chain.bak_20261128 ]] || cp -rp "$EXP/jobs/chain" "$EXP/jobs/chain.bak_20261128"
  [[ -e $BODY.bak_20261128 ]] || cp -p "$BODY" "$BODY.bak_20261128"
  for f in chain_segment.sh chain_handover.py chain_pool.py pilot_job.sh kernel_guard.py; do cp "$NEW/$f" "$EXP/jobs/chain/$f"; done
  cat "$NEW/SHA.txt" > "$EXP/jobs/chain/COMMIT.overlap" 2>/dev/null || true
  cp "$BODY.new" "$BODY"; chmod +x "$BODY"
  echo "43200 overlap begin delay (s), MN5 Muon, 28 Nov 2026" > "$ROOT/CHAIN.overlap"
  bash -n "$BODY" && echo "APPLIED: $BODY, $ROOT/CHAIN.overlap"
fi
rm -f "$BODY.new"
