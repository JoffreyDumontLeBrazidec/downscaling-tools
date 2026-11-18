#!/bin/bash
# Step 2 of building gl512-20261118-prepml-wrap-fast-tp12h (18 Nov 2026): forcing-date fix in a NEW frozen runner copy,
# and the new wrap pointed at it. Shared copies are not edited.
set -eu
SB=/home/ecm5702/hpcperm/sandbox/20261109-gl512-prepml
O=$SB/season-scale-tools-frozen-20261110
N=$SB/season-scale-tools-frozen-20261118-forcingfix
X=/home/ecm5702/agent-work/20261117-forcing-date/season-scale-tools-fix
A=/home/ecm5702/dev/anemoi-inference-configs/prepml_envs/gl512-20261118-prepml-wrap-fast-tp12h/bin/activate
[ -e "$N" ] && { echo "EXISTS $N"; exit 1; }
cp -a "$O" "$N"
rm -rf "$N/__pycache__"
cp "$X/downscaling_unified_runner.py" "$N/downscaling_unified_runner.py"
cp "$X/fix.diff" "$N/FORCING_DATE_FIX.diff"
echo "== diff frozen-20261110 vs forcingfix:"
diff "$O/downscaling_unified_runner.py" "$N/downscaling_unified_runner.py" || true
sed -i "s#season-scale-tools-frozen-20261110#season-scale-tools-frozen-20261118-forcingfix#" "$A"
# restore the fast-wrap comment line that the earlier global rename touched
sed -i "s#^\# --- gl512-20261118-prepml-wrap-fast-tp12h (2026-11-11)#\# --- gl512-20261111-prepml-wrap-fast (2026-11-11)#" "$A"
cat >> "$A" <<'EOF'
# --- gl512-20261118-prepml-wrap-fast-tp12h (2026-11-18): the fast wrap plus two fixes (docs quaver scorecards note s5.9).
# (1) site/gl512_tp_patches.py: tp/cp = 0 at step 0 in Accumulate; the lane steps every 12 h so the 12-h tp/cp windows of the
#     checkpoint tile the forecast. (2) season-scale-tools-frozen-20261118-forcingfix: UnifiedDownscalingRunner.predict_step
#     uses input_date = date (valid date) for the high-res time forcings, not date - step (start date); FORCING_DATE_FIX.diff
#     there, same change as anemoi-inference 39d5371 (agent-work/20261117-forcing-date).
EOF
echo "== activate PYTHONPATH and tail:"
grep -n "season-scale\|wrap-fast" "$A"
