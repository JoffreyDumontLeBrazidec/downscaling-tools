# shellcheck shell=bash
# gs-global-crop smoke acceptance (sourced by jobs/gs_job.sh; tools/accept_only.sh re-runs it on a finished log).
# Needs: JOBLOG CKPT_ROOT EXP LL_TAG LL_MODE JOBID EDM_LINE peak_mib. Never exits the caller; sets acc (0 = PASS).
# 11 Nov 2026 19:10 UTC: errexit/pipefail off inside (a grep|head SIGPIPE killed the crop smokes' gate, exit 13);
# fallback check reads the crop counter line only (it matched the datamodule's "Falling back rollout to: 0").
accept() {
  ACC_KIND="$1"; export ACC_KIND
  local _opts="$-"; set +e +o pipefail
  # ─── Smoke acceptance for a from-scratch start; a failed check exits 1 ───
  acc=0
  logs=("$JOBLOG" "${JOBLOG%.out}.err")
  r0() { cat "${logs[@]}" 2>/dev/null | grep -aE "^ *0: .*$1" || true; }
  nm="$(r0 'GraphTransformerMapperBlock: QK-norm parameters frozen' | wc -l)"
  np="$(r0 'GraphTransformerProcessorBlock: QK-norm parameters frozen' | wc -l)"
  if [[ "${ACC_FREEZE:-true}" == true ]]; then wm=2; wp=16; else wm=0; wp=0; fi
  echo "[accept] rank-0 'QK-norm parameters frozen' lines: mapper $nm (want $wm), processor $np (want $wp)"
  [[ "$nm" == "$wm" && "$np" == "$wp" ]] || acc=1
  for want in "Loading residual statistics from /e/data1/jureap-data/ecmwf/users/jdumont/anemoi/residuals_statistics/o1280_dict_0_72_precip.npy" \
              "Direct-only prognostic channels \(out_hres not in in_lres\): \['cp', 'tp'\]" \
              "Training noise: mixture 0.50 x LogNormal\(0.693, 0.750\) \+ 0.50 x LogNormal\(1.000, 2.500\)"; do
    n="$(cat "${logs[@]}" 2>/dev/null | grep -acE "$want" || true)"
    echo "[accept] lines matching '$want': $n"; [[ "$n" -gt 0 ]] || acc=1
  done
  tl="$(cat "${logs[@]}" 2>/dev/null | grep -acE 'Loading weights with Transfer Learning|Warm start by variable name' || true)"
  echo "[accept] warm-start lines (want > 0, warm start from $LL_WARM_FILE): $tl"; [[ "$tl" -gt 0 ]] || acc=1
  fe="$(grep -acF "$EDM_LINE" "$JOBLOG" || true)"
  echo "[accept] EDM line: $fe"; [[ "$fe" -gt 0 ]] || acc=1
  grep -aF "$EDM_LINE" "$JOBLOG" | head -1 | cut -c1-300
  if [[ "$LL_MODE" == crop ]]; then
    echo "[accept] crop lines:"; grep -ahsE 'crop loader_read on|training\.crop dynamic|dynamic crop' "${logs[@]}" | head -12 | cut -c1-400
  fi
  python3 - "$JOBLOG" <<'PYL' || acc=1
import re, sys
txt = open(sys.argv[1], errors="replace").read()
import os
LR = float(os.environ.get("ACC_LR", "5.0e-4"))   # per-arm lr (16 Nov 2026)
groups = re.findall(r"^ *0: \[lr\]   group \d+ '([^']+)': (\d+) tensors, ([\d,]+) params, initial_lr=([0-9.e+-]+)(?:, lr now=[0-9.e+-]+)?(?:, weight_decay=([0-9.e+-]+|None))?", txt, re.M)
print(f"[accept] lr groups on rank 0 (name, tensors, params, initial_lr, weight_decay): {groups}")
ok_groups = bool(groups) and all(abs(float(g[3]) - LR) < 1e-12 and g[4] not in ("", "None") and float(g[4]) == 0.0 for g in groups)
print(f"[accept] every group initial_lr {LR:g} and weight_decay 0.0 -> {'ok' if ok_groups else 'FAIL'}")
lines = re.findall(r"\[lr\] optimizer param_groups lr after global_step=(\d+) \(applied at step \d+\): ([^\n\r\[]*)", txt)
import os
kind = os.environ.get("ACC_KIND", "smoke")
after = [(int(st), dict(re.findall(r"([\w]+)=([0-9.e+-]+)", rest))) for st, rest in lines if int(st) >= 50]
if kind == "smoke":
    ok_lr = bool(after) and all(abs(float(v) - LR) < 1e-12 for st, d in after for k, v in d.items())
    print(f"[accept] lr after the 50-step smoke warm-up: {after[-1] if after else None} -> {'ok' if ok_lr else 'FAIL'}")
else:
    # the wrapper prints the lr at steps 1, 2, warm-up/2, warm-up, warm-up+1 and every 1000: within 200 steps of a
    # 1000-step warm-up only steps 1 and 2 are printed (fixed 11 Nov 19:15 UTC: the check looked at steps >= 50 only)
    allw = [(int(st), dict(re.findall(r"([\w]+)=([0-9.e+-]+)", rest))) for st, rest in lines]
    ok_lr = bool(allw) and all(0.0 < float(v) <= LR * (st + 2) / 1000 + 1e-12 for st, d in allw for k, v in d.items())
    print(f"[accept] lr inside the 1000-step warm-up (0 < lr <= 5e-4 x (step+2)/1000): {allw[:3]} -> {'ok' if ok_lr else 'FAIL'}")
loss = [float(x) for x in re.findall(r"train_multi_dataset_loss_step=([0-9.eE+-]+|nan|inf)", txt)]
w = min(50, len(loss) // 2)
first, last = (loss[:w], loss[-w:]) if w >= 10 else ([], [])
mf, ml = sum(first) / max(len(first), 1), sum(last) / max(len(last), 1)
finite = all(x == x and abs(x) != float("inf") for x in loss)
ok_loss = bool(first) and finite and (ml < mf if kind == "smoke" else ml <= 1.3 * mf)
print(f"[accept] train loss: {len(loss)} values, finite {finite}, first-{len(first)} mean {mf:.4g}, last-{len(last)} mean {ml:.4g} "
      f"({'from scratch: must fall' if kind == 'smoke' else 'blow-up gate: last <= 1.3 x first'}; fell: {ml < mf}) -> {'ok' if ok_loss else 'FAIL'}")
print(f"[accept] first 5 train losses {loss[:5]}")
sys.exit(0 if ok_groups and ok_lr and ok_loss else 1)
PYL
  if grep -qiE 'loss[^ ]*[=: ]+(nan|inf)' "$JOBLOG"; then echo "[accept] FAIL non-finite loss in the log"; acc=1; fi
  grep -aE '^ *0: .*(\[timing\] SUMMARY|\[wait\] SUMMARY|BUILD-HALO\] my_rank=0)' "$JOBLOG" | head -5 | cut -c1-400
  grep -aE '^ *[0-9]+: \[(timing|wait)\] SUMMARY' "$JOBLOG" | cut -c1-300
  grep -aE '^ *[0-9]+: \[peakmem\]' "$JOBLOG" | sort | head -8
  last="$(ls -t "$CKPT_ROOT"/*/anemoi-by_step-*.ckpt 2>/dev/null | head -1)"
  if [[ -n "$last" ]]; then
    echo "[accept] last smoke checkpoint: $last"
    python3 - "$last" <<'PYQK' || acc=1
import sys, torch
sd = torch.load(sys.argv[1], weights_only=False, map_location="cpu")["state_dict"]
qk = {k: v for k, v in sd.items() if ".q_norm." in k or ".k_norm." in k}
import os
frozen = os.environ.get("ACC_FREEZE", "true") == "true"
warm = torch.load(os.environ["LL_WARM_FILE"], weights_only=False, map_location="cpu")["state_dict"]
notone = [k for k, v in qk.items() if k not in warm or not torch.equal(v.float(), warm[k].float())]   # frozen = equal to the warm file
vals = torch.cat([v.float().flatten() for v in qk.values()]) if qk else torch.zeros(1)
print(f"[accept] QK-norm gain tensors in the step checkpoint: {len(qk)} (want 36); gains min/median/max "
      f"{vals.min():.4f}/{vals.median():.4f}/{vals.max():.4f}; different from the warm file: {len(notone)} tensors ({'want 0, frozen' if frozen else 'free gains: informative only'})")
sys.exit(0 if len(qk) == 36 and (not frozen or not notone) and bool(torch.isfinite(vals).all()) else 1)
PYQK
    python3 "$EXP/tools/all_blocks_scale.py" "$last" > "$EXP/checks/s11_${LL_TAG}_${JOBID}.txt" 2>&1 || acc=1
    cat "$EXP/checks/s11_${LL_TAG}_${JOBID}.txt"
    python3 - "$EXP/checks/s11_${LL_TAG}_${JOBID}.txt" <<'PYS' || acc=1
import re, sys
pat = re.compile(r"edge-logit scale per head [^|]*max ([0-9.e+-]+)")
now = {l.split("|")[0].strip(): float(pat.search(l).group(1)) for l in open(sys.argv[1]) if l.strip().startswith("model.") and pat.search(l)}
bad = [b for b, v in now.items() if not v < 80]
print(f"[accept] s11 on {len(now)} blocks (want 18); highest {max(now.values()) if now else None}; at or above the trap level 80: {bad or 'none'}")
sys.exit(0 if len(now) == 18 and not bad else 1)
PYS
  else
    echo "[accept] FAIL no step checkpoint under $CKPT_ROOT"; acc=1
  fi
  # inchain gate (cloud thread 11 Nov 13:45 UTC): peak memory under 90 GB, no crop fallback, bounded recompiles
  pk="$(grep -aE '^ *[0-9]+: \[peakmem\]' "$JOBLOG" | sed -E 's/.*max_memory_reserved=([0-9.]+) GiB.*/\1/' | sort -g | tail -1)"
  echo "[accept] torch max_memory_reserved over ranks: ${pk:-none} GiB (gate < 83.8 GiB = 90 GB); nvidia-smi peak ${peak_mib} MiB (gate < 85831 MiB = 90 GB)"
  python3 -c "import sys; sys.exit(0 if (float('${pk:-1e9}') < 83.8 and float('${peak_mib:-1e9}') < 85831) else 1)" || { echo "[accept] FAIL peak memory"; acc=1; }
  fbmax="$(cat "${logs[@]}" 2>/dev/null | grep -aoE 'fallbacks [0-9]+ of [0-9]+ draws' | awk '{if ($2+0>m) m=$2+0} END {print m+0}')"
  fbn="$(cat "${logs[@]}" 2>/dev/null | grep -acE 'fallbacks [0-9]+ of [0-9]+ draws')"
  echo "[accept] crop fallbacks: max $fbmax over $fbn counter lines (want 0; lines appear every 1000 steps, so none in 200 steps is normal)"; [[ "$fbmax" == 0 ]] || acc=1
  rcmp="$(cat "${logs[@]}" 2>/dev/null | grep -av '\[cfg\]' | grep -aciE 'hit config\.(recompile|cache_size)_limit|(recompile|cache_size)_limit[^ ]* *(\([0-9]+\) )?reached' || true)"
  echo "[accept] recompile-limit warnings: $rcmp (want 0); recompile lines: $(cat "${logs[@]}" 2>/dev/null | grep -aci 'recompil' || true)"; [[ "$rcmp" == 0 ]] || acc=1
  echo "[accept] $([[ $acc == 0 ]] && echo 'PASS every check' || echo 'FAIL')"
  [[ "$_opts" == *e* ]] && set -e; set -o pipefail
  return 0
}
