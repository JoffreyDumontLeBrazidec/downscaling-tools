#!/usr/bin/env python3
"""reg_job.sh and reg_accept.sh of the regression baseline from FS-242's sandbox body (gs_job.sh, gs_accept.sh of
sandbox/20261111-gs-global-crop on Jupiter, docs copy epics/paper-o320-o1280/support/20261117_gs_rope/recon/jupiter_sandbox/jobs).
Every replacement must match exactly once, else nothing is written. Changes: sandbox path; warm-start checks instead of the
from-scratch checks (warm file path and sha256, transfer_learning and load_weights_only true); training_approach
deterministic; GL512's 10 outputs instead of 12; acceptance wants the warm-start lines and frozen QK gains equal to the warm
file's (not all ones).
Usage: python3 make_reg_job.py <gs_job.sh> <gs_accept.sh> <out_dir>
"""
import sys
from pathlib import Path

job, acc, out = Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text(), Path(sys.argv[3])
NEW_EXP = "/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261117-regression-baseline"


def sub(s, a, b, what):
    n = s.count(a)
    if n != 1:
        raise SystemExit(f"{what}: pattern found {n} times, want 1: {a[:120]!r}")
    return s.replace(a, b)


job = sub(job, "set -eo pipefail\nEXP=/e/data1/jureap-data/ecmwf/users/jdumont/sandbox/20261111-gs-global-crop\n",
          "# REGRESSION BASELINE (17 Nov 2026, downscaling-tools kits/20261117-baselines/reg, made by make_reg_job.py from FS-242's\n"
          "# gs_job.sh): GL512 450k EMA-2000 warm start, training_approach deterministic, 10 outputs, lr 1e-4, 4,000 km caps.\n"
          f"set -eo pipefail\nEXP={NEW_EXP}\n", "EXP")
job = sub(job, """grep -qE '^    warm_start: null$' "$CFG_FILE" || { echo "FATAL: from-scratch arm: system.input.warm_start must be null" >&2; exit 42; }\n""",
          """: "${LL_WARM_FILE:?}" "${LL_WARM_SHA:?}"
grep -qxF "    warm_start: $LL_WARM_FILE" "$CFG_FILE" || { echo "FATAL: system.input.warm_start must be $LL_WARM_FILE" >&2; exit 42; }
[[ -f "$LL_WARM_FILE.sha256ok" && "$(cat "$LL_WARM_FILE.sha256ok")" == "$LL_WARM_SHA" ]] || {
  [[ "$(sha256sum "$LL_WARM_FILE" | awk '{print $1}')" == "$LL_WARM_SHA" ]] || { echo "FATAL: warm file sha256 is not $LL_WARM_SHA" >&2; exit 42; }
  echo "$LL_WARM_SHA" > "$LL_WARM_FILE.sha256ok"; }
echo "[warm] $LL_WARM_FILE sha256 $LL_WARM_SHA"
grep -qxF "  training_approach: deterministic" "$CFG_FILE" || { echo "FATAL: training_approach must be deterministic" >&2; exit 42; }
export LL_WARM_FILE
""", "warm check")
job = sub(job, """out12 = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp", "z_500", "t_850"]\ngpm""",
          """out12 = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"]   # GL512's 10 outputs (name kept)\ngpm""", "out list")
job = sub(job, """    "FROM SCRATCH: transfer_learning false, load_weights_only false, no warm_start": t.get("transfer_learning") is False and t.get("load_weights_only") is False
            and c["system"]["input"].get("warm_start") is None,""",
          """    "WARM START: transfer_learning true, load_weights_only true, warm_start set": t.get("transfer_learning") is True and t.get("load_weights_only") is True
            and bool(c["system"]["input"].get("warm_start")),
    "training_approach deterministic": t.get("training_approach") == "deterministic",""", "pychk warm")
job = sub(job, """    "12 outputs (rule 29) in training/validation/test": all(c["dataloader"][s]["datasets"]["out_hres"].get("reorder") == out12
            and not {"z_500", "t_850"} & set(c["dataloader"][s]["datasets"]["out_hres"]["drop"]) for s in ("training", "validation", "test")),""",
          """    "GL512's 10 outputs in training/validation/test": all(c["dataloader"][s]["datasets"]["out_hres"].get("reorder") == out12
            and {"z_500", "t_850"} <= set(c["dataloader"][s]["datasets"]["out_hres"]["drop"]) for s in ("training", "validation", "test")),""", "pychk outputs")
job = sub(job, """    ("full resume of " + resume if resume else "from scratch (transfer_learning false, no warm_start)"): t["transfer_learning"] is False and t["load_weights_only"] is False and c["system"]["input"].get("warm_start") == resume,""",
          """    ("full resume of " + resume if resume else "warm start (transfer_learning true, load_weights_only true)"): (t["transfer_learning"] is False and t["load_weights_only"] is False and c["system"]["input"].get("warm_start") == resume) if resume
            else (t["transfer_learning"] is True and t["load_weights_only"] is True and c["system"]["input"].get("warm_start") == __import__("os").environ["LL_WARM_FILE"]),
    "training_approach deterministic": t.get("training_approach") == "deterministic",""", "pyres warm")
job = sub(job, """    "12 outputs": c["dataloader"]["training"]["datasets"]["out_hres"]["reorder"][-2:] == ["z_500", "t_850"],""",
          """    "10 outputs": c["dataloader"]["training"]["datasets"]["out_hres"]["reorder"] == ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"],""", "pyres outputs")
job = sub(job, 'source "$EXP/jobs/gs_accept.sh"', 'source "$EXP/jobs/reg_accept.sh"', "accept source")
# deterministic objective: the EDM weight line is the FLAT one (sigma fixed at 5e5), not FS-242's sigma-dependent one
# (REG gate 2964697 failed on this line only, 17 Nov 19:15 UTC)
job = sub(job, 'EDM_LINE="[weight] preconditioning=edm sigma_data=1.0 loss weight sigma-dependent"',
          'EDM_LINE="[weight] preconditioning=edm sigma_data=1.0 loss weight FLAT: sigma=5e+05"', "EDM line")

acc = sub(acc, """  echo "[accept] warm-start lines (want 0, from scratch): $tl"; [[ "$tl" == 0 ]] || acc=1""",
          """  echo "[accept] warm-start lines (want > 0, warm start from $LL_WARM_FILE): $tl"; [[ "$tl" -gt 0 ]] || acc=1""", "accept warm")
acc = sub(acc, """notone = [k for k, v in qk.items() if not torch.all(v == 1)]""",
          """warm = torch.load(os.environ["LL_WARM_FILE"], weights_only=False, map_location="cpu")["state_dict"]
notone = [k for k, v in qk.items() if k not in warm or not torch.equal(v.float(), warm[k].float())]   # frozen = equal to the warm file""", "accept qk")
acc = sub(acc, """not all ones: {len(notone)} tensors""", """different from the warm file: {len(notone)} tensors""", "accept qk msg")
out.mkdir(parents=True, exist_ok=True)
(out / "reg_job.sh").write_text(job)
(out / "reg_accept.sh").write_text(acc)
print("wrote", out / "reg_job.sh", out / "reg_accept.sh")
