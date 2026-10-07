# LAUNCH: T1d stage A on Atos (box screen of the 1.2M parent), 2026-10-07

For the executing session on the owner's Mac, run on hpc-login. Everything GPU waits for the owner's typed go
(rule below). Paths come from `t1d_env.sh`; export an override before any script if a path differs (for example
`T1D_SANDBOX` when `exp new` runs on another date).

**Owner's-go rule.** Every GPU submission waits for the owner's typed go in the cluster thread. Two gos are needed:
go 1 for the two gate jobs (G1, about 0.2 GPU-hours), go 2 for stage A (five prediction runs, about 6.7 GPU-hours).
Steps 0-2 (preparation, G2 part 1, G3 test-only) are CPU only and need no go.
**Two routes, one switch:** `T1D_HOST=ac` (default, sections 0-8 as written: A100 on AC) or `T1D_HOST=ag` (GH200 on AG for
the gate and the five predictions; everything CPU stays on AC). Section 10 lists what changes on AG. Pick one route
for G1 AND all five predictions: never mix clusters inside stage A (the CUDA noise stream differs between A100 and
GH200, `eval/predict/seeding.py`, so draws are only paired within one GPU class). Do not merge the two gos: stage A is
submitted only after G1 passed and was reported.

## What runs

Checkpoint: the 1.2M parent, run `551dfd1e`, step 400,000 of the c3 fork:
`/home/ecm5702/perm/checkpoints/o320_o1280/551dfd1eb2a649d49f099acd30b59483/inference-anemoi-by_step-epoch_171-step_400000.ckpt`
(the lane's `predict.checkpoint`; the job passes the base file `anemoi-by_step-epoch_171-step_400000.ckpt` of the
same folder as `--name-ckpt`, as `eval.cli predict` does, and the runtime loads the inference file). Never the donor
`12dcefea` (`resolve_lane_t1d.py` refuses it).

| arm | lane | schedule | levels | calls | max ln-step above 10 | max ln-step below 10 | seeds |
|---|---|---|---:|---:|---:|---:|---|
| p12m_pw30_c0 | tc_o320_o1280_p12m_pw30_c0 | piecewise 10 exp + 20 Karras rho 7, sigma_max 1e5, transition 10 | 30 | 59 | 0.921 | 0.461 | 756 and 757000 |
| p12m_c0_pw16_s1k | tc_o320_o1280_p12m_c0_pw16_s1k | piecewise 5 exp + 11 Karras rho 7, sigma_max 1e3, S_max 1e3 | 16 | 31 | 0.921 | 0.851 | 756 |
| p12m_st2 | tc_o320_o1280_p12m_st2 | custom: 1e5, 1e4, 1e3 + the 15 nodes of arm 2 below 1e3 | 18 | 35 | **2.303** | 0.851 | 756 |
| p12m_st4 | tc_o320_o1280_p12m_st4 | custom: 1e5, 3e4, 1e4, 3e3, 1e3 + the same 15 nodes | 20 | 39 | **1.204** | 0.851 | 756 |

All Heun, `S_churn 0`, `S_noise 1.0` (inert at churn 0: the Heun loop draws no noise, so all arms consume only the
initial-noise draws and share them per member), sigma_min 0.03. The full lists with log10 and ln steps:
`python3 make_lanes_t1d.py --out /tmp/t1d_lanes_check`. The T1 Heun high-segment rule is ln-step <= 1.15 safe,
>= 1.3 fails: st2 (2.30) is beyond it by design of the arm, st4 (1.20) sits between the two.

Box protocol (T1 stage 1): Franklin/Idalia box 10-40N 100-58W, cut graph, 1 A100 on AC, fp32; dates 20230826-30,
leads 24 and 120, members 1-10 = 10 files, 100 draws per run. Run roots
`/home/ecm5702/scratch/eval/o320_o1280_p12m_<arm>[_r757]_tc100_20261007` (`t1d_root` in `t1d_env.sh`); the submit
scripts refuse an existing root before submitting anything.

## 0. Preparation (CPU, no go needed)

```bash
W=/home/ecm5702/work-t1d-20261007
mkdir -p $W/code $W/logs $W/notes/gates $W/stageA && readlink -f $W     # must print $W itself: a real dir, not a scratch link
# downscaling-tools: a fresh worktree of the pushed T1d branch (from the Atos clone that has the GitHub fork as a remote)
git -C <atos downscaling-tools clone> fetch <fork remote> claude/project-thread-tpts4t
git -C <atos downscaling-tools clone> worktree add --detach $W/code/downscaling-tools FETCH_HEAD
git -C $W/code/downscaling-tools rev-parse HEAD | tee $W/notes/dstools_sha.txt   # record it; it is the commit every job logs as dstools=
source $W/code/downscaling-tools/scripts/t1d_sampler_20261007/t1d_env.sh
```

Lanes: they are committed; prove the generator reproduces them (no overwrite, temp dir):
```bash
cd $T1D_CODE && python3 $T1D_S/make_lanes_t1d.py --out /tmp/t1d_lanes_check | tee $W/notes/gates/arm_table.txt
for f in /tmp/t1d_lanes_check/*.yaml; do diff -q $f eval/config/lanes/$(basename $f) || echo "LANE DIFFERS: $f"; done
```

Sandbox (owner's rule: rebuild from git in a fresh sandbox with its own uv venv, never from memory; v2 convention of
`docs/epics/certified-runtime-provenance/RUNTIME-LAYOUT.md`):
```bash
quota                                       # $HPCPERM is inode-bound; a uv venv costs ~6k inodes
exp new t1d-stagea                          # -> ~/hpcperm/sandbox/<YYYYMMDD>-t1d-stagea, branch exp/t1d-stagea-<YYYYMMDD>
export T1D_SANDBOX=~/hpcperm/sandbox/<YYYYMMDD>-t1d-stagea     # only if the date is not 20261007
C=$T1D_SANDBOX/code/anemoi-core
git -C $C fetch https://github.com/JoffreyDumontLeBrazidec/anemoi-core claude/project-thread-ncf9gh
git -C $C reset --hard 27391c1597e8090b6e0da9a9ce79352eaa6c6530          # samplers patch on hres-lead 00472fb6f
git -C $C rev-parse HEAD                                                  # must be 27391c1597e8...
git -C $C merge-base --is-ancestor c286c1a3 HEAD && echo "c286c1a3 (the core the 1.2M checkpoint records) is an ancestor"
git -C $C push <fork remote> HEAD:exp/t1d-stagea-<YYYYMMDD>               # best effort; 27391c1 itself is already on the fork
set +u; source $T1D_SANDBOX/activate.sh; set -u                          # the guard must print OK
python -c "import anemoi.models.samplers.diffusion_samplers as d; print(d.__file__, 'custom' in d.NOISE_SCHEDULERS)"   # path inside the sandbox, True
cd $T1D_CODE && python -c "import eval.predict.main, netCDF4, xarray; print('dstools imports OK')"
deactivate
```
If an import of the downscaling-tools predict path fails inside the sandbox venv, STOP and report it: a dependency
change is an edit of `code/env/pyproject.toml` + `uv lock` committed on the exp branch (the owner's call), never an
untracked `pip install`. The stage 1b sandbox `/home/ecm5702/hpcperm/sandbox/20260930-fewstep-dpm/` may still exist:
record `git -C .../code/anemoi-core rev-parse HEAD` (or its equivalent) in `$W/notes/gates/old_1b_sandbox.txt` for
the note, and do not use it.

Commits the 13 steps between c286c1a3 and 00472fb6f add (all default-off, "bit-exact when absent" by their messages):
hres_branch, LeadEmbedding, static node attributes, the local downscaler, optimizer param groups,
gradient-checkpointing forwarding, and the hidden-mesh ensemble-rows fix. Plus 27391c1 (custom schedule, DPM fix;
EDMHeunSampler untouched). G1 is the test that none of them changes a draw of this checkpoint.

Input bundles and the lost stage 3 script: `se_tc_predict_t1d.sbatch` is new (the stage 3 `se_tc_predict_s3.sbatch`
was lost on Atos). Diff it against the Mac copy in `/Users/ecm5702/agent-work/20260930-fewstep-sampler/`; expected
differences are the checkpoint (from the lane), the runtime switch, the resolver, leads 24,120 and the probe. The
default input root is `T1D_INPUT_ROOT=/ec/res4/scratch/ecm5702/eval/o320_o1280/manual_731d203a_pristine_20260818/bundles_with_y`
(the stage 3 AC script's root for the 2023 dates). If the Mac copy used another `--input-root` for the box, export
that one instead and record it in `$W/notes/gates/input_root.txt`. Check the 10 date/lead bundles exist there.

## 1. Gate G2, part 1: lane resolution (CPU, no go needed)

```bash
cd $T1D_CODE; set +u; source $T1D_CERT_VENV/bin/activate; set -u
for a in p12m_pw30_c0 p12m_c0_pw16_s1k p12m_st2 p12m_st4; do
  python $T1D_S/resolve_lane_t1d.py tc_o320_o1280_$a --outdir $W/notes/gates/samplers --check-files
done | tee $W/notes/gates/g2_resolve.txt
```
Pass: four blocks, each with `CHECKPOINT ... run=551dfd1eb2a649d49f099acd30b59483`, the bbox scope with
`cut_graph: true`, `S_churn 0.0`, `S_noise 1.0`, levels/calls 30/59, 16/31, 18/35, 20/39, `QOS(predict) ng`, and both
checkpoint files found. Part 2 runs after the first file lands (step 5).

## 2. Gate G3: `sbatch --test-only` of every job (CPU, no go needed)

```bash
TEST=1 bash $T1D_S/submit_gates_t1d.sh   2>&1 | tee    $W/notes/gates/g3_test_only.txt
TEST=1 bash $T1D_S/submit_stageA_t1d.sh  2>&1 | tee -a $W/notes/gates/g3_test_only.txt
```
Pass: 2 + 13 lines `TEST: sbatch: Job ... to start at ...`, no `error`, no `REFUSED`. (Dependencies are left out in
test mode, as in stage 3.)

## 3. Gate G1: runtime gate (GPU: wait for go 1)

One draw of `p12m_pw30_c0` (date 20230826, lead 24, member 1, base seed 756 = `ANEMOI_BASE_SEED` unset) under the
fresh patched sandbox and under the certified venv `~/dev/.ds-260612` (the x86 twin of `~/dev/.ds-ag-260616`, in
which the 1.2M parent's own reads ran). Same GPU class (A100), same rank count (1), same draw order, so the initial
noise is the same stream.
```bash
bash $T1D_S/submit_gates_t1d.sh          # two jobs, ~0.1 GPU-h each
# when both are COMPLETED:
G=$T1D_E/o320_o1280_p12m_pw30_c0_g1; F=predictions/predictions_20230826_step024.nc
python $T1D_S/g1_compare.py ${G}_sandbox_$T1D_TAG/$F ${G}_certified_$T1D_TAG/$F | tee $W/notes/gates/g1_compare.txt
python $T1D_S/check_attrs_t1d.py ${G}_sandbox_$T1D_TAG/meta/tc_o320_o1280_p12m_pw30_c0.sampler.json --members 1 \
   --log $W/logs/t1d_g1_sandbox_<jobid>.out ${G}_sandbox_$T1D_TAG/$F ${G}_certified_$T1D_TAG/$F | tee -a $W/notes/gates/g1_compare.txt
grep -h "^RUNTIME\|^GPU\|^host=\|^SE_TC_T1D" $W/logs/t1d_g1_*.out | tee -a $W/notes/gates/g1_compare.txt
```
Pass: every weather state of `y_pred` bit-identical, or max |sandbox - certified| <= 1e-3 of the field's max |value|
(`G1 PASS`); `y`, `x_interp` and the coordinates identical; the probe shows 30 levels, 59 calls, S_churn 0.
If G1 fails: STOP. Report the table, both `RUNTIME` lines (torch versions: a different torch build can change the
CUDA normal stream, which moves the whole draw, not one ulp) and the probe line. Options for the owner: (a) a sandbox
whose anemoi-core is c286c1a3 plus a cherry-pick of 27391c1 (samplers file only), then G1 again; (b) run the two
piecewise arms in the certified venv and the custom arms in the sandbox (breaks the one-runtime rule; owner only).

## 4. Stage A submission (GPU: wait for go 2)

```bash
bash $T1D_S/submit_stageA_t1d.sh | tee $W/notes/submit_stageA.txt     # job ids in $W/notes/jobs.tsv
```
Order and dependency graph (5 A100 at once on AC, inside the 10-GPU cap):
```
predict p12m_pw30_c0 s756 ----afterok--> eval s756  --\
predict p12m_pw30_c0 s757000 -afterok--> eval r757  ---\
predict p12m_c0_pw16_s1k -----afterok--> eval pw16  ----+--afterok(all five)--> tc_intensity (box_post_t1d)
predict p12m_st2 ------------afterok--> eval st2   ---/                     \-> spectra v3 (spectra_t1d)
predict p12m_st4 ------------afterok--> eval st4   --/                       \-> read (read_t1d: read_stage1.py)
```
The predictions run in the sandbox with the probe on (print-only). Evaluations, intensity, spectra and read run on
CPU (qos nf) in the certified venv from the same worktree. Evaluators: `tc,texture,wind_extremes,probabilistic,
surface,shape` (all of stage 1; stage 3 ran only `tc` and lost the peak-wind row).

If a prediction job fails, its evaluation and the three final jobs stay pending with reason DependencyNeverSatisfied:
`scancel <its eval id> <tc_intensity id> <spectra id> <read id>` (ids in `$W/notes/jobs.tsv`), report, and on the
owner's word move the failed run root aside (`mv $RR ${RR}_failed_<jobid>`) and resubmit that one chain by hand with the
`sbatch` lines the submit script prints (predict, then its eval `--dependency=afterok:<new predict id>`), then the three
final jobs with `--dependency=afterok:<all five eval ids>`; append every new id to `jobs.tsv`.

## 5. Gate G2, part 2: after the first file of each run lands

```bash
for spec in "p12m_pw30_c0 756" "p12m_pw30_c0 757" "p12m_c0_pw16_s1k 756" "p12m_st2 756" "p12m_st4 756"; do
  read -r A S <<<"$spec"; RR=$(t1d_root $A $S); J=<that run's predict job id>
  python $T1D_S/check_attrs_t1d.py $RR/meta/tc_o320_o1280_$A.sampler.json --log $(ls $W/logs/*_$J.out) $RR/predictions/predictions_*.nc
done | tee $W/notes/gates/g2_attrs.txt
```
Pass: every file `OK` (sampler block = the resolved one, checkpoint_id `551dfd1eb2a649d49f099acd30b59483`, checkpoint
path in that folder, members 1-10), the probe schedule equal to the arm's levels and the calls 59/59/31/35/39. A `BAD`
line: `scancel` that run's prediction and its evaluation, report, wait for the owner.
Information (not a gate): member 1 of the first file of the seed-756 pw30 run against the G1 sandbox draw,
`python $T1D_S/g1_compare.py $(t1d_root p12m_pw30_c0)/predictions/predictions_20230826_step024.nc ${G}_sandbox_$T1D_TAG/$F --member 1`;
expected bit-identical (same seed, same first draw); a difference means the stream order differs between a one-draw
and a full run, which is recorded, not a failure.

## 6. Expected cost

Box: about 60 s per draw at 59 calls on one A100, in proportion to the calls, plus model loading (~5 min per job).

| job | draws | calls | GPU-hours |
|---|---:|---:|---:|
| G1 sandbox + G1 certified | 1 + 1 | 59 | 0.2 |
| p12m_pw30_c0 seed 756 | 100 | 59 | 1.7 |
| p12m_pw30_c0 seed 757000 | 100 | 59 | 1.7 |
| p12m_c0_pw16_s1k | 100 | 31 | 0.9 |
| p12m_st2 | 100 | 35 | 1.0 |
| p12m_st4 | 100 | 39 | 1.1 |
| loading, 5 jobs | | | 0.4 |
| **total** | | | **about 7.0** (stage A 6.7 + gates 0.2) |

Wall time: about 2 h for stage A (the two pw30 runs are the longest), then the CPU jobs (evaluation up to a few hours,
spectra about 1 h, intensity and read minutes). Wall limits: 4 h (pw30), 2.5 h (pw16, st2), 3 h (st4).

## 7. sacct lines to record

```bash
IDS=$(awk -F'\t' 'NR>1 && $7 ~ /^[0-9]+$/ {print $7}' $W/notes/jobs.tsv | paste -sd,)
sacct -X -P -j $IDS --format=JobID,JobName%40,Partition,QOS,State,ExitCode,Submit,Start,End,Elapsed,AllocTRES%80,NodeList | tee $W/notes/sacct_stageA.txt
grep -h "^SE_TC_T1D\|^T1D_" $W/logs/t1d_*.out | tee $W/notes/rc_lines.txt
```
Per prediction run record: State, ExitCode, Elapsed, the `SE_TC_T1D ... wall_s= s_per_draw= peak_mem_mib=` line.

## 8. Bundle

```bash
cd $T1D_CODE && set +u && source $T1D_CERT_VENV/bin/activate && set -u
python $T1D_S/build_bundle_t1d.py          # -> $W/bundle/stageA (refuses if it exists); prints missing files and files > 1 MB
```
Then copy `$W/bundle/stageA/` to docs `epics/fast-generative-downscaling/in-progress/20261007_T1d_sampler_beat_1p2M_results/stageA/`
and add its README there (which commit each job ran, from `dstools=` / `core=` in `logs_rc.txt`). Every commit the
bundle pins must be pushed first (downscaling-tools `claude/project-thread-tpts4t`, anemoi-core 27391c1 and the
sandbox's exp branch).

## 9. Reading notes for whoever reads the bundle

- The read has ONE seed replicate. With one replicate the band is 1.5 x a single deviation, narrower and noisier than
  T1's two-replicate band. Dry run on the T1 stage 1 bundle: `c0_pw16_s1k` is "inside on all judged metrics" with the
  replicates s757 + s758 and "outside on surface (2 metrics)" with s757 alone. Read the flags with the band table.
- The read lists the replicate itself as an arm (`p12m_pw30_c0_r757`): it must read "inside" (it set the band).
- tc and shape are recorded, not judged (T1 section 6); with `--only tc` a read would say "inside" on nothing.

## 10. AG route (`T1D_HOST=ag`): GH200 on Atos AG, CPU jobs on AC

Why: AC is above its 10-GPU cap (start ~14 h out); AG has GH200s free under its cap, and the 1.2M parent's own reads of
5 Oct ran on AG in `~/dev/.ds-ag-260616`. `tc_o320_o1280` allows predict on atos_ag. Everything not listed here is as
in sections 0-9 (lanes, run roots, gates' pass rules, evaluators, bundle).

**Runtime on AG (the documented rule).** `runbook-experiment-sandbox.md` and `exp.sh`: the uv layer of v2 sandboxes is
validated on x86 only; an experiment that must run on AG/GH200 uses the grandfathered overlay pattern and says so in
`EXPERIMENT.md`. So:
- Build the sandbox exactly as in section 0, on hpc-login (x86): `exp new t1d-stagea`, anemoi-core reset to 27391c1.
  Its git worktrees are architecture-independent; its `.venv-x86_64` is not used on AG. Do NOT run `exp new` or
  `uv sync` on an AG node (exp.sh warns the aarch64 sync may fail; a `.venv-aarch64` costs ~6k inodes and is unvalidated).
- Add one line to `$T1D_SANDBOX/EXPERIMENT.md`: "AG/GH200 runs (T1d stage A) use the grandfathered overlay: certified
  arm venv ~/dev/.ds-ag-260616 + PYTHONPATH onto code/anemoi-core/{training,models,graphs}/src and
  code/anemoi-inference/src, with an import guard; runbook aarch64 clause." Commit it on the exp branch.
- `se_tc_predict_t1d.sbatch` does this when `T1D_HOST=ag`: sandbox = `.ds-ag-260616` + that PYTHONPATH + a guard that
  refuses unless anemoi.models/training/graphs/inference and the samplers module resolve inside the sandbox (the stage 3
  AG guard), then checks anemoi-core HEAD = 27391c1; certified = `.ds-ag-260616` alone (PYTHONPATH unset). It refuses
  to start if `T1D_HOST` and the node's architecture disagree (ag = aarch64).
- Check once on an AG node (interactive or the G1 log's GUARD lines): the overlay imports and
  `cd $T1D_CODE && python -c "import eval.predict.main"` work under `.ds-ag-260616`.

**Resources per job (passed on the sbatch command line by the submit scripts, overriding the AC header):**
`--partition=gpu --qos=ng --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=32 --mem=120G`
(one GH200 = a quarter node: 72 Grace cores, ~120 GB LPDDR; the stage 3 AG jobs used 32 CPUs per GPU). Wall limits as on
AC (4 h pw30, 2.5 h pw16/st2, 3 h st4, 45 min gates). If `--mem=120G` is refused by AG's limits, use
`--mem=0` only with a whole node; otherwise lower to 100G and record it.

**Commands (AG halves on ag-login, AC half on hpc-login; same `$W`, same worktree, shared filesystems):**
```bash
export T1D_HOST=ag
# G3 (no go): on ag-login
TEST=1 bash $T1D_S/submit_gates_t1d.sh; TEST=1 bash $T1D_S/submit_stageA_t1d.sh     # 2 + 5 "TEST: sbatch" lines
# on hpc-login (AC half, no go): TEST=1 SKIPCHECK=1 bash $T1D_S/submit_post_t1d.sh    # 8 "TEST: sbatch" lines
# G1 (go 1): on ag-login; both gate jobs on AG, sandbox overlay vs certified .ds-ag-260616, same GH200 class
bash $T1D_S/submit_gates_t1d.sh
python $T1D_S/g1_compare.py $(t1d_g1_root sandbox)/predictions/predictions_20230826_step024.nc \
                            $(t1d_g1_root certified)/predictions/predictions_20230826_step024.nc | tee $W/notes/gates/g1_compare_ag.txt
#   (+ check_attrs_t1d.py --members 1 --log $W/logs/t1d_g1_ag_sandbox_<jobid>.out, as in section 3)
# Stage A predictions (go 2): on ag-login; five GH200 at once (6 free under the AG cap)
bash $T1D_S/submit_stageA_t1d.sh                     # submits ONLY the five predictions; prints the next step
# G2 part 2 as section 5 (logs are t1d_pred_ag_<arm>_<jobid>.out; the G1 member-1 check uses $(t1d_g1_root sandbox))
# When all five are COMPLETED: on hpc-login (AC; Slurm cannot chain afterok across clusters)
bash $T1D_S/submit_post_t1d.sh                       # refuses unless each run has 10 files and an rc=0 SE_TC_T1D line
#   -> five evaluations on AC (no dependency), then intensity, spectra, read afterok on the five
# sacct: AG ids on ag-login, AC ids on hpc-login
awk -F'\t' 'NR>1 && $8=="ag" && $7 ~ /^[0-9]+$/ {print $7}' $W/notes/jobs.tsv | paste -sd, | xargs -I{} sacct -X -P -j {} \
  --format=JobID,JobName%40,Partition,QOS,State,ExitCode,Submit,Start,End,Elapsed,AllocTRES%80,NodeList > $W/notes/sacct_ag.txt   # on ag-login
```
`build_bundle_t1d.py` (hpc-login) runs sacct for the AC ids itself and copies every `$W/notes/sacct_*.txt`; the
timings table gets a `cluster` column. A failed AG prediction: nothing on AC is queued yet; move its root aside
(`mv $RR ${RR}_failed_<jobid>`), resubmit that one prediction on ag-login with the sbatch line the script printed, and
run `submit_post_t1d.sh` when all five are complete.

**AG job table (stage A + gates; GPU-hours: A100 figure = upper bound, GH200 expected about 0.5-0.7 of it).**

| job | cluster, resources | after | GPU-h upper (A100) | GPU-h expected (GH200) |
|---|---|---|---:|---:|
| t1d_g1_ag_sandbox, t1d_g1_ag_certified (1 draw each) | AG, 1 GH200, 32 CPU, 120G, 45 min | go 1 | 0.2 | ~0.15 |
| t1d_pred_ag_p12m_pw30_c0 (s756), ..._r757 | AG, 1 GH200 each, 4 h | G1 pass + go 2 | 1.7 + 1.7 | ~1.0 + 1.0 |
| t1d_pred_ag_p12m_c0_pw16_s1k, _st2, _st4 | AG, 1 GH200 each, 2.5/2.5/3 h | go 2 | 0.9 + 1.0 + 1.1 | ~0.55 + 0.6 + 0.7 |
| loading, 5 jobs | | | 0.4 | ~0.3 |
| t1d_eval_* (5) | AC CPU, qos nf, 8 CPU, 64G | all 5 AG predictions COMPLETED (manual) | - | - |
| tc_intensity, spectra_v3, read | AC CPU, qos nf | afterok on the 5 evals | - | - |
| **total GPU** | | | **6.9** | **~4.3** |

The GH200 factor is an estimate (no T1d draw on GH200 yet); the G1 sandbox log's `s_per_draw` (one draw, includes no
loading) and the first prediction's wall time give the measured figure; report it with the go-2 request.
