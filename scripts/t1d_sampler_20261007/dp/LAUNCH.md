# LAUNCH: T1d dense-trajectory diagnostic on the 1.2M parent (Atos)

For the executing session on the owner's Mac (Remote Control, `/Users/ecm5702/doc/docs`). Commands run on Atos
(hpc-login for CPU and AC, ag-login for AG). **Every GPU submission below (step 3 smoke, step 4 full, step 6
verification) waits for Joffrey's typed go in that session.** Nothing here has run yet; the CPU parts were tested in
the cloud thread (section 8).

Code: downscaling-tools branch `claude/project-thread-tpts4t` (this directory). Runtime: the SAME as stage A
(`../t1d_env.sh`: sandbox `exp new t1d-stagea` with anemoi-core `27391c1` = custom schedules on hres-lead `00472fb6f`;
on AG the sandbox's own `.venv-aarch64`, owner's option 1 of 2026-10-07, through the same guarded `activate.sh` as
`.venv-x86_64` on AC; `T1D_RUNTIME=overlay` = the older certified-arm-venv overlay, opt-in only). **Stage A's gate G1 is this diagnostic's runtime gate: do not
submit step 3 before G1 has passed on the cluster you use.** Checkpoint: the 1.2M parent, NOT the donor 12dcefea:
`/home/ecm5702/perm/checkpoints/o320_o1280/551dfd1eb2a649d49f099acd30b59483/inference-anemoi-by_step-epoch_171-step_400000.ckpt`.

## Route and cost

The trajectory tool runs the global checkpoint on the regional Franklin-Idalia cut graph (10-40N, 100-58W, hidden
halo 1, the `tc_o320_o1280` lane's `local_scope`) on ONE GPU (`--local-scope-json`). Chosen over the global graph
sharded on 4 A100 (the path of jobs 39291117/39291122):

| route | s per call | GPU-s per call | 16 draws x 479 calls | with loads (4 jobs) |
|---|---:|---:|---:|---:|
| box, cut graph, 1 GPU (chosen) | about 1.1 (A100) | 1.1 | 2.3 GPU-h | **about 3 GPU-h** (limits: 2 h x 4 jobs = 8 GPU-h max) |
| global, sharded, 4 A100 (fallback, AC only) | about 3 | 12 | 25.5 GPU-h | about 28 GPU-h (too close to the 30 GPU-h cap) |

Cluster: `T1D_HOST=ac` (default, A100) or `T1D_HOST=ag` (GH200; owner's decision for stage A: AG is under the cap, AC
over it). On AG the per-call time is not measured; expect it no slower than A100. Smoke about 0.1 GPU-h; verification
(step 6) about 1.6 GPU-h. Total about 5 GPU-h.

Draws: date 20230826 at leads 024 and 120 h, date 20230828 at leads 024 and 096 h, member 01, four seeds each (16
draws): seeds 1000-1003 (0826/024), 1010-1013 (0826/120), 1020-1023 (0828/024), 1030-1033 (0828/096). The 0828 draws
use +96 h, not +120 h: on 2 Sep Idalia sits near 32N 295E, within 0.5 deg of the window bound the cut edge forces, and
that window's 290-295.6E part could hold Franklin. The date hold-out is unchanged (fit on 0826, test on 0828 and back). Schedule: 240 log-uniform levels
1e5 -> 0.03 as `schedule_type: custom` (`schedules/dense240.json`, `--num-steps 240`, 479 Heun calls per draw), churn
off (`{"sampler":"heun","S_churn":0.0,"S_noise":1.0}`), fp32; `--lockin` on.

Box: a 500 km disc around IDALIA: the truth-msl minimum inside a per-bundle window (`common.IDALIA_WINDOWS`) that
excludes Franklin (about 289-295E on 27-31 Aug) and keeps the disc at least 1 deg inside the cut graph:

| bundle (valid time) | window lat, lon (E) | Idalia then (NHC) |
|---|---|---|
| 0826 + 24 h (27 Aug 00Z) | 17-26N, 270-280E | TD near 20.5N 86W (274E), NW Caribbean |
| 0826 + 120 h (31 Aug 00Z) | 27-34N, 272-285E | near 32.5N 80W (280E), SE US coast |
| 0828 + 24 h (29 Aug 00Z) | 19-28N, 270-280E | near 23N 85W (275E) |
| 0828 + 96 h (1 Sep 00Z) | 27-34N, 284-293E | near 31N 70W (290E); Franklin north of 40N, outside the cut |

The 27 Aug centre (about 20.5N) falls outside a single "25-33N, 273-290E" range; the per-bundle windows follow the
track. `locate_storms` (step 2) checks all four on the truth before any GPU use.

Output: `/home/ecm5702/hpcperm/t1d_traj_20261007/` (about 3.2 GB in about 60 files, plus about 0.1 GB of
verification). Per draw `trajectory_states_s<seed>.npz` about 195 MB.

## 1. Code and runtime

Stage A's section 0 builds the sandbox (`exp new t1d-stagea`, anemoi-core reset to 27391c1) and its worktree
`$T1D_W/code/downscaling-tools`. The diagnostic uses that sandbox unchanged and its own detached worktree of the
same branch at the commit that has `dp/` (so stage A's recorded `dstools=` does not move):

```bash
source /home/ecm5702/work-t1d-20261007/code/downscaling-tools/scripts/t1d_sampler_20261007/t1d_env.sh
git -C $T1D_CODE fetch <fork remote> claude/project-thread-tpts4t      # https fetches fail on the login nodes
git -C $T1D_CODE worktree add --detach $T1D_W/code/downscaling-tools-diag FETCH_HEAD
export T1D_CODE=$T1D_W/code/downscaling-tools-diag            # every diagnostic command below uses this
git -C $T1D_CODE rev-parse HEAD | tee $T1D_W/notes/dstools_diag_sha.txt
export T1D_OUT=/home/ecm5702/hpcperm/t1d_traj_20261007 T1D_HOST=ag   # or ac; same value as stage A uses
ls $T1D_SANDBOX/activate.sh && git -C $T1D_SANDBOX/code/anemoi-core rev-parse HEAD   # 27391c1597e8... (sandbox git: ag-login only)
```
Keep `T1D_CODE`, `T1D_OUT` and `T1D_HOST` set in the shell that runs `submit_traj.sh`. **Atos sets `SBATCH_EXPORT=NONE`:
sbatch does NOT pass that environment to the job.** `submit_traj.sh` therefore writes every setting the job needs on each
sbatch line (`--export=ALL,T1D_HOST=..,T1D_CODE=..,T1D_SANDBOX=..,T1D_RUNTIME=..,T1D_INPUT_ROOT=..,ROUTE=..,T1D_OUT=..`
plus `T1D_BUNDLES` and `T1D_WINDOW` when set; the window travels as `T1D_WINDOW_COLON`, since --export splits on commas).
`TEST=1 bash .../submit_traj.sh smoke|full|verify` prints each full sbatch line and runs it with `--test-only`; read the
export list there before every real submission. `cpu_dp.sbatch` takes the output root (and VERIFY) as arguments.
Atos traps: system `python3` is 3.6 (use a venv's python); sandbox git commands work only from ag-login; `~/.local/bin`
uv and Python are x86; https fetches fail on the login nodes (use the fork remote); `$HPCPERM` inode margin is about
9,900 (outputs under `T1D_OUT` count: prefer scratch if the npz states are many files).

## 2. CPU checks (hpc-login; no GPU)

```bash
cd $T1D_CODE && set +u && source $T1D_SANDBOX/activate.sh && set -u       # x86 sandbox venv on hpc-login
python -c "import interp.tools.trajectory, eval.predict.graph_cut, scipy, netCDF4, xarray; print('imports ok')"
python -m interp trajectory --help | grep -E -- "--save-trajectory-states|--local-scope-json|--trajectory-states-stride"
python scripts/t1d_sampler_20261007/dp/tests/test_dp.py            # about 10 s; includes the trajectory argv check
python -m scripts.t1d_sampler_20261007.dp.locate_storms --bundle-dir $T1D_INPUT_ROOT   # all four PASS, or stop
deactivate
```
`locate_storms` prints each bundle's centre, truth msl minimum and disc, and FAILs when the disc comes within 1 deg of
the cut edges (10-40N, 260-302E) or the centre sits on its window edge (the minimum was cut off: another low, or the
storm outside the window). On a FAIL, adjust that bundle's window in `common.IDALIA_WINDOWS` (commit, push, re-fetch)
or, for a test only, `export T1D_WINDOW=lat0,lat1,lon0,lon1`. Pre-register in the note before step 3: expectation (T1c
section 6) msl lock-in sigma50 above 300; fewer than half of c0_30's 19 intervals below sigma 10 carry measurable cost.

## 3. Smoke: one draw, 12 levels (23 calls). GPU: wait for the owner's typed go (and stage A's G1 PASS)

```bash
bash $T1D_CODE/scripts/t1d_sampler_20261007/dp/jobs/submit_traj.sh smoke
# when it ends:
tail -n 40 $T1D_OUT/logs/t1d_*smoke12_*.out | grep -E "RUNTIME|GUARD|host=|local cut graph|storm box|saved|T1D_TRAJ"
cd $T1D_CODE && python -m scripts.t1d_sampler_20261007.dp.check_states $T1D_OUT/smoke12_d20230826_l024 \
  --schedule scripts/t1d_sampler_20261007/dp/schedules/dense12.json --project-levels 240 --project-draws 16
```
Gate (all must hold before step 4): rc 0; every `check_states` line PASS: 23 calls, first-evaluation sigmas = the
custom list, second evaluations at the next level, finite arrays, final = last D, projected total under 30 GB, the box
centre and cell lat/lon range printed with the disc at least 1 deg inside the cut and the centre strictly inside the
Idalia window, the disc not truncated, `trajectory.json` with `schedule_type: custom`, `local_scope` recorded and a
lock-in payload. The log shows the cut graph active (data nodes about 1.8e5 of 6.6e6). Record the job id, wall time
and peak memory (`sacct -j <id> -o JobID,Elapsed,MaxRSS,State`).

If the cut graph fails (an error at `activate_local_graph_cut` or a size mismatch), stop and report; the fallback is
`ROUTE=global` on AC (4 A100 sharded, about 28 GPU-h for 16 draws), which needs a fresh go with that cost stated.

## 4. Full: 16 draws, 240 levels. GPU: wait for the owner's typed go

```bash
bash $T1D_CODE/scripts/t1d_sampler_20261007/dp/jobs/submit_traj.sh full      # 4 jobs x 1 GPU, about 40 min each on A100
cat $T1D_OUT/jobs.txt
# when they end:
grep -h T1D_TRAJ $T1D_OUT/logs/t1d_*d08*_l*.out
cd $T1D_CODE && for d in $T1D_OUT/d2023*_l*; do echo $d; python -m scripts.t1d_sampler_20261007.dp.check_states $d \
  --schedule scripts/t1d_sampler_20261007/dp/schedules/dense240.json | grep -E "FAIL|box centre"; done
du -sh $T1D_OUT
```
Expected: 4 x 4 files of about 195 MB, no FAIL line, about 3 GPU-h in total.

## 5. CPU: cost matrices, DP, lock-in, verification schedule file (hpc-login; no GPU, no go needed)

```bash
source $T1D_CODE/scripts/t1d_sampler_20261007/t1d_env.sh     # for t1d_export (SBATCH_EXPORT=NONE on Atos)
sbatch "$(t1d_export)" --output=$T1D_OUT/logs/%x_%j.out $T1D_CODE/scripts/t1d_sampler_20261007/dp/jobs/cpu_dp.sbatch $T1D_OUT
# about 20-30 min (cost_matrix about 4 min per draw per core, 4 workers); certified x86 venv, as stage A's CPU jobs
cat $T1D_OUT/dp/dp_summary.md; python -m json.tool $T1D_OUT/lockin/lockin_sigma50.json | head -40
python -m json.tool $T1D_OUT/verify/schedules_verify.json | grep -E '"calls"|gpu_h'
```

## 6. Verification draws (batch 2). GPU: wait for the owner's typed go

Runs the candidate schedules of `verify/schedules_verify.json` (DP fits on all draws for C_L2 and C_band at K = 8, 10,
12, 16, identical ones merged; c0_30 and c0_pw16_s1k on their nominal levels; log-uniform 16) on the same 16 bundles and
seeds, one draw each, the model loaded once per job. 297 calls per draw if all 11 differ, about 1.5 GPU-h plus 4 loads
(under 0.2 GPU-h per 16-step schedule, about 0.3 for c0_30). The tool seeds `torch.manual_seed(seed)` right before
`sample()`, whose first draw is `y_init = randn(shape) * sigma_max`, so the unit noise is the dense run's at equal seed;
`analyze` checks it on the saved call-0 inputs and stops if it differs.

```bash
bash $T1D_CODE/scripts/t1d_sampler_20261007/dp/jobs/submit_traj.sh verify $T1D_OUT/verify/schedules_verify.json   # 4 jobs x 1 GPU
# when they end (CPU, from hpc-login):
sbatch "$(t1d_export)" --output=$T1D_OUT/logs/%x_%j.out $T1D_CODE/scripts/t1d_sampler_20261007/dp/jobs/cpu_dp.sbatch $T1D_OUT 1
cat $T1D_OUT/verify/verify_table.md
```
The table: schedules x (calls, draws, C_L2 run, C_band run, predicted L2, predicted band, ratio L2, ratio band, same
noise), plus the log-log correlation of run against predicted. Run error = final state against the dense 240-level
final state of the same seed, per variable and band with the cost's normaliser. On the linear toy the ratio was 1.0 to
3.1 with log-log correlation 0.993; a much lower correlation here means the summed cost does not rank schedules on
the real model and the DP schedules should not go to the screen without that caveat.

## 7. Bundle for docs

Copy `$T1D_OUT/bundle_for_docs/diag/` (a few MB) to
`docs/epics/fast-generative-downscaling/in-progress/20261007_T1d_sampler_beat_1p2M_results/diag/`:

```
diag/
  jobs.txt                         job names, ids, cluster and arguments (smoke, 4 full, 4 verify)
  logs/<job>.out                   last 60 lines of every SLURM log
  cost/cost_manifest.json          draws, sets, grid, dx, Nyquist, box size, fixed weights
  cost/cost_C_all.npz              C_L2_*, C_band_*, C_band300_* (euler, heun_ref), trunc_C_*; float32, 240 x 240
  cost/cost_C_20230826.npz         same, fit set 1
  cost/cost_C_20230828.npz         same, fit set 2
  dp/dp_schedules.json             every DP and reference schedule: sigmas, ln-steps, calls, noise_scheduler block,
                                   costs in-sample / hold-out / all, per-variable-band costs
  dp/dp_summary.md                 the tables
  dp/c0_30_running_cost.json       c0_30 interval costs, cumulative sum, share below sigma 10
  dp/schedules/dp_<cost>_K<k>_fit<set>.json   ready `schedule_type: custom` blocks (for lanes)
  lockin/lockin_sigma50.json       per-variable anomaly lock-in sigma50 (trajectory.json source)
  lockin/lockin_sigma50_from_states.json   same rule from the saved states
  verify/schedules_verify.json     the batch-2 candidates and their GPU cost
  verify/verify_table.md, .json    run error against predicted cost (after step 6)
```
The raw states (`d*/trajectory_states_s*.npz`, `verify/*/d*/...`), `trajectory.json` files and per-draw cost matrices
stay in `$T1D_OUT` on Atos HPCPERM; list their paths in the note. Do not copy the npz states into docs.

## 8. What was tested in the cloud thread (no cluster)

- `py_compile` of `interp/tools/trajectory.py` and every file here; `tests/test_dp.py`: DP against brute force; equal
  spacing on a convex cost; the real writer on fake arrays read back; the numpy piecewise schedules equal to the fork's
  scheduler; the custom JSON accepted by the fork's CustomScheduler with num_steps 240 and sigma_min 0.03; both
  capture_denoiser call forms; the verification `run` argv parsed by the trajectory tool's own parser; cost_matrix ->
  dp_schedule -> verify make/analyze -> lockin_read end to end on four fake draws over two dates. The dense job's
  argv was parsed by the tool's parser as well.
- `tests/toy_linear.py`: the fork's EDMHeunSampler (27391c159) through the real capture and writer on a
  linear-Gaussian toy, 240 levels, 479 calls, bookkeeping exact; numbers in README.
- Not testable here: the cut-graph path on the real checkpoint (including its reuse across schedules in
  `verify_schedules run`), the bundle reads (`locate_storms`), the GPU timings, the AG `.venv-aarch64` route (and the overlay opt-in), the sharded fallback
  with `--save-trajectory-states`.
