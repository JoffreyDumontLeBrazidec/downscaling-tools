# LAUNCH: T1d dense-trajectory diagnostic on the 1.2M parent (Atos)

For the executing session on the owner's Mac (Remote Control, `/Users/ecm5702/doc/docs`), commands run on
Atos AC. **Every GPU submission below (step 3 smoke, step 4 full) waits for Joffrey's typed go in that
session.** Nothing here has run yet; the CPU parts were tested in the cloud thread (section 7).

Code: downscaling-tools branch `claude/project-thread-tpts4t` (this directory), anemoi-core
`claude/project-thread-ncf9gh` = `27391c159` (custom schedules; EDMHeunSampler unchanged) on hres-lead
`00472fb6f`. Checkpoint: the 1.2M parent, NOT the donor 12dcefea:
`/home/ecm5702/perm/checkpoints/o320_o1280/551dfd1eb2a649d49f099acd30b59483/inference-anemoi-by_step-epoch_171-step_400000.ckpt`.

## Route and cost

The trajectory tool now runs the global checkpoint on the regional Franklin-Idalia cut graph
(10-40N, 100-58W, hidden halo 1, exactly the `tc_o320_o1280` lane's `local_scope`) on ONE A100
(`--local-scope-json`, single-GPU, dict-API; the truth and the hres coordinates are cut with the model's
data mask). Chosen over the global graph sharded on 4 A100 (the tool's path of jobs 39291117/39291122):

| route | s per call | GPU-s per call | 16 draws x 479 calls | with loads (4 jobs) |
|---|---:|---:|---:|---:|
| box, cut graph, 1 A100 (chosen) | about 1.1 | 1.1 | 2.3 GPU-h | **about 3 GPU-h** (time limit 2 h x 4 jobs = 8 GPU-h max) |
| global, sharded, 4 A100 (fallback) | about 3 | 12 | 25.5 GPU-h | about 28 GPU-h (too close to the 30 GPU-h cap; use 3 seeds per bundle if needed) |

Smoke: 23 calls plus the model load, about 0.1 GPU-h. Total expected about 3.2 GPU-h.

Draws: dates 20230826 and 20230828, leads 024 and 120 h, member 01, four seeds each (16 draws):
seeds 1000-1003 (0826/024), 1010-1013 (0826/120), 1020-1023 (0828/024), 1030-1033 (0828/120). Schedule:
240 log-uniform levels 1e5 -> 0.03 as `schedule_type: custom` (`schedules/dense240.json`, `--num-steps 240`,
479 Heun calls per draw), churn off (`{"sampler":"heun","S_churn":0.0,"S_noise":1.0}`), fp32. Box: 500 km disc
around the deepest msl minimum inside the window 15-35N, 266-296E (5-6 deg inside the cut region, so the
disc never touches the cut edge); `--lockin` on (per-variable lock-in curves from the same run).

Output: `/home/ecm5702/hpcperm/t1d_traj_20261007/` (about 3.2 GB in about 60 files: HPCPERM is inode-bound,
this is bytes-light in files). Per draw `trajectory_states_s<seed>.npz` about 195 MB.

## 1. Sandbox (experiment-isolation v2, RUNTIME-LAYOUT.md), on hpc-login (x86)

Same recipe as stage A (`../LAUNCH.md` section 0): a fresh `exp new` folder whose uv venv is synced from the
pristine lock, with only anemoi-core moved to the patched fork commit; a separate slug so the diagnostic has its
own folder. If `quota` shows fewer than about 10k free inodes on HPCPERM, reuse the stage-A folder instead
(`20261007-t1d-stagea`, same code: anemoi-core 27391c1) and record that in the note.

```bash
quota                                                  # HPCPERM is inode-bound: a uv venv costs about 6k inodes
exp new t1d-traj                                       # -> ~/hpcperm/sandbox/<YYYYMMDD>-t1d-traj (uv sync --frozen, about 1 min)
SB=$HOME/hpcperm/sandbox/$(date -u +%Y%m%d)-t1d-traj; echo "SB=$SB"   # record it in the note
C=$SB/code/anemoi-core
git -C $C fetch https://github.com/JoffreyDumontLeBrazidec/anemoi-core claude/project-thread-ncf9gh
git -C $C reset --hard 27391c1597e8090b6e0da9a9ce79352eaa6c6530    # samplers patch on hres-lead 00472fb6f
git -C $C merge-base --is-ancestor c286c1a3 HEAD && echo "c286c1a3 (the core the 1.2M checkpoint records) is an ancestor"
git -C $C push fork HEAD:exp/t1d-traj-$(date -u +%Y%m%d)    # best effort; 27391c1 itself is already on the fork
# downscaling-tools at the pushed branch, inside the sandbox folder
git -C ~/dev/downscaling-tools fetch https://github.com/JoffreyDumontLeBrazidec/downscaling-tools claude/project-thread-tpts4t
git -C ~/dev/downscaling-tools worktree add --detach $SB/downscaling-tools FETCH_HEAD
git -C $SB/downscaling-tools rev-parse HEAD            # record: every job logs it as dstools=
```
If the Atos downscaling-tools clone is elsewhere than `~/dev/downscaling-tools`, use that clone (any clone works:
the worktree is detached at the fetched commit).

## 2. CPU checks (login node; no GPU)

```bash
source $SB/activate.sh && cd $SB/downscaling-tools
python -c "import anemoi.models.samplers.diffusion_samplers as d; assert 'custom' in d.NOISE_SCHEDULERS; print(d.__file__)"
python -c "import interp.tools.trajectory, eval.predict.graph_cut, scipy, netCDF4; print('imports ok')"
# if an import fails inside the sandbox venv: STOP and report (a dependency change is a uv lock edit on the
# exp branch, the owner's call; never an untracked pip install)
python -m interp trajectory --help | grep -E -- "--save-trajectory-states|--local-scope-json|--trajectory-states-stride"
python scripts/t1d_sampler_20261007/dp/tests/test_dp.py      # against the sandbox's own fork samplers; about 10 s
B=/ec/res4/scratch/ecm5702/eval/o320_o1280/manual_731d203a_pristine_20260818/bundles_with_y   # stage A's T1D_INPUT_ROOT
for d in 20230826 20230828; do for s in 024 120; do ls $B/*date${d}*mem01*step${s}h*input_bundle.nc; done; done
ls -la /home/ecm5702/perm/checkpoints/o320_o1280/551dfd1eb2a649d49f099acd30b59483/inference-anemoi-by_step-epoch_171-step_400000.ckpt
```
If a bundle is missing there, use the o320_o1280 lane's root `/home/ecm5702/hpcperm/data/input_data/o320_o1280/idalia`
via `export T1D_BUNDLES=...` before submitting.
Pre-register in the note before step 3: expectation (T1c section 6) msl lock-in sigma50 above 300; fewer than
half of c0_30's 19 intervals below sigma 10 carry measurable cost.

## 3. Smoke: one draw, 12 levels (23 calls). GPU: wait for the owner's typed go

```bash
export T1D_SB=$SB T1D_OUT=/home/ecm5702/hpcperm/t1d_traj_20261007
bash $SB/downscaling-tools/scripts/t1d_sampler_20261007/dp/jobs/submit_traj.sh smoke     # 1 x A100, AC, qos ng
# when it ends:
tail -n 30 $T1D_OUT/logs/t1d_smoke12_*.out
grep -E "local cut graph active|storm box|saved .* calls" $T1D_OUT/logs/t1d_smoke12_*.out
python -m scripts.t1d_sampler_20261007.dp.check_states $T1D_OUT/smoke12_d20230826_l024 \
  --schedule scripts/t1d_sampler_20261007/dp/schedules/dense12.json --project-levels 240 --project-draws 16
```
Gate (all must hold before step 4): rc 0; every `check_states` line PASS (23 calls, first-evaluation sigmas =
the custom list, second evaluations at the next level, finite arrays, final = last D, projected total under
30 GB); the log shows the cut graph active (data nodes about 1.8e5 of 6.6e6) and the storm box inside the
window; `trajectory.json` has `schedule_type: custom` and a lock-in payload. Record the job id, wall time and
peak memory (`sacct -j <id> -o JobID,Elapsed,MaxRSS,State`).

If the cut graph fails (any error at `activate_local_graph_cut` or a size mismatch), stop and report; the
fallback is `ROUTE=global` (4 A100 sharded, about 28 GPU-h for 16 draws), which needs a fresh go with that
cost stated.

## 4. Full: 16 draws, 240 levels. GPU: wait for the owner's typed go

```bash
bash $SB/downscaling-tools/scripts/t1d_sampler_20261007/dp/jobs/submit_traj.sh full      # 4 jobs x 1 A100, about 40 min each
cat $T1D_OUT/jobs.txt
# when they end:
grep T1D_TRAJ $T1D_OUT/logs/t1d_d08*.out
for d in $T1D_OUT/d2023*_l*; do python -m scripts.t1d_sampler_20261007.dp.check_states $d \
  --schedule scripts/t1d_sampler_20261007/dp/schedules/dense240.json --project-levels 240 --project-draws 16 | grep -c PASS; done
du -sh $T1D_OUT
```
Expected: 4 x 4 files of about 195 MB; about 3 GPU-h in total.

## 5. CPU: cost matrices, DP, lock-in (no GPU, no go needed)

```bash
sbatch --output=$T1D_OUT/logs/%x_%j.out $SB/downscaling-tools/scripts/t1d_sampler_20261007/dp/jobs/cpu_dp.sbatch
# about 20-30 min: cost_matrix (about 4 min per draw per core, 4 workers), dp_schedule (seconds), lockin_read
cat $T1D_OUT/dp/dp_summary.md; python -m json.tool $T1D_OUT/lockin/lockin_sigma50.json | head -40
```

## 6. Bundle for docs

Copy `$T1D_OUT/bundle_for_docs/diag/` (a few MB) to
`docs/epics/fast-generative-downscaling/in-progress/20261007_T1d_sampler_beat_1p2M_results/diag/`:

```
diag/
  jobs.txt                         job names, ids and arguments (smoke + 4 full)
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
```
The raw states (`d*/trajectory_states_s*.npz`), `trajectory.json` files and per-draw cost matrices stay in
`$T1D_OUT` on Atos HPCPERM; list their paths in the note. Do not copy the npz states into docs.

## 7. What was tested in the cloud thread (no cluster)

- `py_compile` of `interp/tools/trajectory.py` and every file here; `tests/test_dp.py` (DP vs brute force;
  equal spacing on a convex cost; the real writer on fake arrays read back; the numpy piecewise schedules
  equal to the fork's scheduler; the custom JSON accepted by the fork's CustomScheduler with num_steps 240 and
  sigma_min 0.03; capture_denoiser two- and three-argument calls; cost_matrix -> dp_schedule -> lockin_read
  end to end on four fake draws over two dates).
- `tests/toy_linear.py`: the fork's EDMHeunSampler (27391c159) through the real capture and writer on a
  linear-Gaussian toy, 240 levels, 479 calls, bookkeeping exact; see README for the numbers.
- Not testable here: the cut-graph path inside the trajectory tool on the real checkpoint, the bundle reads,
  the GPU timings, the sandbox build, the sharded fallback with `--save-trajectory-states`.
