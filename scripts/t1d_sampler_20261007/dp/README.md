# T1d dense-trajectory diagnostic: cost matrices and DP schedules

Plan: T1c (`completed-tasks/20261002_T1c_step_schedule_literature_review.md`) sections 3d and 6, ranks 1-2:
dense churn-off reference trajectories on the Franklin-Idalia box, the matrix of one-step errors between
every pair of dense levels, and GITS-style (2405.11326) / optimal-stepsize-distillation-style
(2503.21774) dynamic-programming schedules from it. Model: the 1.2M parent `551dfd1e` step 400k.
Launch: `LAUNCH.md`.

## Inputs

`interp/tools/trajectory.py --save-trajectory-states` (new flag, default off) writes one
`trajectory_states_s<seed>.npz` per seed, rank 0, right after that seed finishes (memory holds one seed):

| key | shape, dtype | content |
|---|---|---|
| `x_in`, `D` | (n_calls, V, n) float32 | denoiser input x_t and output D(x_t), NORMALISED residual units (the sampler's state space) |
| `sigma`, `call_idx`, `step_idx`, `heun_eval` | (n_calls,) | sigma of the call; call index; Heun step; 1 = first evaluation (on the sampler state at its own level), 2 = second (on the Euler-predicted point at the next level) |
| `final`, `truth_residual` | (V, n) float32 | the sampler's output; the true residual |
| `vars`, `var_out_index` | (V,) | surface targets present (10u, 10v, 2t, msl, and tp when in the schema) |
| `lat`, `lon`, `box_rows`, `stride`, `n_box_full` | (n,) | box cells (500 km disc around the storm), index into the (cut) grid, fixed stride (1 = all) |
| `meta_*` | scalars | seed, checkpoint, bundle, centre, radius, local scope, scheduler and sampler JSON, num_steps |

float32, not float16 as first planned: at sigma 1e5 the state exceeds the float16 range (65504), and
float16's relative rounding (5e-4) times sigma_j is an error floor above the true one-step errors for
sigma_j above about 10. Size: 479 calls x 2 x 5 vars x about 10,160 cells x 4 B = about 195 MB per draw,
3.1 GB for 16 draws (cap 30 GB), so no stride is needed.

`--local-scope-json` (new flag, default off) runs the global checkpoint on a regional cut graph on one GPU
(`eval/predict/graph_cut.py`, the `tc_o320_o1280` lane's `local_scope`), cutting the truth and the hres
coordinates with the same data mask; a model already cut for the same scope (reused in-process by
`verify_schedules run`) is not cut twice. Files are written to `<name>.npz.partial` and renamed, so a crash never
leaves a file that matches the readers' `trajectory_states_s*.npz` globs.

Runtime: the GPU jobs source `jobs/_runtime.sh`, i.e. stage A's `t1d_env.sh` and its sandbox runtime (T1D_HOST=ac:
the sandbox uv venv; T1D_HOST=ag: certified arm venv + overlay + import guard). Stage A's gate G1 is the runtime gate.

## Costs (weights FIXED before any read)

For draws d and dense levels i < j (sigma_i > sigma_j), from the reference state x_i to level j against x_j:

- Euler: `x_i + (sigma_j - sigma_i) d_i`, `d_i = (x_i - D_i) / sigma_i`;
- Heun, approximated on the reference (`heun_ref`): trapezoid with `d_i` and `d_j = (x_j - D_j) / sigma_j`.
  The real Heun step evaluates D at the Euler-predicted point, not at x_j. heun_ref drops the Euler
  predictor's error from the end slope (an oracle corrector), so it UNDER-estimates the true one-step Heun
  error, more for long steps. On the linear-Gaussian toy: ratio heun_ref / true about 1.07 for ln-steps
  below 0.3, 0.26-0.33 for longer steps, 0.30 summed along the DP 16-step path. The summed path cost still
  ranks schedules correctly (log-log correlation 0.993 with the run error of the schedules; run / predicted
  1.0-3.1).

Band split: the box is resampled to the v3 regular grid (`plot_sampler_texture_v3.py` Sampler: 0.07 deg,
linear barycentric) over the square inscribed in the 500 km disc (about 690 km a side, Nyquist about 16 km),
2-D Hann window, FFT, sharp cut at 100 km (fine = wavelength < 100 km, coarse = the rest including the box
mean); 300 km cut reported too. Each (variable, band) error energy is divided by the variance of the draw's
final reference state in that (variable, band).

- `C_L2   = mean over {10u, 10v, 2t, msl} of E_total / V_total`
- `C_band = mean over {10u, 10v, 2t, msl} of 0.5 E_fine100 / V_fine100 + 0.5 E_coarse100 / V_coarse100`

tp is computed per (variable, band) but is not in either fixed cost. Per-draw matrices are averaged over the
draws of a set (all; one set per date).

## Schedules

K = sampler steps = positive levels from 1e5 to 0.03 inclusive (K - 1 intervals), plus the terminal zero
the custom scheduler appends: 2K - 1 Heun calls. K = 8, 10, 12, 16 (plan), 20, 29, 30 (comparison; c0_30
has K = 30, 29 intervals, 59 calls; c0_pw16_s1k has K = 16, 31 calls). The last step 0.03 -> 0 is the same
for all and not costed. DP: exact minimum of the summed cost over paths through K dense levels, O(K N^2).
Fits on 20230826, on 20230828 (each evaluated on the other date, hold-out) and on all. Draws: 0826 at +24 h and
+120 h, 0828 at +24 h and +96 h (on 2 Sep, +120 h, Idalia is too close to the cut edge and to Franklin's window),
four seeds each; the date split is the hold-out.

References costed on the same matrices, mapped to the nearest dense level (max ln error 0.031): c0_30
(piecewise 10 + 20, 1e5, transition 10, exponential above, Karras rho 7 below) and c0_pw16_s1k
(5 + 11, sigma_max 1e3), the latter charged with a start-truncation term (paired seeds: the reference state
at its first level against `sigma_s * x_0 / sigma_0`, the noise pw16 would start from with the same seed);
log-uniform K-step baselines. c0_30's running sum and the share of its cost below sigma 10.

## Box

The 500 km disc is centred on Idalia: the truth-msl minimum inside a per-bundle window (`common.IDALIA_WINDOWS`,
following the NHC track and excluding Franklin), with the disc at least 1 deg inside the cut graph.
`locate_storms.py` checks this on the truth before any GPU use; `check_states.py` checks it again on the run.

## Verification (batch 2)

`verify_schedules.py`: `make` writes the candidates (DP fits on all draws, K = 8-16, both costs; c0_30, c0_pw16_s1k
on their nominal levels; log-uniform 16), `run` (GPU, `jobs/verify.sbatch`) runs them on the same bundles and seeds
with the model loaded once, `analyze` compares each final state with the dense run's (same seed, same unit noise,
checked) per variable and band and tabulates run error, predicted summed cost and their ratio.

## Outputs

- `cost/per_draw/cost_<bundle>_s<seed>.npz`: raw energies `E_<method>_<var>_<band>`, `V_<var>_<band>`,
  `C_L2_*`, `C_band_*`, `C_band300_*`, normalised `n_<method>_<var>_<band>`, `trunc_*`, `dense_step_*`
  (Euler-predictor gap of the dense run itself; a convergence check of the reference).
- `cost/cost_mean_<set>.npz`, `cost/cost_manifest.json`.
- `dp/dp_schedules.json` (every schedule: levels, sigmas, ln-steps, calls, `noise_scheduler` block, costs on
  every set, per-variable/band costs), `dp/dp_summary.md`, `dp/c0_30_running_cost.json`,
  `dp/schedules/<name>.json` (the `schedule_type: custom` block for `--noise-scheduler-json` or a lane),
  `dp/bundle/cost_C_<set>.npz` (the fixed-cost matrices only, float32, for the docs bundle).
- `lockin/lockin_sigma50.json`: anomaly lock-in sigma50 per variable (findings-20260805 point 2 rule) from
  trajectory.json, and the same from the states.

## Run

```
python -m scripts.t1d_sampler_20261007.dp.dense_schedule --n 240 --out dense240.json
python -m scripts.t1d_sampler_20261007.dp.check_states <traj out dir> --schedule schedules/dense12.json
python -m scripts.t1d_sampler_20261007.dp.cost_matrix --inputs '<root>/d2023*_l*/trajectory_states_s*.npz' --out-dir <root>/cost
python -m scripts.t1d_sampler_20261007.dp.dp_schedule --cost-dir <root>/cost --out-dir <root>/dp
python -m scripts.t1d_sampler_20261007.dp.lockin_read --inputs '<root>/d2023*_l*/trajectory.json' --out <root>/lockin/lockin_sigma50.json
python -m scripts.t1d_sampler_20261007.dp.locate_storms --bundle-dir <bundle root>
python -m scripts.t1d_sampler_20261007.dp.verify_schedules make|run|analyze ...      # see its docstring
python scripts/t1d_sampler_20261007/dp/tests/test_dp.py        # CPU, about 10 s
python scripts/t1d_sampler_20261007/dp/tests/toy_linear.py      # CPU, about 3.5 min; needs torch and the fork's samplers
```

The tests load the fork's `diffusion_samplers.py` from the installed anemoi-models when it has the custom
scheduler, else from `T1D_FORK_SAMPLERS` or `../anemoi-core`, and the capture/writer functions straight
from `interp/tools/trajectory.py` (so they test the code that runs on the cluster).
