# Few-step big eval, global o96->o320: scoring on Atos

This kit scores the 20 pre-registered rows of
`docs/epics/fast-generative-downscaling/in-progress/20261015_fewstep_big_eval_global_o96_o320.md`
on two case sets:

- set **A**: validation-36, step 60;
- set **B**: Idalia-25.

The aggregator (`aggregate.py`) then writes the read.

| file | role |
|---|---|
| `rows.yaml` | The 20 rows: label, arm, step, sampler, calls, sets and figure group. **VERIFY** the arm directories before submitting. |
| `slim_predictions.py` | Copies the prediction trees, keeping only the weather states the eval reads. |
| `score_row.py` / `score_row.sbatch` | Runs one row on one set: one `eval.cli evaluate --only <evaluator>` per evaluator. Evaluators that already have `.complete` are skipped. |
| `submit_all.sh` | Submits all rows on both sets. Dry run by default. The reference row goes first, and the other rows of a set wait for it (`afterany`). |
| `maps.sh` | Draws the zoom_maps figures: Alps and Idalia storm boxes, one figure group per call count, each with the reference row. |
| `AUDIT.md` | The verdict for each evaluator, with evidence, the fixes, and what cannot run. |
| `aggregate.py`, `aggregate_lib/` | The multi-row read (written by another agent). |

## 0. Code and environment

`~/dev/downscaling-tools` on Atos is an older main. Do not use it or modify it. Make a separate
worktree of this branch instead:

```bash
git -C ~/dev/downscaling-tools fetch origin claude/project-thread-72mhur
git -C ~/dev/downscaling-tools worktree add ~/dev/dt-bigeval origin/claude/project-thread-72mhur
export DT_ROOT=~/dev/dt-bigeval
```

The jobs use the environment that `eval/config/hosts/atos_ac.yaml` declares:

- `module load ecmwf-toolbox` (metview, for tc in regridded mode);
- `source /home/ecm5702/dev/.ds-260612/bin/activate`;
- `PYTHONPATH=$DT_ROOT` only;
- `INTER_MAT_DIR` and `RESIDUAL_STATISTICS_DIR` from the host file.

Make the log directory once: `mkdir -p /home/ecm5702/scratch/eval/_slurm`.

## 1. Slim the predictions (mandatory: scratch has about 3.6 TB free)

```bash
B=/home/ecm5702/scratch/agent-work/20261015-fewstep-bigeval
python $DT_ROOT/scripts/fewstep_bigeval_20261015/slim_predictions.py --src <copied tree of set A> --dst $B/pred/setA
python $DT_ROOT/scripts/fewstep_bigeval_20261015/slim_predictions.py --src <copied tree of set B> --dst $B/pred/setB
```

`--src` is any tree above `<arm>/step_NNNNNN/<sampler>/predictions/`. The relative layout is
kept, so `costs.jsonl` stays next to each step directory.

The script processes one file at a time and one member at a time per variable. It writes
`<name>.part` and then renames it. It never overwrites a file: rerun with `--skip-existing`
to resume.

**Weather states kept:** `10u 10v 2t 2d msl sp tp cp t_850 z_500`. Add more with `--extra a,b`.

**Kept unchanged:** every dimension, variable, dtype, fill value, compression setting and
attribute. The only change is the length of `weather_state`. Two attributes are added:
`slimmed_from` and `slimmed_weather_states`.

**Dropped by default:** the per-member views `x_<i>`, `y_<i>`, `x_interp_<i>` and
`y_pred_<i>`. They duplicate the 4-d variables and no evaluator here reads them. Pass
`--keep-member-views` to keep them.

At ten states, a file holds about 0.5 GB of uncompressed fields, against about 3.5 GB at 70
states. 20 rows × 61 files come to about 0.6 TB.

## 2. Score

```bash
export PRED_ROOT=$B/pred EVAL_ROOT=$B/eval DT_ROOT=~/dev/dt-bigeval
$DT_ROOT/scripts/fewstep_bigeval_20261015/submit_all.sh            # dry run: lists jobs and rows with no files
$DT_ROOT/scripts/fewstep_bigeval_20261015/submit_all.sh --submit   # 40 jobs, CPU, QOS nf, default account
```

To score one row:

```bash
sbatch --export=ALL,ROW=mf_j1,SET=A,PRED_ROOT=$PRED_ROOT,EVAL_ROOT=$EVAL_ROOT,DT_ROOT=$DT_ROOT $DT_ROOT/scripts/fewstep_bigeval_20261015/score_row.sbatch
```

To see the exact commands without running them:
`python score_row.py --row mf_j1 --set A --pred-root $PRED_ROOT --eval-root $EVAL_ROOT --dry-run`.

**Optional variables:**

- `ONLY` / `SKIP`: comma-separated evaluator lists.
- `ECMWF_SPECTRA=1`: adds spectra_ecmwf_v2. AC only; no tp, no residual (see AUDIT.md).
- `PRED_SET_DIR`: the set sub-directory under `PRED_ROOT`. The default is `set$SET`.

To rerun, resubmit. Finished evaluators are skipped. A failed evaluator does not stop the
others, and the job exits 1.

**Shared inputs.** The reference job builds these once under `$EVAL_ROOT/_cache`:

- `inter_mat/` holds the O96<->O320 linear matrices for texture, built by `eval.shared.inter_mat`
  from the first file.
- `texture/o320_knn32.npz`.
- `shape/` holds the north_atlantic box geometry and mask. The first build reads the O1280
  statics on Atos.

**Evaluators per set:**

- Set A: probabilistic, box_spectra (Europe view, plus the north Atlantic box as view `natl`),
  spectra_coherence, shape, texture, wind_extremes, precip_dist.
- Set B: tc, tc_structure.

## 3. Maps

```bash
PRED_ROOT=$PRED_ROOT MAP_ROOT=$B/maps DT_ROOT=$DT_ROOT $DT_ROOT/scripts/fewstep_bigeval_20261015/maps.sh          # dry run
PRED_ROOT=$PRED_ROOT MAP_ROOT=$B/maps DT_ROOT=$DT_ROOT $DT_ROOT/scripts/fewstep_bigeval_20261015/maps.sh --run    # draw
```

Output goes to `$B/maps/{alps,storm}/calls{1,2,4,59}/`. Each directory holds one PNG per panel
(input, truth, reference and every row of the group), for each variable (`10mwind`, `2t`) and
field (`value`, `-fine`). All panels of a group share one colour scale.

The storm box is 12° wide, centred on the truth's msl minimum (member 1) inside the Idalia box.
It uses the same logic as `eval.cli judge`.

## 4. Aggregate

```bash
python $DT_ROOT/scripts/fewstep_bigeval_20261015/aggregate.py --eval-root $EVAL_ROOT \
    --rows $DT_ROOT/scripts/fewstep_bigeval_20261015/rows.yaml --reference edm_pw30c0 \
    --pred-root A=$PRED_ROOT/setA --pred-root B=$PRED_ROOT/setB --out $B/read
```

Box-spectra bins go down to about 30 km, because the regrid is at 0.125°. O320's spacing is
about 32 km, so bins below about 65 km show the interpolation rather than the model. The
aggregator's `--min-km` cuts them if the read wants that.

## 5. Outputs per row (what the aggregator reads)

`R = $EVAL_ROOT/<set>/<row label>`. Every directory below also contains `.complete`,
`run_meta.json` or `metrics.json`, and plots. `R/effective_config.json` records the options used.

| evaluator | set | file | columns / keys |
|---|---|---|---|
| probabilistic | A | `R/evaluators/probabilistic/scores_by_lead.csv` (per date) | date, step, weather_state, domain, metric, value, n_points, n_members, source_path. `metric` ∈ crps, fcrps, spread, rmse_ens_mean. `domain` ∈ global, n.hem, tropics, s.hem, europe. Wind speed is `10ff`. |
| | | `R/evaluators/probabilistic/summary_by_lead.csv` | step, weather_state, domain, metric, mean, std, stderr, n_dates, n_points_total |
| box_spectra | A | `R/evaluators/box_spectra/bins.csv` (Europe view 33-70 N, 15 W-40 E) and `R/natl/evaluators/box_spectra/bins.csv` (35-55 N, 45-20 W); box in `run_meta.json` | scope (`all` or YYYYMMDD), variable (10u, 10v, ws, 2t, msl, tp), bin_lo_km, bin_hi_km, bin_centre_km, amp_resid_pred, amp_resid_true, ratio_resid, amp_pred_full, amp_truth_full, ratio_full, amp_interp, ratio_interp_to_truth, coherence, aligned, n_draws |
| spectra_coherence | A | `R/evaluators/spectra_coherence/curves.csv` (also `coherence.json`) | scope (`all` or YYYYMMDD), state, ell, wavelength_km, n_samples, amplitude_ratio, coherence, aligned, residual_amplitude_ratio, residual_coherence, interp_amplitude_ratio, interp_coherence, P_true, P_pred, P_interp, P_resid_pred, P_resid_true. nside 128, lmax 256. |
| shape | A | `R/evaluators/shape/rows.csv`, `summary.json`, `run_meta.json` | arm, dirname, date, step, var, member, field (pred/truth/input), band (mid, b12), A_open_ocean, A_all_interior, ell_major_km_median, ell_minor_km_median, ell_ratio_median, ell_angle_to_wind_deg_median, n_windows, elong_frac_gt3, elong_median, major_km_median, minor_km_median, area_km2_median, angle_to_wind_median, n_components. `summary.json` holds pooled means with date-clustered SE. |
| texture | A | `R/evaluators/texture/texture.json` | `aggregate[]`: one entry per (state, stratum), with n_samples and the nearest-neighbour correlation and fine variance of model, truth and input. `samples[]`: one entry per member. |
| wind_extremes | A | `R/evaluators/wind_extremes/wind_extremes.json` | `aggregate[]`: box, n_samples, model, truth, input, model_minus_truth (retention per radius), peak_displacement_km |
| precip_dist | A | `R/evaluators/precip_dist/tail_stats.csv` | scope (`all` or YYYYMMDD), step, series (pred, truth, input), variable (tp), n, wet_frac, mean_mm, p99, p99_9, p99_99, max_mm, neg_share, min_mm, n_members |
| tc | B | `R/evaluators/tc/stats.json` | `events.<idalia\|franklin>.extreme_tail.rows` (msl minimum, 0.1/0.01 percentiles, wind 99.99 percentile and maximum, per source including the OPER analysis), n_dates, steps_hours |
| tc_structure | B | `R/evaluators/tc_structure/cases.csv` (also `summary.json`, `profiles.npz`) | arm, event, date, lead, field, member, storm_ok, found, reason, box_min_hpa, pmin_hpa, centre_lat, centre_lon, fg_lat, fg_lon, fg_pmin_hpa, displacement_km, rmw_km, vmax_tan_ms, vmax_tan_bin_ms, maxwind300_ms, r34_km, r34_censored, r50_km, r50_censored, zeta50_s, zeta100_s, zeta200_s, asym_rmw, ring_points, n_valid_bins, file |
| spectra_ecmwf_v2 (opt-in) | A | `R/evaluators/spectra_ecmwf_v2/` | Amplitude spectra at T319 of y_pred, y and x_interp. The aggregator does not read them. |
| costs | A, B | `<PRED_ROOT>/set<S>/<arm>/step_NNNNNN/costs.jsonl` | spec, calls, seconds. There is no peak-memory field. |

## Tests (CPU)

```bash
python -m pytest scripts/fewstep_bigeval_20261015/tests/test_driver.py   # slim, score_row, rows, submit_all
python -m pytest eval/tests/test_spectra_coherence_residual.py eval/tests/test_texture_o320.py \
    eval/tests/test_inter_mat.py eval/tests/test_wind_extremes_o320.py eval/tests/test_tc_structure_o320.py \
    eval/evaluators/precip_dist eval/evaluators/shape
```

`test_atos_real_tree_has_every_row` runs only where the slimmed tree exists, which means Atos.
Set `BIGEVAL_PRED_ROOT` to point it at another tree. It checks that every row has 36 files in
set A and 25 in set B.
