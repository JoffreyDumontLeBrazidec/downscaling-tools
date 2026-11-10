# Compatibility audit: eval.cli evaluators on the few-step big eval (lane o96_o320)

Branch `claude/project-thread-72mhur`, 15 Oct 2026. Inputs: the harness files of
`predict_members.py` (x on O96, 40,320 points; y, y_pred, x_interp on O320, 421,120 points; ten
members; lon -180..180).

- **Set A** (validation-36): 36 dates in 2022, all at step 60. y is the same for every member.
- **Set B** (Idalia-25): inits 2023-08-26..30, steps 24-120. y is per member, and tp/cp are NaN in y.

Every evaluator runs as
`python -m eval.cli evaluate --lane o96_o320 --predictions-dir D --output-dir <EVAL_ROOT>/<set>/<row>[/<view>] --only <name> --opt ...`.
The `.complete` marker makes reruns skip finished evaluators (`eval/cli/evaluate.py:249-267`).
`--opt` values are read as YAML (`eval/cli/_common.py:196-211`). A second-level block given
with `--opt` replaces the lane's block whole, so `texture.paths` must list every path. The
exact options are in `score_row.py` (`plan()`).

## Verdicts

| evaluator | set | verdict | evidence (file:line) | options used |
|---|---|---|---|---|
| probabilistic | A | **works with opts** | It scores against y of member 0 only (`probabilistic/core/scoring.py:232-233`). That is right on A and wrong on B. `global` is not a default domain (`probabilistic/runner.py:41`, `core/scoring.py:22`). Wind speed is `10ff`. Per-date rows are already in `scores_by_lead.csv`. | weather_states [10u,10v,10ff,2t,msl,tp,t_850,z_500]; domains [global,n.hem,tropics,s.hem,europe]; steps [60] |
| box_spectra | A | **works with opts** | Residual, input/truth, coherence and the aligned part, pooled and per date (`box_spectra/runner.py:1-24, 44-49`). Default steps are [24] (`:262`). The box is a list or a name (`:120-131`). It reads y per member. | Europe view box [33,70,-15,40] into `<row>/`; north Atlantic box `north_atlantic` into `<row>/natl/`; steps [60]; variables [10u,10v,ws,2t,msl,tp]; res_deg 0.125 |
| spectra_coherence | A | **needed fix (done, aa3e8b8)** | The default nside 512 (`spectra_coherence/runner.py:230`) fills only 13 % of the HEALPix map on O320, which biases every amplitude. There was no residual, no aligned part and no per-date output. Now: `nside: auto` gives 128 on O320 (lmax 256, about 156 km), plus residual ratio and residual coherence per degree, `aligned`, `per_date`, and `curves.csv`. | nside auto; steps [60]; weather_states [10u,10v,2t,msl,tp]; per_date true |
| spectra_ecmwf_v2 | A | **works with opts, AC host only; no tp; no residual** | It needs the host prefix `ac` (`evaluators/registry.py:97`, `cli/evaluate.py:101-121`). Dates default to the lane's Idalia dates (`spectra_ecmwf_v2/runner.py:127`), so set A needs `dates`. There is no tp directory (`:30-38`). It reads only y_pred, y and x_interp (`:160, 174, 227`), so there is no residual. | opt-in (`ECMWF_SPECTRA=1`): steps [60]; dates = the 36 file dates; members 1-10; weather_states [10u,10v,2t,msl,sp,t_850,z_500]. Proposed patch below. |
| shape | A | **needed fix (done, ca4d31d)** | The kernels are O1280 sizes (`shape/instrument.py:72` SIGMAS, `:88` GRAD_H_KM, `:83-84` regrid and window). The north_atlantic box was refused on O320: its first row is at 35.27 N, more than the 0.2 deg tolerance (`shape/boxes.py:109-121`). Added `kernel_scale`, `window_deg` and a tolerance of one grid row. Masks are built from O1280 statics on Atos (`instrument.py:188-233`). | box north_atlantic; steps [60]; kernel_scale 4 (b12 is about 84-160 km, mid about 160-436 km); regrid_deg 0.36; window_deg 8; cache_dir `$EVAL_ROOT/_cache/shape`; stats_file o320_dict_0_72.npy |
| texture | A | **needed fix (done, c51481e)** | The default paths are the O320 to O1280 matrices and the O1280 statics (`texture/runner.py:156-170`). Added `paths.forcings_npz` (`:339, 384`), `input_source: x_interp` (`:809`), and an O96<->O320 linear matrix pair built from a file (`eval/shared/inter_mat.py`). | steps [60]; weather_states [10u,10v,2t,msl,t_850]; input_source x_interp; paths {up/down: `$EVAL_ROOT/_cache/inter_mat/*`, residual_stats o320_dict_0_72.npy, forcings_npz o320-forcings.npz, knn_cache `$EVAL_ROOT/_cache/texture/o320_knn32.npz`} |
| wind_extremes | A | **needed fix (done, 9922866)** | The default up matrix is O320 to O1280 (`wind_extremes/runner.py:105`), which has the wrong shape for O96 x. Added `input_source: x_interp`, and the grid check now runs before the box is built. The default radii and adjacency are O1280 sizes. | steps [60]; input_source x_interp; radii_km [50,75,100,150,200]; adjacency_km 40 |
| precip_dist | A | **needed fix (done, 4e0de85)** | Its PDF uses one member and binned values (`precip_dist/core/tp_histogram_comparison.py:145,167`). It wrote no numbers for the tail. `tail_stats: true` now writes exact wet fraction, p99/p99.9/p99.99, max and the negative share, over all members, pooled and per date (`precip_dist/core/tail_stats.py`). Series that are all NaN are skipped (set B truth). | steps [60]; tail_stats true; wet_threshold_mm 0.1 |
| tc | B | **works (Atos only)** | It reads msl, 10u and 10v (`tc/core/loading_predictions.py:133-136`). The lane block has events idalia and franklin, grib_dir and OPER_O320_0001 (`config/lanes/o96_o320.yaml:30-33`). The regridded mode needs metview (`module load ecmwf-toolbox`). | the lane's block |
| tc_structure | B | **works with opts** | The default cell area is O1280's (`tc_structure/core.py:46,62`) and dr is 10 km (`:59`). On O320 every radial bin is then empty and RMW is NaN (`eval/tests/test_tc_structure_o320.py`, 188cdbc). | events [idalia,franklin]; params {cell_area_km2: 1211.2, dr_km: 20} |
| zoom_maps | A, B | **works with opts** (maps.sh) | Hres panels are triangulated from any grid. The default titles say O1280 (`zoom_maps/core/plot_member_wind_maps.py:78-88`). The default fine cut of 0.6 deg is set for an O320 input. | `--region alps` or `--extent` (storm); `--fine-cut-deg 2.0`; `--title` per panel |
| eval.cli judge | – | **not used: pairwise only** | It takes one `--step` for evaluators and maps (`cli/judge.py:58`), and one run against one reference. For 20 rows the read is evaluate + `aggregate.py`. maps.sh reuses `judge.storm_centre` (`:143`) and STORM_HALF_DEG (`:46`). | – |

## What cannot run, and why

- **probabilistic on set B.** It scores against member 0's truth. Set B truth differs per
  member, so the scores would be wrong. It is not run on B.
- **spectra_ecmwf_v2 off AC, and for tp or the residual.** It needs gptosp, which exists on AC
  only. It has no tp mapping and no residual variable. It is opt-in. The global per-degree read
  comes from spectra_coherence, which stops at about 156 km on O320 (nside 128, lmax 256). To
  reach finer degrees, apply the patch below and run spectra_ecmwf_v2 at T319.
- **Peak memory per row.** `costs.jsonl` has no peak-memory field.
- **shape mask build off Atos.** It needs the O1280 kNN file
  (`/home/ecm5702/scratch/agent-work/20260901-sigma-scale-map/o1280_knn128.npz`, in scratch and
  liable to purge), the forcings zarr and the IM matrices. After the first build, the mask cache
  under `$EVAL_ROOT/_cache/shape` is enough.

## Proposed patch (not applied): spectra_ecmwf_v2 residual and tp

1. In `spectra_ecmwf_v2/_grib_stager.py`, let `prediction_var` take a difference `"A-B"`, for
   example `y_pred-x_interp`. The stager reads both variables for the member and state, and
   writes `A - B` into the GRIB template in place of the single field.
2. In `runner.py:160-227`, after the y_pred, y and x_interp passes, add two passes labelled
   `resid_pred` (`y_pred-x_interp`) and `resid_truth` (`y-x_interp`). They go through the same
   amplitude computer. In `scorer.py`, write `ratio_resid = amp(resid_pred) / amp(resid_truth)`
   per wavenumber, next to the full-field ratio.
3. Add `"tp": "tp_sfc"` and `"cp": "cp_sfc"` to `_WEATHER_STATE_TO_DIR`. tp is accumulated, so
   the template seed must have `stepType=accum` (`_grib_stager.resolve_or_synthesize_template`).
4. Keep the reference cache keyed by window as it is. The residual passes need no reference.

Tests: a two-file synthetic set where y_pred = x_interp + 2 (y - x_interp) must give
ratio_resid = 2 at every wavenumber. Use a mock amplitude computer (the existing tests mock
gptosp).

## Fixes on this branch (each with CPU tests on the real O96/O320 point sets)

| commit | what |
|---|---|
| aa3e8b8 | spectra_coherence: `nside: auto`, residual ratio and coherence per degree, aligned part, `per_date`, `curves.csv` |
| 4e0de85 | precip_dist: `tail_stats.csv` (exact tail over every member, pooled and per date) |
| 9922866 | wind_extremes: `input_source: x_interp` |
| c51481e | texture: `paths.forcings_npz`, `input_source: x_interp`; `eval.shared.inter_mat` (O96<->O320 linear pair) |
| ca4d31d | shape: one-row box tolerance, `kernel_scale`, `window_deg` |
| 188cdbc | tc_structure: O320 test; parameters for O320 (`cell_area_km2` 1211.2, `dr_km` 20) |

Other lanes are unchanged. Every new key is off by default, and the defaults keep their old values.
