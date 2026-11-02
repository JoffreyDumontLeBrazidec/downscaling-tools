# o2560 Europe-box read kit (built 2 Nov 2026, NOT submitted)

## What this is

This kit is the first read instrument for the o1280 (9 km) to o2560 (4.4 km) Europe-box fine-tunes of DB300. Two training
chains finished on 1 Nov 2026 with no way to evaluate them: the AG crop chain `db_o2560_crop_100k` (100,000 steps, each
step on one of nine overlapping windows of the box) and the Jupiter full-box chain `db_o2560_e2_50k_v2` (50,000 steps on
the whole box). The read compares them at equal GPU-hours (A100 against B24) and at their last steps (A100 against B50),
with the o1280 input interpolated to o2560 as the reference (the objective is to beat it), on the only dates outside
training: 1-12 November 2024. The kit predicts on Jupiter, one GH200 per job, and scores on CPU. The record of the lane
lives in the docs repo note `epics/o1280-o2560-lane-defects/in-progress/20261027_ca300_finetune_o1280_o2560_smoke.md`.

The live copy is on Jupiter at `/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/`; this
folder is the reviewed copy (same files, plus the dry-run and test outputs under `checks/`). Outputs go to
`/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102/` (never home).

**Status: built and dry-run on the login node; nothing submitted. The GPU smoke and the read wait for the review and
Joffrey's go.**

## Rows

| Row | Run | Step | GH200-h (sacct) | EMA-10000 file (sha256) | Code pin |
|---|---|---|---|---|---|
| A100 | `db_o2560_crop_100k` (AG, 8 GH200) | 100,000 | 234.1 | Jupiter copy `ckpt/A100k/inference-ema10000-anemoi-by_step-epoch_198-step_100000.ckpt` (53ef84ac…) | 4ad32838c (its training code; crop off at inference) |
| B24 | `db_o2560_e2_50k_v2` (Jupiter, 16 GH200) | 24,000 | 229.9 | `ckpt_root/168c4355…/inference-ema10000-anemoi-by_step-epoch_036-step_024000.ckpt` | 20ba15241 |
| B50 | same chain | 50,000 | 477.0 | `…/inference-ema10000-anemoi-by_step-epoch_076-step_050000.ckpt` | 20ba15241 |
| IN | the o1280 operational IFS input, interpolated to o2560 by the config's own matrix | - | - | x_interp in every prediction file | - |

GPU-hours: A = ElapsedRaw of 50072780, 50072814, 50072858 (43,212 + 43,224 + 18,929 s) x 8 GH200. B = segment 1
2593073 (43,275 s, ends at 20k) x 16, plus the time from the start of segment 2 (2604539, 19:08:20 UTC 31 Oct) to the
checkpoint's write time x 16: 24k at 21:29:16 = 229.9, 25k = 239.3, 26k = 248.5; 50k = 477.0. B24 is the nearest saved
checkpoint to A100's 234.1.

The A checkpoint files (raw, inference, EMA-2000, EMA-10000 at 100k) were also copied off the purged Atos scratch to
`/lus/h2resw01/hpcperm/ecm5702/sandbox/20261031-db-o2560-crop-100k/ckpt_100k_safe/` (sha256 equal on both ends).

## Settings (identical across rows)

- Dates: 20241101 to 20241112 (12 base dates, 00 UTC). Both chains train on `fake_forecasts end: 2024-10` and validate on
  2024-11 (`tests/test_dates.py` reads it from both configs and from each checkpoint's metadata).
- Leads: +12, +24, +36, +48, +60, +72 h (`LEADS` in `scripts/submit_all.sh`; all twelve 6-72 h leads are possible at
  twice the cost and disk, decided after the smoke measures the time per draw).
- 10 members per (date, lead); seed of member m = `derive_draw_seed(1234, yyyymmdd, lead_h, m)` (numpy SeedSequence,
  rule 31), identical in every row and written into each file's `seeds` attribute.
- Sampler: 30-step piecewise (10 exponential + 20 Karras), sigma_max 1e5, Heun, churn off (59 model calls per draw).
- fp32 with TF32 matmul, `ANEMOI_INFERENCE_NUM_CHUNKS=16`, one GH200, fixed Triton kernel (env.sh checks the two `gt.py`
  lines of the training job's exit-47 guard).
- Config: `configs/db_o2560_read.yaml` = the Jupiter chain's config a6eb3db6 for all rows. `scripts/compare_configs.py`
  checks that A's config (AG 4b87aff5) differs only in paths, crop, compile, hardware, run length and naming
  (`checks/config_diff_A_vs_B.txt`: 27 keys, CONFIG_COMPARE_PASS).
- Outputs: all 11 (10u 10v 2d 2t msl skt tcw cp tp z_500 t_850; no sp), checked against each checkpoint's metadata.
- Precipitation: tp and cp are 6 h windows ending at the lead in both stores (anemoi `accumulations`, period [0, 6]);
  nothing is de-accumulated (the dry run prints the box mean per lead as evidence). Negative tp and cp in the prediction
  and the input are clipped to 0 before writing; the count and minimum clipped are file attributes.
- Input tp and cp: the box store has none, so the predictor reads them from the global od o1280 source store
  (`/e/data1/jureap-data/ecmwf/ai-ml/datasets/downscaling-od-fc-oper-0001-mars-o1280-2023-2024-24h-v1.zarr`) at the same
  (date, step), cut to the same 876,159 points (coordinates equal to 0) and interpolated with the same matrix.

## Files

- `kit/o2560_box_predict.py`: the predictor (MODE dry, smoke or predict). `kit/o2560_common.py`: seeds, dates, outputs,
  clipping, fair CRPS. `kit/rows.sh`, `kit/env.sh`, `kit/eval_env.sh`: rows and environments.
- `kit/smoke.sbatch` + `kit/seed_check.py` + `kit/smoke_compare.py`: the GPU smoke (one date, +24 h, 2 members, rows B24, A100, B50, B24X; B24X
  runs B24's weights on A's code to show the crop commits leave full-box inference unchanged).
- `kit/predict.sbatch`: one row at one lead. `kit/score.sbatch`: one lead (box spectra per row, spectra table, paired
  scores). `kit/final.sbatch`: zoom maps, summary tables, `eval.cli judge` (rule 32 v2).
- `scripts/paired_scores_o2560.py`, `scripts/spectra_table.py`, `scripts/zoom_maps_o2560.py`, `scripts/summarize.py`,
  `scripts/compare_configs.py`, `scripts/dry_all.sh`, `scripts/submit_all.sh`.
- `tests/`: pytest unit tests (run on the Jupiter login node; output in `checks/pytest_login.txt`).
- `checks/`: dry-run logs, pytest output, config diff, MANIFEST.json, provenance of each chain's training inputs
  (`provenance_atos_A.json`, `provenance_jupiter_B.json`: residual statistics 10e6c0a4 and DB300 warm start b7e854f7 on
  both, and identical mean/stdev/minimum/maximum of both box stores).

## Scoring (rule order, wind speed and tp first)

The 1-degree edge strip of the box IS excluded from every score below (points within 1 degree of 10 N, 70 N, 80 W or
40 E). Area weights are cos(lat) divided by the number of points on the full octahedral ring of the point
(4 i + 16 on ring i), which is the point area of the o2560 reduced Gaussian grid; `o2560_common.area_weights` checks the
longitude spacing of every ring, so a wrong ring index fails instead of weighting silently. Note for Joffrey: the
repository's `eval/evaluators/surface/core/compute.py:_area_weights` uses cos(lat) per point, without the division by
the ring count; that is not changed here.

1. Per-bin amplitude spectra, full field and residual against the o2560 truth, and the input's ratio: `box_spectra` of
   downscaling-tools 4e659c3 at 0.035 degrees (the o2560 spacing), ws tp 10u 10v 2t msl, pooled over 12 dates x 10
   members, on two regions: the whole box less a 1-degree margin, and Europe 35-70 N 25 W-40 E. Bins below 15.6 km are
   marked model against model. box_spectra takes one dx, at the region's middle latitude (40 N for the box, 52.5 N for
   Europe), so the wavelength labels are approximate across latitudes; the ratios between rows, truth and input are valid.
2. tp tail per date over the box: p99, p99.9 and p99.99 with all points and members pooled (labelled `_pooled`), and the
   mean over members of each member's maximum (`max_member_mean`), for each row, IN and the truth; counts of points at or
   above 10, 25, 50, 100 mm per 6 h.
3. Zoom maps at +24 and +72 h (those of the two among the read's leads): the Alps box 43-49 N 4-16 E on the wettest Alps
   date, and the date with the deepest truth low north of 35 N (called a storm only if below 990 hPa); cases chosen among
   the dates every row holds; truth, input, members 1-3 of each row; tp and wind speed.
4. Per date: fair CRPS, spread = sqrt(area-mean ensemble variance, ddof 1) and RMSE = sqrt(area-mean squared error of the
   ensemble mean), for all 11 outputs plus 10 m wind speed (per member from 10u and 10v), over the box, Europe and the
   Alps, with paired SE over the 12 dates. The corrected spread/RMSE ratio sqrt((M+1)/M) x sqrt(mean over dates of the
   variance) / sqrt(mean over dates of the squared error), pooled over dates as in
   `eval/evaluators/probabilistic/core/calibration.py`, with a 95 % date-bootstrap interval (seed 32, 2000 resamples)
   and a paired bootstrap of the ratio difference between rows. All rows must hold the same dates, member count, seeds,
   truth, input and grid; the scoring stops otherwise.
5. `eval.cli judge --lane o1280_o2560` (rule 32 v2) A100 vs B24 and A100 vs B50 at every lead. It has never run on o2560
   rows; a failure is recorded, not patched.

Not in this kit: the global fccc23df box-cut reference (it needs 4-GPU sharded o2560 inference that has never run on this
box or these dates; not cheap), the shape (elongation and rotation) evaluator, and the TC storm evaluator (no tropical
cyclone in the box in these dates).

## Jobs (to submit after the review and the go)

1. Smoke: `ssh ac6-100 'ssh jupiter sbatch /e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/kit/smoke.sbatch'`
   (1 GH200, at most 2.5 h; outputs `outputs/o2560_box_read_20261102/smoke_<jobid>/`). Rows on 20241101: B24 at +24 and
   +48 h (2 members), B24 at +24 h with 1 member in a separate process, B24 at +24 h with chunks 8, A100, B50 and B24X
   (B24 on A's code) at +24 h (2 members). Read: O2560_SMOKE_DONE; SEED_CHECK_PASS (member 1 reproduced by the 1-member
   run, members 1 and 2 differ, +24 and +48 differ, seeds as derive_draw_seed; chunks 8 against 16 reported);
   SMOKE_COMPARE_PASS or WARN (code pins, information only, 1e-2 of the truth's std); finite draws, RANGE and SANITY
   lines, seconds and peak GiB per draw; then `sacct -j <id> --format=JobID,MaxRSS,Elapsed`.
2. Read: `ssh ac6-100 'ssh jupiter bash /e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/scripts/submit_all.sh'`:
   18 predictions (3 rows x 6 leads), 6 scorings (afterany on their lead's 3 predictions), 1 final (afterany on the 6
   scorings). Ids in `state/jobs.txt`; a second run refuses (`state/SUBMITTED`). If an sbatch fails part way, the script
   prints the jobs already queued and the MISSING ones, to submit by hand.

Progress check: `ssh ac6-100 'ssh jupiter "squeue -u $USER -o \"%.10i %.24j %.8T %.10M %R\" | grep o25; ls /e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102/pred/*/ | grep -c nc; grep -h DONE /e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs/o25*.out"'`

Results: `outputs/o2560_box_read_20261102/pairs/L<lead>/{paired.md,tail.md,ratio.csv,per_date.csv,spectra_box.md,spectra_europe.md,spectra_*.csv}`,
`summary/summary_{box,europe,alps}.md`, `summary/summary_scores.csv`, `maps/L024/*.png`, `judge/`.

## Dry run and tests (2 Nov 2026, Jupiter login node, CPU)

`scripts/dry_all.sh` (logs in `checks/dry_logs/`) printed DRY_ALL_OK at 07:20:45 UTC: A100 over all 144 (date, lead) rows of
the split (12 dates x leads 6-72 h), B24 and B50 on one case each. What it showed:
- each EMA-10000 file loads into the model with 0 missing and 0 unexpected keys, on its own code pin;
- each EMA file differs from the raw weights of the same step (processor weights, median of the per-tensor max
  difference 0.022 for A100, 0.019 for B24, 0.026 for B50), so no file holds raw weights by mistake;
- the od o1280 tp/cp points equal the in_lres box points (876,159 points, coordinate difference 0);
- box-mean tp over the 12 dates per lead stays between 0.64 and 0.72 mm in both truth and input from +6 h to +72 h, so
  both stores hold 6 h windows (a total since the start would grow about twelvefold); truth and input agree within 0.02 mm;
- the model's outputs are the 11 expected in the expected order; all 144 rows resolve; truth and input are finite.

`python -m pytest tests` (output `checks/pytest_login.txt`): 39 passed (after the 2 Nov review fixes). The tests cover the
seed check values, one seed per draw with no reuse, identical seeds across rows (also read back from the dry-run records),
the 11 outputs and the EMA-10000 half-life in each checkpoint's metadata, tp clipping and de-accumulation on synthetic
arrays, the point counts of the stores, the graph and the matrix (3,248,042 o2560 points, 434,914 hidden nodes, 876,159
o1280 points), validation dates outside training (both configs and each checkpoint's metadata), the octahedral ring
weights (a full synthetic grid, a wrong ring index failing, and the longitude spacing of every ring of the real o2560
box store), equal training inputs of A and B (residual statistics, DB300 donor, store statistics), no FDB call, no
scancel or deletion, no output under home, the A-against-B config comparison, the fair CRPS against a brute-force formula,
and the scoring scripts end to end on synthetic files written by the predictor's own writer: a calibrated synthetic
ensemble reads a corrected spread/RMSE within 0.02 of 1 for every variable (under- and over-dispersive rows below 0.9 and
above 1.2), tp percentiles, member maxima and the 10 m wind speed scores match values computed by hand, rows with
different dates are refused, the box_spectra runner reads the layout, and the spectra table puts every value in its column.

Not tested before a GPU job: memory and time per draw with the model sharded over 4 (and 2) GH200, and the RANKCHECK
agreement across ranks (the second smoke tests them).

Note for the reviewer: B's inference files (2.46 GB) hold 48 more pickled tensors than A's (2.13 GB; 691 against 643
zip entries). They are not in the state dict (both load strictly into the same model), so they are attributes of the
pickled model object, most likely cached graph or sharding tensors of B's two-GPU layout. The predictor loads only
`state_dict()` into a model built from the shared r9 graph files, so they do not enter the read.

## One GH200 is not enough: the model is sharded over 4 (smoke 2645083, 2 Nov 2026)

The first smoke (Jupiter job 2645083, 08:30-08:36 UTC, FAILED 1:0, MaxRSS 828 MB on the batch step) ran out of GPU memory
on the first draw of every row: "Tried to allocate 24.78 GiB ... this process has 89.33 GiB memory in use", in the node
MLP of a mapper block (`block.py:1007`, 3,248,042 o2560 points x 2,048 hidden units x 4 bytes). The inference chunking
does not cover that MLP. No draw, seed check or comparison ran. Training needed 2 GH200 per model for the full box too.

The predictor therefore runs the model sharded over the ranks of one node, the route of the GL512 kit
(`agent-work/20261030-gl512-100k-wide/kit/gl512_predict.py`, the same DB model class, read successfully on 4 GH200 with the
global o1280 grid of 6.6 million points): every rank reads the sample, the model_comm_group spans the job's ranks, and
`GlobalNoise` draws the full box-wide noise field with the case seed and keeps each rank's slice, so the noise of a draw
does not depend on the number of ranks. A RANKCHECK line per draw checks that the noise and the gathered output are
identical on every rank. Rank 0 computes the input reference and writes the file. Prediction jobs take one node of
4 GH200. The second smoke also runs B24 on 2 ranks: if 2 GH200 suffice and cost fewer GPU-hours per draw, the read can use
2; the seed check requires the 2-rank and 4-rank draws to agree to float level.

## Cost (estimate, the smoke measures it)

Forward-only inference of this model on the full box is unmeasured. From training (1.99 s per fwd+bwd step on 2 GH200),
a 59-call draw on 4 GH200 is guessed at 30-60 s: 3 rows x 6 leads x 12 dates x 10 members = 2,160 draws = about
72-144 GH200-h for the predictions (each job 1-2 h of wall for its 120 draws, wall limit 6 h), plus CPU scoring.
Disk: about 1.77 GB per file, 216 files = about 382 GB under `/e/data1`. All twelve leads would double both.

## Caveats a reader needs

- A's EMA half-lives are in steps (2,000 and 10,000) on 3.9x cheaper steps, so A's EMA-10000 averages over about a quarter
  of the compute that B's does. Both rows use EMA-10000 as briefed.
- 12 dates of one autumn fortnight: no summer convection; only large differences settle (the E2 wide read needed 30 dates
  for 2-5 % score gaps). Every number carries its n.
- Training forecasts from late October reach into the first validation days by valid time (31 Oct +72 h is valid 3 Nov);
  they are different forecasts of overlapping weather.
- One GH200 per model ran out of memory (smoke 2645083); the read shards the model over 4 GH200 (see above).
