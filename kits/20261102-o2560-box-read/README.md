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
- `kit/smoke.sbatch` + `kit/smoke_compare.py`: the GPU smoke (one date, +24 h, 2 members, rows B24, A100, B50, B24X; B24X
  runs B24's weights on A's code to show the crop commits leave full-box inference unchanged).
- `kit/predict.sbatch`: one row at one lead. `kit/score.sbatch`: one lead (box spectra per row, spectra table, paired
  scores). `kit/final.sbatch`: zoom maps, summary tables, `eval.cli judge` (rule 32 v2).
- `scripts/paired_scores_o2560.py`, `scripts/spectra_table.py`, `scripts/zoom_maps_o2560.py`, `scripts/summarize.py`,
  `scripts/compare_configs.py`, `scripts/dry_all.sh`, `scripts/submit_all.sh`.
- `tests/`: pytest unit tests (run on the Jupiter login node; output in `checks/pytest_login.txt`).
- `checks/`: dry-run logs, pytest output, config diff, MANIFEST.json.

## Scoring (rule order, wind speed and tp first)

1. Per-bin amplitude spectra, full field and residual against the o2560 truth, and the input's ratio: `box_spectra` of
   downscaling-tools 4e659c3 at 0.035 degrees (the o2560 spacing), whole box without a 1-degree margin, ws tp 10u 10v 2t
   msl, pooled over 12 dates x 10 members. Bins below 15.6 km are marked model against model.
2. tp tail: p99, p99.9, p99.99 and the per-date maximum over the box without the edge strip, members pooled, for each row,
   IN and the truth; counts of points at or above 10, 25, 50, 100 mm per 6 h.
3. Zoom maps at +24 and +72 h: the Alps box 43-49 N 4-16 E on the wettest Alps date, and the date with the deepest truth
   low north of 35 N (called a storm only if below 990 hPa); truth, input, members 1-3 of each row; tp and wind speed.
4. Fair CRPS, spread, ensemble-mean RMSE and the corrected spread/RMSE ratio sqrt((M+1)/M) x spread / RMSE, per date, for
   all 11 outputs plus 10 m wind speed, over the box, Europe and the Alps, with paired SE over the 12 dates.
5. `eval.cli judge --lane o1280_o2560` (rule 32 v2) A100 vs B24 and A100 vs B50 at every lead. It has never run on o2560
   rows; a failure is recorded, not patched.

Not in this kit: the global fccc23df box-cut reference (it needs 4-GPU sharded o2560 inference that has never run on this
box or these dates; not cheap), the shape (elongation and rotation) evaluator, and the TC storm evaluator (no tropical
cyclone in the box in these dates).

## Jobs (to submit after the review and the go)

1. Smoke: `ssh ac6-100 'ssh jupiter sbatch /e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/kit/smoke.sbatch'`
   (1 GH200, at most 2 h). Read: O2560_SMOKE_DONE, finite draws, RANGE and SANITY lines, seconds and peak GiB per draw,
   SMOKE_COMPARE_PASS, then `sacct -j <id> --format=JobID,MaxRSS,Elapsed`.
2. Read: `ssh ac6-100 'ssh jupiter bash /e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/scripts/submit_all.sh'`:
   18 predictions (3 rows x 6 leads), 6 scorings (afterany on their lead's 3 predictions), 1 final (afterany on the 6
   scorings). Ids in `state/jobs.txt`; a second run refuses (`state/SUBMITTED`).

Progress check: `ssh ac6-100 'ssh jupiter "squeue -u $USER -o \"%.10i %.24j %.8T %.10M %R\" | grep o25; ls /e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102/pred/*/ | grep -c nc; grep -h DONE /e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read/logs/o25*.out"'`

Results: `outputs/o2560_box_read_20261102/pairs/L<lead>/{paired.md,tail.md,spectra.md,per_date.csv,spectra.csv}`,
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

`python -m pytest tests` (output `checks/pytest_login.txt`): 30 passed. The tests cover the seed check values, one seed
per draw with no reuse, identical seeds across rows (also read back from the dry-run records), the 11 outputs and the
EMA-10000 half-life in each checkpoint's metadata, tp clipping and de-accumulation on synthetic arrays, the point counts
of the stores, the graph and the matrix (3,248,042 o2560 points, 434,914 hidden nodes, 876,159 o1280 points), validation
dates outside training (both configs and each checkpoint's metadata), no FDB call, no scancel or deletion, no output under
home, the A-against-B config comparison, the fair CRPS against a brute-force formula, and an end-to-end run of the scoring
scripts and of the `box_spectra` runner on synthetic files written by the predictor's own writer.

Not tested before a GPU job: memory and time per draw on one GH200 with chunking 16, and whether heads sharding with a
one-rank group runs on GPU (the smoke tests both).

Note for the reviewer: B's inference files (2.46 GB) hold 48 more pickled tensors than A's (2.13 GB; 691 against 643
zip entries). They are not in the state dict (both load strictly into the same model), so they are attributes of the
pickled model object, most likely cached graph or sharding tensors of B's two-GPU layout. The predictor loads only
`state_dict()` into a model built from the shared r9 graph files, so they do not enter the read.

## Cost (estimate, the smoke measures it)

Forward-only inference of this model on the full box is unmeasured. From training (1.99 s per fwd+bwd step on 2 GH200),
a 59-call draw is guessed at 60-120 s on one GH200: 3 rows x 6 leads x 12 dates x 10 members = 2,160 draws = about
36-72 GH200-h for the predictions (each job 2-4 h for its 120 draws, wall 6 h), plus CPU scoring. Disk: about 1.77 GB
per file, 216 files = about 382 GB under `/e/data1`. All twelve leads would double both.

## Caveats a reader needs

- A's EMA half-lives are in steps (2,000 and 10,000) on 3.9x cheaper steps, so A's EMA-10000 averages over about a quarter
  of the compute that B's does. Both rows use EMA-10000 as briefed.
- 12 dates of one autumn fortnight: no summer convection; only large differences settle (the E2 wide read needed 30 dates
  for 2-5 % score gaps). Every number carries its n.
- Training forecasts from late October reach into the first validation days by valid time (31 Oct +72 h is valid 3 Nov);
  they are different forecasts of overlapping weather.
- The inference memory and time per draw on one GH200 are not measured until the smoke runs; chunking 16 is the guess
  (training needed 2 GPUs per model for the full box, with gradients).
