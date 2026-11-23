# Extremes and cyclone evaluation set XSET-1 (36 to 9 km lane)

42 start dates of 2024-2025 (00 UTC), frozen 16 Nov 2026: 22 cyclone dates (65 storms, 425 IBTrACS cases) and 20
heavy-rain dates chosen on station observations. Leads +12 to +72 h every 12 h, 10 members, one draw per input member,
rule-31 seeds. Reference rows: INPUT (36 km ensemble, native grid), ENS9 (9 km ensemble, the target), CF9 (9 km control).
Design, reading rules, detectable effects and results: docs
`epics/fine-scale-o320-o1280/in-progress/20261116_extremes_cyclone_eval_set.md`.

## One command per model row

On `ac6-100` (Atos login node; it reaches Jupiter by ssh), from the kit checkout
`K=/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set/downscaling-tools/kits/20261116-extremes-set`:

    bash $K/xset_row.sh <ROW> <MODEL> <STEP> [EMA_TAG] [--wait] [--table "ROW ROW"] [--smoke-only]

    bash $K/xset_row.sh GL512_450k   GL512 450000 --table "GL512_500k GL512_450k" --wait
    bash $K/xset_row.sh FS242_freeze FS242 <step of the frozen export> --wait

- `ROW`: the name of the line in the table; one row = one export, defined once (files and sha256 of the weights are
  recorded in `XOUT/rowdefs/<ROW>.env` on Jupiter; the same name with another export is refused).
- `MODEL`: a file `jupiter/models/<MODEL>.env` = what is common to all exports of a run (run directory, config, code tree,
  pins). `GL512.env` exists. A new model: copy `jupiter/models/TEMPLATE.env`, fill, commit, push, pull on both clusters.
- `STEP`, `EMA_TAG` (default `ema2000`): the export. Files outside the run directory: `EMA_FILE=... RAW_FILE=...` in front.
- The command does the next missing step and can be run again at any time (exit 0 = table written, 3 = a step is
  running): define, smoke of a new model, six Jupiter jobs (252 files, about 100 GH200-h), copy to Atos, score job
  (stations, cyclones against IBTrACS, IMERG, tail check; about 1 h), table with detectable effects in
  `A/table_<ROW>/xset_table.md`. With `--wait` it polls every 20 minutes to the end.
- It never submits while jobs of the row are queued, only asks for the dates whose files are missing, refuses a kit
  checkout with local or unpushed changes, and stops after 3 submissions of one row (`XSET_FORCE=1` to go on).

## What a model must provide (checked in every job)

- the 70 input variables of the global `in_lres` store, in its order; the O1280 grid of the ENS9 row files;
- the outputs `10u 10v 2t msl tp` BY NAME. A model with more outputs (12 with `z_500` and `t_850`) is read unchanged; the
  extra outputs are not written. Its noise tensor has another shape, so its draws are date-paired but not noise-paired
  with a 10-output row, even under the same seeds;
- a test-split case that proves the input paths (`ANCHOR`, default `20230101:24`);
- for a run trained on crops: the hydra overrides that build the global model (`EXTRA_OVERRIDES` in the model file).

A new model is smoked before its first row (`--smoke-only`, one case, 2 members, about 0.7 GH200-h): trainer build,
strict weights load, anchor, outputs by name, and fatal bounds on the difference to the 9 km control (catches a wrong
normalisation, unit or variable order). Do it days before the row is due, on any existing export.

A sampler other than the 30-step standard: `XSET_SCHED_JSON` and `XSET_SAMPLER_JSON` in front of the command when the
row is defined (recorded in the row definition and in every file). Not yet run with a few-step (MeanFlow) model.

## Pieces

| step | where | script |
|---|---|---|
| selection of the dates (done, frozen) | Mac / Atos | `ibtracs_extract.py`, `select_tc_dates.py`, `scan_station_tp.py`, `select_rain_dates.py`, `lists/` |
| retrievals, input stores, ENS9 row files (done for the 42 dates) | Atos | `xset_chain_submit.sh` |
| row definition, state, submission | Jupiter | `jupiter/xset_define_row.sh`, `xset_row_state.sh`, `xset_submit_row.sh` |
| draws | Jupiter | `jupiter/xset_predict.sbatch`, `xset_smoke.sbatch`, `xset_run.sh`, `xset_runs.sh`, `xset_env.sh`, `xset_predict_global.py` |
| copy, scores, table | Atos | `xset_copy_from_jupiter.sh`, `xset_score.sbatch`, `xset_table.sh` |

The first rows `GL512_500k`, `GL512_450k`, `GL512_237k` keep their fixed lines in `jupiter/xset_runs.sh` (drawn before
row definitions existed); `xset_row.sh` handles them like any other row.

## Cyclone members per lead (17 Nov 2026)

`xset_score_events.py` also writes `tc_members.csv`: one line per (case, row, member), pressure, wind, centre, radius of
maximum wind. Member k of INPUT is the input of member k of a model row and has the number of member k of ENS9.

- Rows scored before 17 Nov (ENS9, GL512_500k): `sbatch --job-name=xset_tcmem_ENS9 K/xset_tc_members.sbatch ENS9 refs`,
  `sbatch --job-name=xset_tcmem_GL512_500k K/xset_tc_members.sbatch GL512_500k`. Output `W/scores_members/<ROW>/`; the log
  must say `TC_SAME` (the case scores equal the ones of `W/scores/<ROW>`). Rows scored later have the file in `W/scores/<ROW>/`.
- Read: `python3 K/xset_tc_lead_read.py --ens W/scores_members/ENS9 --row GL512_500k=W/scores_members/GL512_500k
  --row GL512_450k=W/scores/GL512_450k --out W/tc_lead_read/tc_lead_read.md` (pandas; seconds). Tables per lead: means,
  what each row adds to its input member, the same by class of input wind, pairing of members between rows, structure.

## SCREEN-TC: the standard screen (21 Nov 2026)

Adopted by the owner 21 Nov 2026 15:13 UTC; calibration in docs `epics/misc/20261121_information_gain_audit.md` s8.
A screen row = the 22 cyclone dates (tag TC or both) x members 1-5 x 6 leads = 132 files, about 26 GH200-h (3 Jupiter
jobs of 8, 8, 6 dates). It FLAGS whether an arm is worth a full row; it never gives a verdict and never enters a table,
scoreboard or paper (rules 46, 48).

    bash $K/xset_row.sh <ROW> <MODEL> <STEP> --screen [--kind sampler] [--table "GL512_450k OTHER_FULL_ROW"]

- The row name gets the suffix `_SCRTC` (added if missing); its definition carries `XSET_SCREEN=TC`, so the draw
  (MEMBERS=5), the state (`need=132`) and every output folder say it is a screen. A full row cannot take that suffix.
- Members 1-5 of a screen row use the same rule-31 seeds as members 1-5 of a full row of the same export (proof below).
- Scoring: `xset_score.sbatch <ROW> screen` = cyclone cases only (no stations: 5-member station scores gave wrong-sign
  calls on the 2023 set; no IMERG; no tail check).
- Read: `xset_screen_read.py` against the `--table` rows (full rows, cut to members 1-5; default GL512_450k), ENS9 and
  INPUT levels for context: pmin and wind fair CRPS against IBTrACS, paired case mean, delete-one-storm jackknife SE,
  MDE, call (|d| > 2.8 SE and > line; line 0.4 hPa / 0.3 m/s training, 0 sampler). Output `A/screen_<ROW>/screen_read.md`
  with a SCREEN FLAG line: WORTH A FULL ROW / NOT WORTH A FULL ROW / MIXED / NO CALL; a screen with fewer than 22
  dates says PARTIAL and gives no flag. Expected MDE (s8): 0.63 hPa / 0.44 m/s training, 0.28 / 0.11 sampler.
- `XSET_DRYRUN=1 bash jupiter/xset_submit_row.sh <ROW>` prints the date groups and submits nothing.

## Lessons of the first cluster runs (17 Nov 2026)

- Start `xset_row.sh` detached (`setsid nohup bash K/xset_row.sh ... > W/logs/xset_row_<ROW>.log 2>&1 &`): the copy of a
  row from Jupiter takes about 20 minutes and dies with the ssh session that started it. A later call resumes.
- A smoke draws 2 members; the input field `x` written to the smoke file is cut to the members drawn (fixed after the
  first smoke, job 2966209, failed in the writer: 10 template members against 2 drawn).
- The account limit on Atos is 60 running jobs for all threads together; `scontrol top` is refused. Short score jobs wait.
- The station cache `W/cache/xset1_v1` was changed on 17 Nov 17:15-17:36 UTC by another set's job (30 dates merged,
  20250920 / 0923 / 0927 fetched again). Rows scored before and after are compared with
  `sbatch --job-name=xset_strk_<ROW> K/xset_station_recheck.sbatch <ROW>` (scores again into `W/scores_recheck/<ROW>`,
  prints STATIONS_SAME or STATIONS_DIFFERENT per date). Other sets must put their folders in `xset_set.env` of the
  submit directory: exported variables do not reach Atos batch jobs.

## Proof state (17 Nov 2026, Jupiter)

- GL512 through a row definition (`GL512_450k_def`, smoke 2968723) equals the first-row file of GL512_450k bit for bit:
  `XSET_COMPARE IDENTICAL first 2 members, 595474 points`, all five variables.
- FS242 (12 outputs, from-scratch paper run) smoke on its 75k export (2970004): EMA load 0 missing 0 unexpected, outputs
  read by name at columns [0, 1, 3, 4, 9], anchor equal to the dataset sample, sanity inside the bounds, `SMOKE_PASS`.
  A full FS242 row is then `bash K/xset_row.sh <ROW> FS242 <STEP>` (about 100 GH200-h, 1 h of scoring).
- Not yet run with a few-step (MeanFlow) model.
