# CorrDiff baseline (17 Nov 2026)

External generative baseline of the paper: NVIDIA PhysicsNeMo CorrDiff (regression U-Net + patched residual EDM
diffusion), **tag v2.2.2 (072465a1), wheel `nvidia-physicsnemo==2.2.2`**, example `examples/weather/corrdiff` (the
example is deprecated on main after v2.2.2 in favour of `regional_weather_diffusion`; v2.2.2 is the published CorrDiff
recipe and differs from main only by the deprecation warning). Published defaults (GEFS-HRRR pair), every change listed
in `make_cd_configs.py`. Hard date for the rows: Fri 27 Nov (plan s9 card 2); late = the paper ships without it and says so.

## Design

- Domain: the E2 box [10, 70] N x [-80, 40] E. Target grid 864 x 1728 (0.0694 deg), input grid 216 x 432 upsampled
  x4 (`cd_grid.py`). Regrid: 4-nearest inverse squared distance (as QM); back to evaluation points: bilinear.
- Data: E2-v4 box stores on Leonardo (hindcast years 2003-2023, leads 12-72 h, control member). Train <= 2021
  (15,870 rows), valid 2022 (864), test 2023 (546), as GL512 / RW50k. Outputs = GL512's 10; inputs = GL512's own
  input variables; invariants = the E2 forcings constant over rows.
- No lead-time conditioning (GL512 has none). Same normalisation scheme as the example (per-channel z-score).
- Inference: one draw per input member, rule-31 seed `derive_draw_seed(1234, date, lead, k-1)`, published sampler
  (stochastic EDM, 18 steps, patches 448, overlap 4, boundary 2).
- Scoring: NaN outside the box. Stations on region `e2_box_in` (box minus 0.5 deg) of `xset_score_stations.py`, every
  row rescored with it; cyclone / IMERG lines restricted to the cases where CorrDiff is finite (`cd_box_table.py`,
  to write when the first rows exist). Spectra on the box grid from the saved member-1 fields (`full` in cdout).

## Cost (to be measured by the smokes before the long runs)

Budget of the plan: 300-600 GH200-h. Published durations: regression 1,000,000 samples, diffusion 10,000,000
samples (x 4 patches). The smokes (`SMOKE=1`, 2,560 samples) give samples/s per node; set `--reg-samples` /
`--diff-samples` so both runs end by Tue 24 Nov, and state any cut in the paper. Disk: export ~1.5 TB on `$WORK`.

## Steps (Leonardo; `ac6-100 -> leonardo` is refused, copies go through the Mac)

1. Sandbox `$CD` (see `cd_env.sh`): `mkdir -p $CD/{logs,data,sets}`; copy this folder to `$CD/kit` and
   `kits/20261117-baselines/qm/qm_lib.py` to `$CD/qm/` (cd_grid imports it from `../qm`).
2. Code and venv (login node, uv, `$WORK` cache; login nodes kill processes past 600 CPU-s, so use `uv`):
   `git clone --depth 1 --branch v2.2.2 https://github.com/NVIDIA/physicsnemo.git $CD/physicsnemo`;
   `uv venv $CD_VENV --python 3.11`; `uv pip install nvidia-physicsnemo==2.2.2 torch<CUDA 12 build> -r
   $CD/physicsnemo/examples/weather/corrdiff/requirements.txt`. Check `python -c "import physicsnemo, torch;
   print(physicsnemo.__version__, torch.cuda.is_available())"` on a compute node.
3. Stores: write the target and input store paths (`out_hres_o1280.zarr`, `in_lres_o320_halo1p5.zarr` under `$E2V4`,
   find with `ls $E2V4 $E2V4/*`) into `$CD/E2V4_OUT_STORE`, `$CD/E2V4_IN_STORE`; forcings = `$E2/.../in_hres_o1280_forcings.zarr`.
   `python3 $CD_KIT/cd_export.py probe ...` must print `rows OK: {'train': 15870, 'valid': 864, 'test': 546}`.
4. Export: `cd_export.py init` (login or 1-task job), `sbatch --array=0-3 cd_export.sbatch`, then `cd_export.py stats`
   and `cd_export.py verify` (`VERIFY_OK 17280 rows`).
   The booster queue can hold the CPU array for many hours (17 Nov: 7 h, priority). Then also submit
   `for T in 0 1 2 3 0 1 2 3; do sbatch --dependency=singleton cd_export_serial.sbatch $T; done` (lrd_all_serial: one
   job at a time, 8 cores, 30 GB, 4 h, 8 parts per job; QOS caps 10 submitted per user). Fill is resumable, so the
   second pass finishes what the 4 h limit cut, and the booster array can still run beside it.
5. GL512's config (`~/dev/db_global_20261027/configs/db_global_512_300k.yaml` on Jupiter) copied to `$CD/gl512.yaml`;
   `python3 $CD_KIT/make_cd_configs.py --gl512-config $CD/gl512.yaml --data $CD_DATA --repo $CD_REPO
   --dataset-py $CD_KIT/cd_dataset.py --run-dir $CD/run` (prints the input variables; check 10 outputs).
6. Smokes (his word covers the batch): `sbatch --nodes=1 --time=00:30:00 --export=ALL,MODE=regression,SMOKE=1
   cd_train.sbatch`; read samples/s and memory; then the regression run; then the diffusion smoke and run with
   `CD_REG_CKPT` = the last regression `.mdlus`.
7. Inference inputs on Atos: `python3 cd_cut_inputs.py --work <set work A> --out <A>/corrdiff_in --date ...` for XSET-1
   and S25 (needs rows/ENS9 of the set), copy to `$CD/sets/{xset1,s25}/in`, `sbatch cd_predict.sbatch` per set, copy
   `out/` back to `<A>/corrdiff_out`, then `python3 cd_rows.py --work A --cdout <A>/corrdiff_out --date ...` and
   score `CORRDIFF` like any row (`xset_score.sbatch CORRDIFF`).

## Self-tests run (17 Nov, cloud, synthetic stores)

`cd_grid.py` check (bilinear exact on a linear field, NaN outside); export probe / init / fill in 2 parts / stats /
verify and the dataset reader on 12 synthetic rows: pass. Not run here: Hydra composition, PhysicsNeMo training and
sampling (no GPU, no hydra build); the Leonardo smokes cover them.

## Opt-in options of 23 Nov (branch `claude/corrdiff-lowcompute-20261123`, Jupiter copy `$CD/kit_20261123`)

Every option below is off by default, and the existing 46-channel checkpoints still load with the old options. The one
exception is the generator's refusal described in the first item.

- `make_cd_configs.py` writes the whole schedule of each stage (`total_batch_size`, `batch_size_per_gpu`, `lr`,
  `lr_rampup`, `lr_decay`, `lr_decay_rate`, `training_duration`, `save_checkpoint_freq`, `validation_freq`) and, for the
  diffusion, `P_mean` / `P_std` when given. Values that are not given come from the example's base files. It refuses
  `lr_decay < 1` with `lr_rampup <= 0`, the combination under which train.py drives the learning rate to zero, so the
  diffusion now needs `--diff-lr-rampup`, `--diff-lr-decay 1` or `--scale-schedule-to-batch` (the published ramp-up of
  1,000,000 samples and decay interval of 500,000 samples, read as values for batch 256, scaled to the chosen batch).
  It prints each schedule in samples and in optimizer updates, and the GPU counts that divide the batch evenly.
  `--name-suffix x` writes `config_training_e2v4x_*.yaml` with run folders `run/reg_x`, `run/diff_x`; `--out-conf-dir`
  writes them outside the PhysicsNeMo checkout (train.py then needs `--config-dir`, which `cd_train_jupiter_body.sh`
  adds when `CD_CONF_DIR` is set). `--time-inputs` and `--val-spread` set the dataset options below.
- `cd_dataset.py`: `time_inputs: true` appends 5 per-row channels after the 46 (cos/sin julian day, cos/sin local
  time, insolation) at the row's valid time, unnormalised. They reproduce the forcings GL512 reads exactly on the O1280
  points (`cd_check_time_inputs.py`, 23 Nov: maximum difference 0 on 6 rows at 00 and 12 UTC; the regrid to the box grid
  alone differs by up to 4e-3). `spread: true` picks `max_rows` rows evenly over the split (used for validation).
- `cd_predict.py` gives the dataset the valid time (start date 00 UTC plus lead) and takes `--sigma-max` / `--sigma-min`
  for the sampler (defaults 800 / 0.002 as before).
- `cd_widen_regression.py` adds input channels to a regression checkpoint with zero weights (output unchanged) and
  writes it as `RUN/checkpoints_regression/CorrDiffRegressionUNet.0.0.mdlus` of a new run folder; training then starts at
  0 samples with a fresh Adam state. `weights` (login) and `check` (GPU) verify it; `selftest` runs a small CPU case.
- `cd_residual_ladder.py`: residual RMS of regression checkpoints per variable, for 48 training and 48 validation rows
  (8 per lead), validation also per lead; restartable, one shard per GPU, `report` writes `ladder.json` and `ladder.md`.
- Jupiter scripts for `$CD/kit_20261123`: `cd_env_jupiter_x.sh`, `cd_ladder_jupiter.sbatch` (one node, 30 min:
  ladder, widening check, optional diffusion smoke of 640 samples), `cd_train_jupiter_x.sbatch` (state
  `state/chain_<mode>_x`, run `run/<reg|diff>_x`, nodes from the command line, successor with the same nodes and time,
  `CD_MAX_SEG` 1 by default).
- Self-test: `python3 cd_selftest.py --work DIR` (synthetic export; dataset options, widening, ladder; the ladder part
  needs the PhysicsNeMo venv) and `python3 cd_widen_regression.py selftest --work DIR`.
