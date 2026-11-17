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
   `sbatch --array=0-31 cd_export_serial.sbatch` (lrd_all_serial, 4 cores, 4 h, one part per task); fill is resumable,
   so the two arrays can coexist. A part cut by the 4 h limit: resubmit that task id.
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
