# Baselines of the 36 -> 9 km paper (17 Nov 2026)

Owner's pick (plan card 2, 17 Nov 10:31 UTC): classical + regression + CorrDiff. Every baseline is scored with the
XSET kit (`kits/20261116-extremes-set`) on two sets, against the same references (ENS9 = 9 km IFS ENS, INPUT = native
36 km input, GL512_450k = the paper's model):

| set | dates | what | work folders |
|---|---|---|---|
| XSET-1 | 42 dates 2024-2025 | extremes and cyclones (65 storms) | Atos `/ec/res4/scratch/ecm5702/agent-work/20261116-extremes-set`, Jupiter `.../agent-work/20261116-extremes-set`, outputs `.../outputs/xset1_20261116` |
| S25 | 2025-09-01..30 (= prepml jdyb dates) | the paper's shared dev set: 16 storms, 379 TC cases, 87 rain events | Atos `/ec/res4/scratch/ecm5702/agent-work/20261117-s25-set`, Jupiter `.../agent-work/20261117-s25-set`, outputs `.../outputs/s25_20261117` |

Rows: `QM` (classical), `REG_GL512_30k` (regression), `CORRDIFF` (E2 box only). The input + lapse-rate row is not a
separate row: the station scorer already lapse-corrects 2t for every row (0.0065 K/m), so it equals INPUT at stations.
Same seeds everywhere: `derive_draw_seed(1234, date, lead, member-1)` (rule 31); one realisation per input member.

## Folders

- `s25/` S25 lists (`make_s25_lists.py`, from the XSET selectors; `lists/` committed).
- `qm/` quantile mapping (Atos CPU): `qm_extract` (training-store pairs per month) -> `qm_fit` (3-month window
  tables, 33 quantiles, per point) -> `qm_apply` (rows + histograms). Synthetic test passed.
- `reg/` regression (Jupiter): GL512 450k EMA-2000 warm start, `training_approach: deterministic`, 30k steps.
  `SETUP.md` = sandbox build; read with one denoiser call (`SAMPLER_MODE=det`).
- `corrdiff/` CorrDiff (Leonardo): `README.md`.

## Environment of the XSET kit for S25

Every XSET script reads these (defaults = XSET-1):

```
# Atos
export XSET_WORK=/ec/res4/scratch/ecm5702/agent-work/20261117-s25-set
export XSET_LISTS=<checkout>/kits/20261117-baselines/s25/lists
export XSET_CACHE=$XSET_WORK/cache/s25_v1
export XSET_DATES=$XSET_LISTS/dates_s25.csv
# Jupiter
export XSET_JWORK=/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261117-s25-set
export XSET_JOUT=/e/data1/jureap-data/ecmwf/users/jdumont/outputs/s25_20261117
```

## Run order

1. **S25 references (Atos).** Orography: copy the 49r1 GRIB of XSET-1 (`$XSET-1 work/grib/`) into `$XSET_WORK/grib/`.
   `sbatch xset_station_cache.sbatch $XSET_DATES` -> job C; `bash xset_chain_submit.sh $XSET_DATES 2 C ret` (retrieve,
   input stores, points, ENS9 rows). Check 180 ENS9 files (30 dates x 6 leads).
2. **S25 GL512_450k (Jupiter, no new GPU run).** Taken from the MI-S25 month row (same predictor settings, rule-31
   seeds, GlobalNoise; uint16-packed, max error span/131070): after `xset_copy_to_jupiter.sh` (templates), on Jupiter
   `python3 s25/mi_to_rows.py --mi /e/data1/jureap-data/ecmwf/users/jdumont/outputs/month_inference_s25/rows/GL512_450k
   --templates $XSET_JOUT/rows/ENS9 --out $XSET_JOUT --date ...` (CPU job), then `xset_copy_from_jupiter.sh GL512_450k`.
   Fallback if MI is late: `jupiter/xset_predict.sbatch GL512_450k <dates>` (~70 GH200-h).
3. **QM (Atos, CPU, no word needed beyond the batch).** `sbatch --array=1-12 qm/qm_extract.sbatch`, then
   `sbatch --array=1-12 qm/qm_fit.sbatch` (afterok), then `sbatch qm/qm_apply.sbatch <set work> <dates>` per set.
4. **REG (Jupiter, his word: new training).** `reg/SETUP.md`; then rows `REG_GL512_30k` (and `REG_GL512_20k`) on both
   sets with `jupiter/xset_predict.sbatch`.
5. **CorrDiff (Leonardo, his word: new training).** `corrdiff/README.md`.
6. **Scoring (Atos).** `sbatch --job-name=xset_score_<ROW> xset_score.sbatch <ROW>` for ENS9 refs, INPUT, GL512_450k,
   QM, REG_GL512_30k, CORRDIFF on each set; `bash xset_table.sh GL512_450k QM REG_GL512_30k CORRDIFF`. The station
   region `e2_box_in` (added 17 Nov) is the CorrDiff comparison; XSET-1 rows scored before 17 Nov need a rescore for it.
7. **Spectra.** Same instrument for every row (spectra standard, rule 14); CorrDiff on its box grid from the member-1
   fields, compared with GL512 and ENS9 regridded to the same grid.

Dates: all rows before the writing freeze Thu 26 Nov 00 UTC; CorrDiff hard date Fri 27 Nov. Record:
docs `epics/paper-o320-o1280/in-progress/20261117_baselines.md`.
