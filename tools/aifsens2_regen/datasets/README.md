# Early-2026 AIFS ENS version 2 training stores

This directory holds everything needed to build the three anemoi stores that extend the existing
summer-2026 training data backwards over 1 January to 12 May 2026, and to open the two periods
together as one training view.

## What the stores are

The downscaling lane takes an AIFS ENS forecast at grid O320 as input and the operational
analysis at grid O1280, at the forecast valid time, as target. The summer stores cover 12 May
12 UTC to 31 August 12 UTC 2026 with fifty operational members. For the earlier part of the
year AIFS ENS version 2 was not yet operational, so the forecasts were regenerated on the
cluster with the public version 2 checkpoint, started from the IFS ensemble perturbed initial
conditions, and written as GRIB under expver `rgn2`. Those regenerated forecasts give ten
members, not fifty.

Three stores are built, all on one shared synthetic forecast axis:

| store | contents | ensemble |
|---|---|---|
| `downscaling-ai-pf-enfo-rgn2-mars-o320-2026-2026-12h-6h-v1-aifsens2-early` | input, 68 variables at O320 | 10 |
| `downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifsens2-early-validtime` | target, 68 variables at O1280 | 1 |
| `...-early-validtime-forcings` | 11 forcings at O1280 | 1 |

The period is 263 forecast starts every 12 hours, each with leads of 6 and 12 hours, so 526
samples.

## The synthetic forecast axis

anemoi stores are indexed by date, but this lane trains on (start, lead) pairs. The fork of
`anemoi-datasets` therefore lays the samples out on a synthetic axis: each sample is one
synthetic date, one hour apart, and the store attribute `fake_forecasts` records which real
(start, lead) pair each synthetic date stands for.

The summer axis runs from 1900-01-01T00:00:00 to 1900-01-19T13:00:00. The early recipes set
`fake_forecasts_origin: 1900-01-19T14:00:00`, one hour later, so the two axes join end to end
and the two stores concatenate with no gap and no overlap. Note that the calendar order is
reversed with respect to the synthetic order: the summer half comes first on the synthetic axis
even though it is later in 2026. Nothing in the lane depends on the synthetic order, only on the
`fake_forecasts` mapping.

## Fork changes

The two changes live on the `anemoi-datasets` branch `feature/aifs-regen-2026-early`, which
starts from `feature/aifs-analysis-target` at `d9f362c6`. That repository has no writable remote,
so the commits are also exported here as patch files under `anemoi_datasets_patches/`, and can be
replayed onto `d9f362c6` with `git am`.

1. `FakeForecastsDates` accepts an optional `dates` key `fake_forecasts_origin`. Absent, the axis
   still starts at 1900-01-01T00:00:00 and behaviour is unchanged.
2. The `grib` source gains a `fake_grib` variant. The plain source selects fields by
   `valid_datetime`, which cannot work when the requested dates are synthetic. `fake_grib` maps
   each synthetic date back to its (start, step) pair, substitutes the start into the path
   pattern, selects the messages by their real GRIB keys, and checks the field count.

## Files

- `recipes/aifs_in_early.yaml`, `recipes/an_target_early.yaml`, `recipes/forcings_early.yaml` —
  the three build recipes for the full early period.
- `build_early.sh` — submits init, then a load job array, then finalise plus inspect plus verify.
  Parameterised so the same script builds a block subset into a separately named store.
- `verify_early.py` — the acceptance gates for the three stores.
- `combined_view.yaml` — the open-time specification of the combined summer plus early view.
- `test_combined_view.py` — opens that view and checks it is what training expects.
- `run_test_input.sbatch`, `run_test_mars.sbatch` — the two jobs that build the tiny fixture
  stores used to test the whole chain before the real O320 files exist.

## Running the real build

Once the O320 GRIB files are in place under
`/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/derived_o320/`:

```
/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/build_early.sh
```

January alone, as a pilot:

```
/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build/build_early.sh \
    --suffix -jan --start "2026-01-01 00:00:00" --end "2026-01-31 12:00:00" --nparts 6
```

Then the combined-view test on the real stores:

```
python test_combined_view.py --mode combined
```
