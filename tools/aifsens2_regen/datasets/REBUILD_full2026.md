# Rebuilding the full-year 2026 AIFS-CRPS stores from scratch

This page is the complete recipe to reconstruct the three full-year 2026 training stores on the ECMWF Atos cluster.
It assumes nothing but this repository, the anemoi-datasets fork patches next to it, and MARS access.

## What the stores are

| store | content |
|---|---|
| `downscaling-ai-pf-enfo-0001-mars-o320-2026-2026-12h-6h-v1-aifscrps` | the archived AIFS-CRPS ensemble (class ai, model aifs-ens, expver 0001, stream enfo, type pf), all fifty perturbed members, 68 lane variables at O320, starts 00 and 12 UTC from 2026-01-01 00 UTC to 2026-09-09 12 UTC (504 starts), leads 6 and 12 h (1,008 samples). Version 1 of the model for every start up to 12 May 00 UTC, version 2 from 12 May 12 UTC. |
| `downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifscrps-validtime` | the operational O1280 analysis (class od, stream oper, type an) at the VALID time of every sample, one member, same 68 variables. |
| `downscaling-od-an-oper-0001-mars-o1280-2026-2026-12h-6h-v1-aifscrps-validtime-forcings` | O1280 land-sea mask and surface geopotential from the analysis plus nine computed forcings at the valid time. |

All three share one synthetic forecast axis: each (start, lead) sample is one hourly date from 1900-01-01 00:00, and
`.zattrs["fake_forecasts"]` maps every synthetic date back to `[start, lead]`. This is the same convention as the
summer stores of 2026-09-06 (`...-v1-aifsens2`), so a training configuration written for those opens these unchanged.

## Ingredients

1. The anemoi-datasets fork with the `fake_forecasts` feature and the analysis-at-valid-time MARS source: branch
   `feature/aifs-regen-2026-early` of `/home/ecm5702/dev/anemoi-datasets` (a superset of `feature/aifs-analysis-target`,
   itself on top of `feature/fake-hindcasts`). The two commits beyond `feature/aifs-analysis-target` are stored as
   patches in `anemoi_datasets_patches/`; apply them with `git am` onto commit d9f362c6 if the branch is lost.
2. A build venv that resolves `anemoi.datasets` to that fork: `/home/ecm5702/hpcperm/venvs/pristine-uv-x86_64`
   with `PYTHONPATH=<fork worktree>/src`. `build_early.sh` refuses to run if the fork does not resolve.
3. The three recipes in `recipes/aifs_in_2026full.yaml`, `recipes/an_target_2026full.yaml`,
   `recipes/forcings_2026full.yaml`. They differ from the summer recipes only in name, description and dates.
   The trap recorded on 2026-09-05 still applies: `build: {group_by: forecast}` must be present, or every synthetic
   date collapses into one retrieval group.

## The build, three commands

```bash
S=/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/datasets-build   # or any copy of this directory
cd $S
./build_early.sh --tag full2026 --recipes "aifs_in_2026full an_target_2026full forcings_2026full" \
    --dest /home/ecm5702/scratch/data/aifscrps_2026_full_20260910/training --nparts 28 --concurrency 14
# prints: SUBMITTED tag=full2026 init=<id> load=<id> finalise=<id>
sbatch --dependency=afterany:<finalise id> $S/verify_full2026.sbatch
```

`build_early.sh` writes run copies of the recipes and three sbatch files under `runs/full2026/`, then submits
`anemoi-datasets init` (three stores), a load array (each task loads one part of the starts for all three stores,
straight from MARS, cache on scratch) and `finalise` + `cleanup` + `inspect`. Every load part is idempotent, so a failed
part is rerun with the same array index. The destination must not already contain a store of the same name
(the script refuses, move the leftover aside). Recipes must keep their description on one quoted line.

`verify_full2026.py` checks the shared axis (1,008 synthetic dates), ensemble sizes 50/1/1, the 68 variables in the
summer order, no NaN, and bit-level agreement of one version 2 sample (7 July 12 UTC, member 7, +12 h) and one
version 1 sample (20 January 00 UTC, member 13, +6 h) with fields retrieved directly from MARS. It must print ALL_PASS.

## After verification

Copy the three stores to `/perm/ecm5702/datasets/aifscrps_2026_full_20260910/` (rsync, then compare file counts and
byte totals), record sizes from `.zattrs["total_size"]`, and update the epic note
`epics/downscaling-aifs-crps/in-progress/20260910_2026_perturbed_regeneration_build.md` in the docs repository.

## Storage decision

Working copy on scratch (Lustre, three-month purge; no pool setting needed), durable copy on perm (NFS, not for
training I/O). `/ec/ai` was not used because the data-owner agreement of 2026-09-05 is still owed.
