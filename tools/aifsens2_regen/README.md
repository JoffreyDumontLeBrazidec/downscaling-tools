# Regenerating AIFS ENS version 2 for early 2026

This directory holds the pipeline that rebuilds the ECMWF machine-learned
ensemble, public checkpoint `aifs-ens-crps-2.0`, for the period 2026-01-01 00
UTC to 2026-05-12 00 UTC.

Each forecast starts from one IFS ensemble member's own perturbed analysis.
Member 1 of a given start is initialised from member 1's analysis, member 2
from member 2's, and so on, and the control member is never used. That is the
whole point of the exercise: an ensemble regenerated from a single analysis
would be far too narrow at short lead times.

The campaign is 263 starts, at 00 and 12 UTC, ten perturbed members each, run
to twelve hours with output at six and twelve hours. That is 2,630 member
forecasts.

## Layout

| file | what it does |
|---|---|
| `calendar.py` | which forecast starts belong to which block, and which analysis times those starts need |
| `gribspec.py` | the composition of every file: parameters, levels, field counts, output encoding |
| `common.py` | shared helpers, including the rule that nothing is ever deleted |
| `retrieve.py`, `retrieve.sbatch` | stage 1, fetch the initial-condition fields from MARS |
| `assemble.py`, `assemble.sbatch` | stage 2, build one 222-field file per start and member |
| `run_forecasts.py`, `run_forecasts.sbatch` | stage 3, run the model on one GPU |
| `regrid.py`, `regrid.sbatch` | stage 4, select 68 variables and regrid N320 to O320 |
| `manifest.py` | stage 5, gather the records for a block and for the campaign |
| `verify_block.py`, `verify.sbatch` | stage 6, decide PASS or FAIL for a block |
| `select_summer.py` | decide which summer validation dates the archive could serve |
| `reproducibility.py` | check that a forecast repeats, that members differ, and that member m used member m's analysis |
| `env.sh` | the common environment every job sources |
| `env/` | the pinned `pyproject.toml` and `uv.lock` for the inference environment |

Every stage is invoked as `python -m aifsens2_regen.<stage>` from the `tools`
directory, never by running the file directly. This matters: the package
contains a module called `calendar.py`, and running a file directly would put
this directory on `sys.path` and shadow the standard library's `calendar`
module for every library that imports it.

## Blocks

A block is the unit of work that is retrieved, assembled, forecast and verified
together, and that can be restarted on its own.

| block | starts |
|---|---|
| `pilot_20260101` | 2, both starts of 1 January, used to prove the code |
| `summer_validation` | the summer 2026 dates the archive can serve, at 00 UTC only |
| `summer_validation_12utc` | the same five summer days, at 12 UTC |
| `202601` | 62 |
| `202602` | 56 |
| `202603` | 62 |
| `202604` | 60 |
| `202605` | 23, through 12 May 00 UTC |

## Where things are

| what | path |
|---|---|
| data root | `/home/ecm5702/scratch/data/aifsens2_regenerated_2026_early_20260910/` |
| inference environment | `/home/ecm5702/hpcperm/sandbox/20260910-aifsens2-2026-early/regen-env/.venv-x86_64` |
| checkpoint | `/home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/ckpt/aifs-ens-crps-2.0.ckpt` |
| land-sea mask | `/home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/ckpt/lsm.grib` |
| this code, deployed | `/home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/` |

Under the data root, `ic/raw/<block>/` holds the grouped MARS files,
`ic/members/<YYYYMMDD_HH>/mNN.grib` the per-member initial conditions,
`native_n320/<YYYYMMDD_HH>/mNN.grib` the full model output,
`derived_o320/<YYYYMMDD_HH>.grib` the file the dataset build reads, and
`manifests/` the records. The summer validation block keeps the same shapes
under `validation/`.

## Running a block

Substitute the block name for `202601` throughout. Ask the retrieval how many
requests the block needs before choosing the array ranges, because the number
depends on how many times of day the block's starts require.

```bash
cd /home/ecm5702/dev/downscaling-tools-aifsens2-regen/tools/aifsens2_regen
source env.sh    # only needed for the --list calls below

python -m aifsens2_regen.retrieve --block 202601 --groupset atm  --list   # 16 requests
python -m aifsens2_regen.retrieve --block 202601 --groupset wave --list   # 4 requests
```

Then, from that same directory:

```bash
# Stage 1, in two independent submissions.  The atmospheric fields are mostly
# online and the wave fields mostly on tape, so they are kept apart: a tape
# mount that takes hours must not delay a request that takes minutes.  The two
# concurrency limits add up to three, the project ceiling on simultaneous MARS
# requests.
ATM=$(sbatch --parsable --array=1-16%2 retrieve.sbatch 202601 atm)
WAV=$(sbatch --parsable --array=1-4%1  retrieve.sbatch 202601 wave)

# Stage 2, once both retrievals have finished.
ASM=$(sbatch --parsable --dependency=afterok:$ATM,afterok:$WAV assemble.sbatch 202601)

# Stage 3, one GPU, no array.
FC=$(sbatch --parsable --dependency=afterok:$ASM run_forecasts.sbatch 202601)

# Stage 4, back on CPU, so it can overlap with the next block's forecasts.
RG=$(sbatch --parsable --dependency=afterok:$FC regrid.sbatch 202601)

# Stages 5 and 6.
sbatch --dependency=afterany:$RG verify.sbatch 202601
```

Only one block's forecast job may be queued or running at a time, because the
project allows only one GPU in use across everything. Chain the next block's
forecast job behind the previous one with `--dependency=afterany:$FC` rather
than submitting both at once.

## Restarting after an interruption

Resubmit the same command. Every stage is idempotent, and idempotent in the
strong sense that it re-checks the file rather than trusting that it exists:

- `retrieve` skips a request whose target file already has exactly the expected
  number of fields, and retries up to three times otherwise, with a ten-minute
  pause between attempts.
- `assemble` re-validates each member file it finds and rebuilds any that fails.
- `run_forecasts` re-validates each output file and skips only those that pass;
  it reloads the checkpoint once and continues through the rest of the block.
- `regrid` re-validates each start's O320 file and rebuilds any that fails.

A partial artifact is never left under its final name: every stage writes to a
temporary name, validates, and only then renames. So a job killed mid-write
leaves a `.tmp` file that the next run moves aside, not a short file that a
later stage would trust.

Nothing in this pipeline deletes anything. When a stage has to discard a bad
file it renames it into a dated `_aside_<YYYYMMDD-HHMMSS>` directory beside it.
Those files keep occupying disk and have to be cleared by hand when the space
is wanted.

To rerun one member deliberately, for instance after a hardware fault:

```bash
python -m aifsens2_regen.run_forecasts --block 202601 \
    --only-start 20260103_12 --only-member 7 --force
```

## What "complete" means

`verify_block.py` looks at the files themselves rather than at the manifests, so
that a manifest written by a job that later crashed cannot make an incomplete
block look finished. A block passes only when every start has ten member
initial conditions of 222 fields, ten native forecasts of 238 fields, an O320
file of 1,360 messages with the right dates, members and lead times, and a
sample of the data that is not undefined.

A missing field is never filled in from the control member or from a
neighbouring member. A member that cannot be built stays absent, is reported in
the log and the manifest, and makes the job exit non-zero while the other
members are still produced.

## Reproducibility

Each forecast's random seed is derived from the start and the member alone:

    seed = int(sha256(f"{start:%Y%m%d%H}-m{member}").hexdigest()[:8], 16)

so rerunning one member reproduces it regardless of what else ran before it,
and the seed is recorded in the start's manifest.

Reusing one loaded checkpoint across many forecasts is only safe because three
pieces of anemoi-inference state are reset between runs, which was established
by reading the library rather than assumed:

- `Accumulate.accumulators` in the accumulation post-processor is never
  cleared, so total precipitation would grow across members if the
  post-processors were not rebuilt each time;
- `Runner.run` sets `self.reference_date = self.reference_date or date`, so the
  first start's date would be stamped on every later forecast's GRIB headers;
- the pre-processors and post-processors are plain attributes assigned in
  `Runner.__init__`, not cached properties, so rebuilding them is enough, while
  the model is a cached property and stays on the GPU.

`reproducibility.py` checks all of this from the outside: it runs one forecast
twice in separate processes and compares the outputs, confirms that two members
and two starts give different fields, and confirms by comparing distances in
two-metre temperature that member m's forecast really came from member m's
analysis.

## The two test blocks

These exist to prove the code before the campaign is produced, and they use
exactly the same modules as the monthly blocks.

The January pilot day, both starts of 1 January:

```bash
sbatch retrieve.sbatch pilot_20260101 atm     # already done, array 35238983
sbatch assemble.sbatch pilot_20260101
sbatch run_forecasts.sbatch pilot_20260101
sbatch regrid.sbatch pilot_20260101
sbatch verify.sbatch pilot_20260101
```

The summer validation dates. These do not go through `retrieve.sbatch`: five of
their nine field groups were fetched date by date by a probe, because a request
for the whole summer at once failed twice on an unavailable tape, and the other
four survive from the 2026-09-09 study and are read from there. So the first
step is to decide which dates the archive actually served:

```bash
python -m aifsens2_regen.select_summer          # writes validation/selected_dates.json
sbatch assemble.sbatch summer_validation
sbatch run_forecasts.sbatch summer_validation
sbatch regrid.sbatch summer_validation
sbatch verify.sbatch summer_validation
```

A date is selected only if all nine groups are present with exactly the
expected field counts. If more dates arrive from the archive later, re-run
`select_summer` and then the same four commands: every stage skips what already
validates, so only the new dates are produced.

The summer block keeps its own directories under `validation/`: `ic_members/`,
`native_n320/` and `derived_o320/`, with the same shapes as the production
areas.

The same five summer days again, this time at 12 UTC. This block is simpler to
retrieve than the 00 UTC one, because a 12 UTC start needs the analysis at 06
UTC and the analysis at 12 UTC of the same day, so nothing straddles a date
boundary and nothing has to be borrowed from the earlier study. All five of its
field groups came from one later probe, which wrote one file per group and date
holding both input times together, under
`validation/ic_raw_12utc/<YYYYMMDD>/<group>_<YYYYMMDD>_0600-1200.grib`.

```bash
python -m aifsens2_regen.select_summer --block summer_validation_12utc
sbatch assemble.sbatch summer_validation_12utc
sbatch run_forecasts.sbatch summer_validation_12utc
sbatch regrid.sbatch summer_validation_12utc
sbatch verify.sbatch summer_validation_12utc
```

A date is selected only if all five groups are present and carry exactly the
expected number of fields at **each** of the two input times: 130 surface, 700
pressure-level, 130 specific-humidity, 110 wave and 4 invariant fields per time,
so 260, 1400, 260, 220 and 8 messages per file. The count is checked per input
time rather than per file, because a file holding the right total at only one
time would be useless and would not announce itself.

This block writes into `validation/ic_raw_12utc/`, `validation/ic_members_12utc/`,
`validation/native_n320_12utc/` and `validation/derived_o320_12utc/`. The two
validation blocks are told apart in exactly one place, the `VALIDATION_BLOCKS`
table in `calendar.py`, which gives each block its start hour, the name of its
selection file and the suffix its directories carry. Every stage asks that
table through `calendar.validation_dir` rather than testing the block name
itself, so the two blocks can never write into each other's output.

## A note on the output GRIB headers

The regenerated fields identify themselves as class `ai`, stream `enfo`, type
`pf`, expver `rgn2`, generating process identifier 2, with the member number in
`number` and the accumulated fields carrying `stepType` `accum` from step 0.

Two things about that encoding were established by experiment rather than
assumed, and both are worth knowing before anyone edits it.

The `number` key cannot simply be set. The templates anemoi falls back on for
variables that are not in the input are deterministic analyses, whose product
definition template has no room for an ensemble member, and eccodes rejects
`number` outright with "Key/value not found". The encoding therefore sets
`eps: 1` first, which moves the message onto an ensemble product definition;
anemoi applies keys in the order given by `ORDERING` in `grib/encoding.py`,
which puts `eps` before `number`, so this ordering is guaranteed rather than
lucky.

The model name is not in the headers at all. eccodes 2.47.0 has no `model` key
and rejects both `model` and `modelName`. The name `aifs-ens` is recorded in
every manifest instead, under `model`, alongside `model_encoded_in_grib: false`
so that the gap is explicit. If a later eccodes gains the key, add it to the
encoding dictionary in `run_forecasts.base_config`.
