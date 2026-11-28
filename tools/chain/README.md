# Chain resume method (every cluster)

A training chain is one run trained by several Slurm segments (time limits). On 2026-10-09 the Atos resume
segments of weight decay 0, Muon, SOAP and the patch decoder were found starting from the donor instead of
resuming: the mode was given in front of `sbatch` (`WK_MODE=resume sbatch ...`), Atos sets
`SBATCH_EXPORT=NONE` so the variable never reached the job, and the job file fell back on its default, a warm
start from the donor. Weight decay 0 trained a donor restart for hours.

The cause is general, not Atos-only: **the segment's mode came from the submitter, and the fallback was the
destructive start.** Any lost variable, wrong default, copied job file or moved run does the same on any
cluster. This method removes the mode variable altogether.

## The method

1. **One job file for every segment.** The file carries no mode. Segments differ only by their
   `--dependency=afterany:<previous>`; the chain is the same file submitted N times.
2. **The mode is read from disk, at the segment's start.** The checkpoint root (the folder holding
   `<run_id>/last.ckpt`) carries a manifest `CHAIN.json`. `chain_state.py resolve` decides:
   - no manifest: **refuse** (exit 3). A chain is declared once, on the login node, with `init` (new chain)
     or `adopt --run-id` (existing or moved run). A typo in the root, a run copied without its manifest,
     or a forgotten init never becomes a fresh start;
   - manifest, no run yet: `fresh` (the body's own donor or fork start; expected step 0);
   - the chain's run has `last.ckpt`: `resume` that run id; expected step = the `global_step` stored in
     that `last.ckpt` (read without torch);
   - that step is at or past the end step: `done`, exit 0 without training;
   - the recorded run has no `last.ckpt`, or a second run folder with checkpoints appeared, or two runs and
     none recorded: **refuse** (exit 3) and say which.
3. **The start step is checked after the checkpoint loads.** Training is launched through
   `chain_guard_launch.py --expect-step N`, which adds a Lightning callback: at `on_train_start`, after the
   restore, `trainer.global_step` must equal N, or every rank stops before the first optimiser step. This
   catches what the resolver cannot see: a lane YAML forcing `load_weights_only` or a `warm_start` (weights
   load, the step and optimiser restart at 0), or anemoi's MLflow dry-run path, which turns
   `start_from_checkpoint` off and trains from scratch (`train.py` `_check_dry_run`). No anemoi change; works
   on any branch.
4. **After training, `record`** writes the run id into the manifest (first segment) and fails loudly if a
   second run appeared. `CHAIN.log` beside the manifest keeps one line per resolve, record and refusal.

Exit 0 on `done` lets an `afterany` chain run past its end harmlessly. Exit 3 stops a segment in seconds,
with the reason in the job's `.err`, and the next segment refuses the same way until a person fixes the state.

## Use

On the login node, once per chain (Python 3.6 is enough):

```bash
python3 chain_state.py init  --root <ckpt_root> --note "fork from donor 12dcefea, NL35, 100k"
python3 chain_state.py adopt --root <ckpt_root> --run-id <id>     # a chain already running, or a moved run
python3 chain_state.py status --root <ckpt_root>
```

In the job body, after activating the venv (copy `tools/chain/` into the sandbox's `jobs/chain/` and record the
commit, so the sandbox stays self-contained):

```bash
source "$EXP/jobs/chain/chain_segment.sh"
chain_resolve "$CKPT_ROOT" "$END_STEP"               # exits 0 if done, 3 if refused
if [[ "$CHAIN_ACTION" == fresh ]]; then
  start=("system.input.warm_start=$DONOR" "training.transfer_learning=True" "training.load_weights_only=True"
         "training.run_id=null" "training.fork_run_id=null")
else
  mapfile -t start < <(chain_resume_overrides)       # training.run_id=$CHAIN_RUN_ID, no fork, no warm start
fi
set +e
srun --kill-on-bad-exit=1 python "$CHAIN_TOOLS/chain_guard_launch.py" --expect-step "$CHAIN_EXPECT_STEP" \
     --root "$CKPT_ROOT" -- anemoi-training train "${overrides[@]}" "${start[@]}"
rc=$?
set -e
chain_record "$CKPT_ROOT"
exit $rc
```

`<entry>` after `--` may be a console script (`anemoi-training train`), a wrapper script
(`$EXP/jobs/train_with_peakmem_h9.py`), or `-m <module>`. Submit the same file N times:

```bash
j=$(sbatch --parsable job.sbatch); for i in 2 3 4; do j=$(sbatch --parsable --dependency=afterany:${j%%;*} job.sbatch); done
```

### On-demand segments (one submission per chain)

Instead of pre-submitting N segments, the job body may call `chain_next "$CKPT_ROOT"` right after `chain_resolve`.
The segment then queues the same job file once more with `--dependency=afterany:<this job>`, before it trains, so
a segment that dies in its first minutes still has its successor. The chain is submitted once
(`sbatch job.sbatch`) and runs until `done`. Idea borrowed from ecmwf/ml-conductor's requeue loop, without its
always-on supervisor (docs `epics/misc/20261011_ml_conductor_evaluation.md`).

No successor is queued when:
- the chain is `done` or refused (the segment exits before `chain_next`);
- `CHAIN.stop` exists in the checkpoint root (also makes any queued segment exit 0 at once): **to stop a chain,
  `touch <root>/CHAIN.stop`**, then `scancel` the running segment if it should not finish. A bare `scancel` of the
  running segment is not a stop: its `afterany` successor starts;
- three segments in a row started at the same step (no checkpoint written by the last two): the chain is not
  progressing and needs a person (`--max-same-start`, second argument of `chain_next`);
- a pending job already runs the same command (an older pre-submitted series, or a resubmission by hand);
- `sbatch` fails (warning in the `.err`, line in `CHAIN.log`; the current segment still trains).

Tested on the clusters on 2026-10-11 (4-job toy chain 0→3000, one `sbatch`): Atos AC (`ecsbatch` wrapper works
inside a job), AG, Leonardo, Jupiter and MN5 (`acc_ehpc`) ran the whole chain. Queue limits that
break self-submission: **MN5 `acc_debug` has MaxSubmitPU=1**, so a running job can never queue its successor (the
segment warns and the chain ends): use `acc_ehpc`. Atos `dg` allows one running job per account, shared by all our
sessions.

Each decision is a line in `CHAIN.log` (`next queued job=...`, `next already queued`, `next NOT submitted`,
`next-check REFUSED`). The job's own `SLURM_*` variables (except `SLURM_CONF`) are removed from the successor's
`sbatch`; it is submitted from `SLURM_SUBMIT_DIR`, with the job file's own `#SBATCH` lines.

Moving a run to another cluster: copy the run folder and `CHAIN.json` together (or `adopt` on arrival).
Extending a run past its planned end: raise `END_STEP` in the job file (and switch the loss per the
standing rules); the resolver then resumes instead of reporting `done`.

## Overlap mode (`chain_handover.py`, 28 Nov 2026)

Why: on MareNostrum 5 a job held by a dependency gets `AccrueTime=Unknown` and no `sprio` row (PriorityFlags
empty, age weight 100000 = as large as fair share), so an `afterany` successor starts queueing with zero age
when its predecessor ends. Our 2-node `acc` jobs waited a median 40.2 h once eligible when they had been held
(n=8), 8.9 h when submitted free (n=13), last 60 days (docs `scripts/queue_probe/`).

The successor is queued with NO dependency (optionally `--begin=now+S`) and takes the run over when it starts:

```bash
source "$EXP/jobs/chain/chain_segment.sh"
chain_takeover "$CKPT_ROOT"                     # FIRST: wait for the running writer's next checkpoint, scancel it,
                                                #  wait until Slurm no longer lists it; exit 3 if that fails
chain_resolve  "$CKPT_ROOT" "$END_STEP"
chain_next_overlap "$CKPT_ROOT" "$BEGIN_DELAY_S"   # instead of chain_next
```

One writer: `CHAIN.writer` names the job that trains; `.chain_takeover.lock` (mkdir) serialises takeovers; the
new job trains only after `squeue`/`sacct` stop listing the old one (a query error is never read as gone). The
old writer is stopped with a plain `scancel` (SIGTERM). Never `--signal=USR1`: pytorch_lightning's SLURM plugin
answers USR1 by saving an `hpc_ckpt` and REQUEUEING the job, which would bring a second writer back later. A
requeued (PENDING) writer found at takeover is cancelled. Options: `--timeout` (default 1.5 h: then
`--on-timeout kill` or `refuse`), `--stable` (90 s unchanged before the cancel), `--kill-wait` (900 s).
Cost per handover: the successor's GPUs idle until the next checkpoint plus its start-up. Set the begin delay
from the cluster's measured waits so handovers happen late in a segment, not minutes after its start.
Tests: `tests/test_handover.py` (fake Slurm). Real-Slurm smoke without GPUs: `smoke/` (CPU partitions,
`overlap_smoke_audit.py` checks that writers never interleave).

### Mode on disk, pools of chains, rollout scripts (28 Nov 2026)

- `chain_next_auto ROOT` replaces `chain_next` in a body that should serve every mode: `ROOT/CHAIN.pool` (or a pilot
  job) -> queue nothing; `ROOT/CHAIN.overlap` (first word = `--begin` delay in s) -> overlap successor; else classic
  `afterany` (chain_next). `chain_takeover` before `chain_resolve` is harmless in classic mode.
- Pilot pool (`chain_pool.py`, `pilot_job.sh`): N chains with the SAME resources share generic pilot jobs. A pilot
  that starts trains the idle chain (oldest checkpoint first), else the chain whose writer has the least time left,
  claims it (`CHAIN.claim`), queues the next pilot while fewer than (active chains) are pending, then runs that
  chain's body. Pool file: `root<TAB>end_step<TAB>body [args]`. Tests `tests/test_pool.py`; smoke
  `smoke/pool_smoke_setup.sh`.
- `rollout/`: `edit_body.py` (exact-anchor body edits, refuses on doubt), `apply_mn5_muon_overlap.sh`,
  `apply_leo_useq_pool.sh` (dry by default, `--apply` to change; they never submit or cancel).

## Input store guard (`store_guard.py`, `chain_store_guard`)

On 2026-11-09 the Atos scratch purge (each file deleted about 30 days after its last access) removed single row
chunks from the E2-v4 box stores and the E2 forcings store, and the small `count/` and `has_nans/` arrays. Zarr
returns the fill value (0.0) for a missing chunk, so a job reading those rows trains or scores on zeros without
any error. R4's job body had no check.

Call the guard in every job body that reads a zarr store from purgeable storage, before `chain_resolve`:

```bash
source "$EXP/jobs/chain/chain_segment.sh"
chain_store_guard --config "$EXP/config/run.yaml"     # every `dataset: <...>.zarr` in the config; exit 49 if not complete
```

It reads only directory listings and the `.zarray` files (no chunk, so no access time is refreshed) and checks,
for every array of every store: `.zarray` present, chunk files = the count the shape and chunks imply (dot-files
and rsync temporaries ignored). `STORE_GUARD_ALLOW_INCOMPLETE=1` downgrades the refusal to a warning, for timing
smokes only. Repair lists: `python3 store_guard.py missing <store> --ranges` prints the missing row indices of
`data`. A store written with skipped all-fill-value chunks would be reported as incomplete; anemoi-datasets
stores write every chunk.

## Tests

`python -m pytest tools/chain` (25 tests, about 40 s on CPU; `tests/test_store_guard.py` covers the store guard): the resolver's decision table on synthetic
checkpoints, and an end-to-end chain through `tests/toy_job.sh` around a tiny Lightning trainer with anemoi's
start semantics: refuse without a manifest, fresh from a donor, resume with the same job file, `done`, the
guard stopping a weights-only resume, and the 2026-10-09 failure (a segment starting from the donor) stopped
before it trains or writes a second run. `tests/test_chain_next.py` runs the self-submitting job
`tests/toy_next_job.sh` under a fake Slurm (`tests/fake_slurm/`): one submission runs a whole chain, no progress
stops it after three segments, progress resets the count, `CHAIN.stop`, no duplicate of a pending segment, and no
`SLURM_*` leak into the successor's submission.

## Triton kernel guard (rule 47, 17 Nov 2026)

`kernel_guard.py` refuses the old (pre-bbca1748) Triton GraphTransformer kernel `anemoi/models/triton/gt.py`.
The test is a git-blob ALLOWLIST (`ALLOWED_GT_BLOBS`, 5 fixed blobs on 17 Nov 2026, ds-trunk = `88b036167a14`):
any other gt.py, old or hand-edited, stops the process with exit 97 before the module body runs. Not an
ImportError, because anemoi falls back to another backend on ImportError.

It is armed in four places, so no single forgotten line lets the old kernel run:

| where | when | how |
|---|---|---|
| `chain_resolve` (chain_segment.sh) | job start, before the resolve | `$KERNEL_GUARD_PY kernel_guard.py check` (default `python` = the venv's) |
| `chain_guard_launch.py` | every srun rank, before the entry point | import hook in-process + check |
| `sitecustomize.py` / `python -m eval.cli` | every eval / inference interpreter | import hook |
| every shared and run venv on the clusters | every interpreter of that venv | `python kernel_guard.py install-hook <site-packages>` writes `_ds_kernel_guard.py` + `.pth` |

A job body that does not use the chain tools adds one line after activating its venv:
`python "$TOOLS/tools/chain/kernel_guard.py" check || exit 97`.
Inventory of files: `python3 kernel_guard.py file PATH...` (FIXED / OLD / UNLISTED-FIX-MARKERS).
A new fixed variant joins the allowlist only after `classify` shows both markers and the one-hot bf16 gradient
test passes (anemoi-core `models/tests/models/test_encoder_trap.py`). Tests: `tests/test_kernel_guard.py`.
