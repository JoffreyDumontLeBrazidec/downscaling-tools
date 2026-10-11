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

## Tests

`python -m pytest tools/chain` (15 tests, about 35 s on CPU): the resolver's decision table on synthetic
checkpoints, and an end-to-end chain through `tests/toy_job.sh` around a tiny Lightning trainer with anemoi's
start semantics: refuse without a manifest, fresh from a donor, resume with the same job file, `done`, the
guard stopping a weights-only resume, and the 2026-10-09 failure (a segment starting from the donor) stopped
before it trains or writes a second run. `tests/test_chain_next.py` runs the self-submitting job
`tests/toy_next_job.sh` under a fake Slurm (`tests/fake_slurm/`): one submission runs a whole chain, no progress
stops it after three segments, progress resets the count, `CHAIN.stop`, no duplicate of a pending segment, and no
`SLURM_*` leak into the successor's submission.
