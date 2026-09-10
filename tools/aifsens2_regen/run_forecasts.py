"""Stage 3: run the AIFS ENS version 2 forecasts on one GPU.

For every start of a block and every one of the ten perturbed members, this
runs a twelve-hour forecast from that member's own initial condition and writes
the complete native output, all 119 output fields at step 6 and at step 12, as
GRIB on the model's own N320 grid.

Loading the checkpoint costs about twenty-five seconds and the forecast itself
about five, so the checkpoint is loaded once and reused for every member of the
block.  Reusing an anemoi-inference runner across forecasts is safe only if
three pieces of state are reset between runs, and this was checked by reading
the library rather than assumed.

    The accumulation post-processor keeps a running total in
    Accumulate.accumulators and never clears it, so total precipitation would
    grow across members if the post-processors were not rebuilt.

    Runner.run sets reference_date with "self.reference_date or date", so the
    first start's date would be stamped on every later forecast's GRIB headers
    if reference_date were not reset.

    The pre-processors and post-processors are plain attributes assigned in
    Runner.__init__, not cached properties, so rebuilding them per run is
    enough; the model itself is a cached property and stays on the GPU.

Everything else the runner needs for a forecast, in particular the input
objects and the forcings, is rebuilt inside each run() call.

Each member's seed is derived from the start and the member number, so a rerun
of any single member reproduces the same numbers without depending on the order
in which members were run.  The seed is recorded in the manifest.

The output is written under a temporary name, validated for field count and for
finite values, and only then renamed.  A member whose final file already
validates is skipped, so the job is resumable after any interruption.  A member
whose initial condition is missing is reported and skipped; it is never
replaced by the control member or by a neighbour.

Usage
-----
    python -m aifsens2_regen.run_forecasts --block pilot_20260101
    python -m aifsens2_regen.run_forecasts --block summer_validation \
        --members-root .../validation/ic_members --native-root .../validation/native_n320
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import random
import sys
import time

from . import calendar as cal
from . import gribspec as spec
from .common import (
    CHECKPOINT,
    DEFAULT_ROOT,
    LSM_PATH,
    atomic_write_json,
    log,
    move_aside,
    sha256_file,
)


def member_seed(start: dt.datetime, member: int) -> int:
    """A deterministic seed for one start and one member.

    Derived from the text "YYYYMMDDHH-mN" so that it depends only on which
    forecast is being run, never on the order of execution or on the machine.
    """
    text = f"{start:%Y%m%d%H}-m{member}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def base_config(checkpoint: str, lsm: str) -> dict:
    """The version 2 inference configuration.

    This reproduces the configuration proven on 2026-09-09 in
    /home/ecm5702/scratch/eval/aifs_v1_vs_regen_20260909/regen/timing_v2/run_cfg.yaml,
    with the output encoding added so that the regenerated fields identify
    themselves as a machine-learned ensemble rather than as operational
    forecasts.  The date, the input path, the output path and the member number
    are filled in per forecast.
    """
    return {
        "checkpoint": checkpoint,
        "date": "2026-01-01T00:00:00",
        "lead_time": spec.LEAD_TIME_HOURS,
        "input": {"grib": "PLACEHOLDER"},
        "use_grib_paramid": True,
        "allow_nans": True,
        "pre_processors": [
            {"forward-transform-filter": "cos_sin_mean_wave_direction"},
            {
                "forward-transform-filter": {
                    "filter": "apply-mask",
                    "param": ["sd", "swvl1", "swvl2"],
                    "path": lsm,
                    "mask_value": 0,
                }
            },
        ],
        "post_processors": [
            {"backward-transform-filter": "cos_sin_mean_wave_direction"},
            "accumulate_from_start_of_forecast",
        ],
        "patch_metadata": {"dataset": {"constant_fields": ["z", "sdor", "slor", "lsm"]}},
        "typed_variables": {
            "mwd": {"mars": {"param": "mwd", "stream": "waef", "levtype": "sfc"}},
            "snowc": {"mars": {"param": "fscov", "stream": "enfo", "levtype": "sfc"}},
        },
        "write_initial_state": False,
        "output": {
            "grib": {
                "path": "PLACEHOLDER",
                # Two notes on this dictionary, both found the hard way on
                # 2026-09-10 rather than assumed.
                #
                # "eps: 1" has to be here.  The templates anemoi falls back on
                # are deterministic analyses, whose product definition template
                # has no room for an ensemble member, so eccodes rejects
                # "number" outright with "Key/value not found".  Setting eps
                # first moves the message to an ensemble product definition and
                # the member number then encodes.  anemoi applies these keys in
                # the order given by ORDERING in grib/encoding.py, which puts
                # eps before number, so this ordering is guaranteed.  For the
                # accumulated fields the library overrides the product
                # definition template with 11 by itself.
                #
                # "model" is deliberately absent: eccodes 2.47.0 has no such
                # key and rejects it, which failed every member of the first
                # pilot attempt.  See the note in gribspec.OUTPUT_MODEL.
                "encoding": {
                    "eps": 1,
                    "class": spec.OUTPUT_CLASS,
                    "stream": spec.OUTPUT_STREAM,
                    "type": spec.OUTPUT_TYPE,
                    "expver": spec.OUTPUT_EXPVER,
                    "generatingProcessIdentifier": spec.OUTPUT_GENERATING_PROCESS,
                    "number": 1,
                },
            }
        },
        "env": {"ANEMOI_INFERENCE_NUM_CHUNKS": 8},
    }


# --------------------------------------------------------------------------
# Validation of a native output file.
# --------------------------------------------------------------------------


def validate_native(
    path: str, start: dt.datetime, member: int, expected_per_step: int
) -> tuple[bool, list[str], dict]:
    """Check one native forecast file and measure how much of it is not a number.

    The checkpoint is run with allow_nans, because some variables, the wave
    fields in particular, are legitimately undefined over land.  So a field
    containing some missing values is normal and a field containing nothing but
    missing values is not.  The counts are reported either way.
    """
    import numpy as np
    from eccodes import (
        codes_get,
        codes_get_values,
        codes_grib_new_from_file,
        codes_release,
    )

    problems: list[str] = []
    per_step: dict[int, int] = {}
    nan_by_field: dict[str, int] = {}
    all_nan: list[str] = []
    numbers: set[int] = set()
    dates: set[int] = set()
    times: set[int] = set()

    with open(path, "rb") as f:
        while True:
            h = codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                step = codes_get(h, "endStep")
                name = codes_get(h, "shortName")
                level = codes_get(h, "level")
                per_step[step] = per_step.get(step, 0) + 1
                dates.add(codes_get(h, "dataDate"))
                times.add(codes_get(h, "dataTime"))
                try:
                    numbers.add(codes_get(h, "number"))
                except Exception:
                    pass
                values = codes_get_values(h)
                n_nan = int(np.count_nonzero(~np.isfinite(values)))
                if n_nan:
                    key = f"{name}{level if level else ''}@{step}"
                    nan_by_field[key] = n_nan
                    if n_nan == values.size:
                        all_nan.append(key)
            finally:
                codes_release(h)

    for step in spec.LEAD_STEPS:
        got = per_step.get(step, 0)
        if got != expected_per_step:
            problems.append(f"step {step} has {got} fields, expected {expected_per_step}")
    extra = sorted(set(per_step) - set(spec.LEAD_STEPS))
    if extra:
        problems.append(f"unexpected lead times present: {extra}")

    if dates != {int(start.strftime("%Y%m%d"))}:
        problems.append(f"dataDate is {sorted(dates)}, expected {start:%Y%m%d}")
    if times != {start.hour * 100}:
        problems.append(f"dataTime is {sorted(times)}, expected {start.hour * 100}")
    if numbers and numbers != {member}:
        problems.append(f"ensemble number is {sorted(numbers)}, expected {member}")
    if all_nan:
        problems.append(f"{len(all_nan)} fields are entirely missing: {sorted(all_nan)[:10]}")

    stats = dict(
        fields_total=sum(per_step.values()),
        fields_per_step=per_step,
        fields_with_some_missing=len(nan_by_field),
        missing_values_total=int(sum(nan_by_field.values())),
        fields_entirely_missing=sorted(all_nan),
    )
    return not problems, problems, stats


# --------------------------------------------------------------------------
# The worker.
# --------------------------------------------------------------------------


def expected_fields_per_step(runner) -> int:
    """How many fields the checkpoint writes at each lead time.

    Asked of the checkpoint rather than hard-coded, so that a different
    checkpoint is noticed instead of silently producing short files.  The
    reference value from the 2026-09-09 run is 119; a disagreement is logged
    loudly and the checkpoint's own answer is used.
    """
    n = None
    try:
        names = set(runner.checkpoint.output_tensor_index_to_variable.values())
        # The model predicts wave direction as a cosine and a sine component,
        # and the backward wave-direction post-processor turns each such pair
        # back into a single direction field before it is written.  So each
        # cos_X / sin_X pair in the output tensor accounts for one written
        # field, not two.  For this checkpoint that is cos_mwd and sin_mwd,
        # which is why 120 output variables become 119 written fields.
        pairs = sum(
            1
            for name in names
            if name.startswith("cos_") and f"sin_{name[4:]}" in names
        )
        n = len(names) - pairs
        # Some variables are also renamed on the way out, snowc being written
        # as fscov here, but a rename does not change how many fields appear.
    except Exception as exc:  # noqa: BLE001
        log(f"could not read the output variables from the checkpoint ({exc})")

    if n is None:
        log(
            "using the reference value "
            f"{spec.NATIVE_FIELDS_PER_STEP_EXPECTED} fields per lead time"
        )
        return spec.NATIVE_FIELDS_PER_STEP_EXPECTED
    if n != spec.NATIVE_FIELDS_PER_STEP_EXPECTED:
        log(
            f"WARNING the checkpoint writes {n} fields per lead time but the "
            f"reference run wrote {spec.NATIVE_FIELDS_PER_STEP_EXPECTED}; "
            f"using the checkpoint's value {n}"
        )
    else:
        log(f"checkpoint writes {n} fields per lead time, matching the reference run")
    return n


def make_runner(cfg: dict):
    """Build the runner and load the checkpoint onto the GPU."""
    import yaml
    from anemoi.inference.config.run import RunConfiguration
    from anemoi.inference.runners.default import DefaultRunner

    tmp = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"aifsens2_run_cfg_{os.getpid()}.yaml"
    )
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)
    config = RunConfiguration.load(tmp, [])
    runner = DefaultRunner(config)
    return runner


def run_one(runner, start: dt.datetime, member: int, ic: str, out: str) -> None:
    """Run one forecast, having reset every piece of per-forecast state."""
    runner.config.date = start
    runner.config.input.grib = ic
    runner.config.output.grib.path = out
    runner.config.output.grib.encoding["number"] = member

    # Reset the state that would otherwise leak from the previous forecast.
    runner.reference_date = start
    runner.pre_processors = runner.create_pre_processors()
    runner.post_processors = runner.create_post_processors()

    runner.execute()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", required=True)
    p.add_argument("--root", default=os.environ.get("AIFSENS2_ROOT", DEFAULT_ROOT))
    p.add_argument("--members-root", help="where the initial conditions live")
    p.add_argument("--native-root", help="where the native forecasts go")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--lsm", default=LSM_PATH)
    p.add_argument("--only-start", help="run a single start, as YYYYMMDD_HH")
    p.add_argument("--only-member", type=int, help="run a single member")
    p.add_argument("--force", action="store_true", help="rerun even if the output validates")
    a = p.parse_args(argv)

    import torch

    starts = cal.block_starts(a.block, a.root)
    if a.only_start:
        starts = [s for s in starts if cal.start_key(s) == a.only_start]
    if not starts:
        log(f"FATAL block {a.block} has no starts to run")
        return 1
    members = [a.only_member] if a.only_member else spec.MEMBERS

    from .assemble import members_root as default_members_root

    ic_root = a.members_root or default_members_root(a.root, a.block)
    if a.native_root:
        native_root = a.native_root
    elif a.block in cal.VALIDATION_BLOCKS:
        native_root = cal.validation_dir(a.root, "native_n320", a.block)
    else:
        native_root = os.path.join(a.root, "native_n320")

    log(f"block {a.block}: {len(starts)} starts x {len(members)} members")
    log(f"initial conditions: {ic_root}")
    log(f"native output:      {native_root}")
    log(f"checkpoint:         {a.checkpoint}")

    ckpt_sha = sha256_file(a.checkpoint)
    log(f"checkpoint sha256:  {ckpt_sha}")

    t_load0 = time.time()
    runner = make_runner(base_config(a.checkpoint, a.lsm))
    _ = runner.model  # force the checkpoint onto the GPU now, not on first use
    load_seconds = time.time() - t_load0
    log(f"checkpoint loaded in {load_seconds:.1f} s")

    per_step = expected_fields_per_step(runner)

    done = failed = skipped = missing_ic = 0
    for start in starts:
        key = cal.start_key(start)
        out_dir = os.path.join(native_root, key)
        os.makedirs(out_dir, exist_ok=True)
        manifest_path = os.path.join(out_dir, "manifest.json")
        manifest = {
            "block": a.block,
            "start": key,
            "start_iso": start.isoformat(),
            "checkpoint": a.checkpoint,
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_load_seconds": round(load_seconds, 2),
            "fields_per_step": per_step,
            "lead_steps": spec.LEAD_STEPS,
            "model": spec.OUTPUT_MODEL,
            "model_encoded_in_grib": spec.OUTPUT_MODEL_IS_ENCODED,
            "grib_identification": {
                "class": spec.OUTPUT_CLASS,
                "stream": spec.OUTPUT_STREAM,
                "type": spec.OUTPUT_TYPE,
                "expver": spec.OUTPUT_EXPVER,
                "generatingProcessIdentifier": spec.OUTPUT_GENERATING_PROCESS,
            },
            "members": {},
        }
        previous = {}
        try:
            import json

            with open(manifest_path) as f:
                previous = json.load(f).get("members", {})
        except Exception:
            previous = {}

        for m in members:
            tag = f"m{m:02d}"
            ic = os.path.join(ic_root, key, f"{tag}.grib")
            out = os.path.join(out_dir, f"{tag}.grib")
            seed = member_seed(start, m)

            if not os.path.exists(ic):
                missing_ic += 1
                log(
                    f"MISSING_IC {key} {tag}: no initial condition at {ic}; "
                    f"this member stays absent and is not substituted"
                )
                manifest["members"][tag] = dict(
                    status="missing_initial_condition", seed=seed, initial_condition=ic
                )
                continue

            if os.path.exists(out) and not a.force:
                ok, problems, stats = validate_native(out, start, m, per_step)
                if ok:
                    skipped += 1
                    done += 1
                    log(f"SKIP {key} {tag}: existing output validates ({stats['fields_total']} fields)")
                    manifest["members"][tag] = previous.get(tag) or dict(
                        status="complete",
                        seed=seed,
                        path=out,
                        bytes=os.path.getsize(out),
                        note="already present and valid, not rerun",
                        **stats,
                    )
                    continue
                move_aside(out, "present but failed validation: " + "; ".join(problems)[:200])

            tmp = f"{out}.tmp.{os.getpid()}"
            if os.path.exists(tmp):
                move_aside(tmp, "leftover temporary output from an earlier attempt")

            set_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                run_one(runner, start, m, ic, tmp)
                rc_error = None
            except Exception as exc:  # noqa: BLE001 - one member must not kill the block
                rc_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.time() - t0
            peak_gib = torch.cuda.max_memory_allocated() / (1 << 30)

            if rc_error:
                failed += 1
                log(f"FAILED {key} {tag} after {elapsed:.1f} s: {rc_error}")
                if os.path.exists(tmp):
                    move_aside(tmp, "forecast raised an exception")
                manifest["members"][tag] = dict(
                    status="failed", seed=seed, error=rc_error,
                    seconds=round(elapsed, 2),
                )
                continue

            t_val = time.time()
            ok, problems, stats = validate_native(tmp, start, m, per_step)
            validate_seconds = time.time() - t_val

            if not ok:
                failed += 1
                log(f"INVALID {key} {tag}: {'; '.join(problems)}")
                move_aside(tmp, "failed validation: " + "; ".join(problems)[:200])
                manifest["members"][tag] = dict(
                    status="invalid", seed=seed, problems=problems,
                    seconds=round(elapsed, 2), **stats,
                )
                continue

            size = os.path.getsize(tmp)
            os.replace(tmp, out)
            done += 1
            log(
                f"OK {key} {tag}: {elapsed:.1f} s forecast, "
                f"{validate_seconds:.1f} s validation, "
                f"{size / (1 << 20):.0f} MiB, peak GPU {peak_gib:.1f} GiB, seed {seed}"
            )
            manifest["members"][tag] = dict(
                status="complete",
                seed=seed,
                path=out,
                bytes=size,
                initial_condition=ic,
                initial_condition_sha256=sha256_file(ic),
                seconds_forecast=round(elapsed, 2),
                seconds_validation=round(validate_seconds, 2),
                peak_gpu_gib=round(peak_gib, 2),
                **stats,
            )

        manifest["complete"] = sum(
            1 for e in manifest["members"].values() if e.get("status") == "complete"
        )
        atomic_write_json(manifest_path, manifest)
        log(f"START_SUMMARY {key}: {manifest['complete']} of {len(members)} members complete")

    want = len(starts) * len(members)
    log(
        f"RUN_SUMMARY block={a.block} complete={done} of {want} "
        f"(of which {skipped} were already present), failed={failed}, "
        f"missing initial conditions={missing_ic}"
    )
    if done != want:
        log(f"RUN_RC=1 block={a.block}: the block is NOT complete")
        return 1
    log(f"RUN_RC=0 block={a.block}: every member of every start is present and valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
