"""Observation CRPS evaluator — run phase.

Scores an FDB ensemble against surface station observations and reports the
result as a function of lead time, averaged over the forecast start dates in the
window.  This is the same measurement quaver makes for surface parameters, and
it was calibrated against quaver before being wired in: on ja6y (2 m
temperature, northern hemisphere) it matched quaver station-for-station and
agreed on the score to 0.09 per cent on average.

Why it exists next to the quaver evaluator: quaver is a multi-hour, FDB-bound
job that returns a PDF, so it cannot be run inside a normal iteration loop and
its numbers cannot reach a scoreboard.  This evaluator returns numbers in
minutes.  It does not replace quaver, which remains the canonical scorecard and
covers upper air as well.

The scoring window (expver, dates, members, lead times, grid) is resolved from
the run's effective_config.json with the SAME helpers the quaver evaluator uses,
so the two always score the same window.  Like quaver, this self-skips for
manual runs that never published an ensemble to FDB, which makes it safe to keep
in a lane's default evaluator group.

The compute runs under the ECMWF module environment rather than the eval venv,
because it needs `vtb` for the observation database and `mars` for the fields.
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from eval.evaluators.quaver.runner import (
    _load_effective_config,
    _resolve_lead_range,
    _sorted_ints,
    resolve_params,
)

LOG = logging.getLogger(__name__)

_BACKEND = Path(__file__).resolve().parent.parent.parent / "_backends" / "obs_crps"
_COMPUTE = _BACKEND / "obs_crps_compute.py"

# Surface parameters STVL holds and quaver scores. Precipitation is deliberately
# absent: it needs the SEEPS climatology and an accumulation period, which is a
# separate piece of work.
_DEFAULT_PARAMETERS = ["2t", "2d", "10ff"]
_DEFAULT_DOMAINS = ["n.hem", "tropics", "s.hem", "europe"]

# The retrieved GRIB is reusable across expvers only for the baselines, but the
# experiment's own fields are worth keeping between reruns of the same window.
_DEFAULT_CACHE_ROOT = "~/scratch/eval/_obs_crps_cache"


def _expand_dates(eff: dict, params: dict) -> list[str]:
    """The explicit list of forecast start dates in the window.

    Prefer the dates the run actually produced; fall back to walking the window
    with the resolved stride so a run that only records first/last still works.
    """
    predict = (eff.get("resolved") or {}).get("predict") or {}
    dates = _sorted_ints(predict.get("dates") or [])
    if dates:
        return [str(d) for d in dates]

    first = datetime.strptime(str(params["first_reference_date"]), "%Y%m%d")
    last = datetime.strptime(str(params["last_reference_date"]), "%Y%m%d")
    stride = max(24, int(params["date_step"]))
    out, cur = [], first
    while cur <= last:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(hours=stride)
    return out


def _expand_steps(eff: dict, eval_config: dict, params: dict) -> list[int]:
    predict = (eff.get("resolved") or {}).get("predict") or {}
    steps = [s for s in _sorted_ints(predict.get("steps") or []) if s > 0]
    first, last, stride = _resolve_lead_range(eff, eval_config, steps or [24])
    return list(range(int(first), int(last) + 1, int(stride)))


def _cache_dir(eval_config: dict, params: dict, dates: list[str], steps: list[int]) -> Path:
    root = Path(eval_config.get("cache_root", _DEFAULT_CACHE_ROOT)).expanduser()
    tag = (
        f"{params['expver']}_{params['grid']}_"
        f"{dates[0]}-{dates[-1]}_lt{steps[0]}-{steps[-1]}s{steps[1] - steps[0] if len(steps) > 1 else 24}"
        f"_m{params['nmem']}"
    )
    return root / tag


def _run_compute(args: list[str], cwd: Path) -> None:
    inner = "python3 " + " ".join(shlex.quote(a) for a in [str(_COMPUTE), *args])
    LOG.info("obs_crps compute: %s (cwd=%s)", inner, cwd)
    subprocess.run(
        ["bash", "-lc", f"module load python3 vtb ecmwf-toolbox && {inner}"],
        check=True,
        cwd=str(cwd),
    )


def run(
    predictions_dir,
    lane_config,
    eval_config,
    *,
    output_dir=None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    results_dir = (
        Path(output_dir) if output_dir
        else Path(predictions_dir) / "evaluators" / "obs_crps"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    eff = _load_effective_config(results_dir)
    eval_config = eval_config or {}
    params = resolve_params(eff, eval_config)
    if params is None:
        reason = (
            "obs_crps needs a run that published an ensemble to FDB under an expver "
            "(mode==prepml, expver set, and predict dates/members/steps). This run is "
            "not one of those, so there is nothing to score against the observations."
        )
        LOG.warning("obs_crps: %s", reason)
        (results_dir / "skipped.json").write_text(json.dumps({"reason": reason}, indent=2))
        return results_dir

    done = results_dir / "compute.done"
    if done.exists() and not overwrite:
        LOG.info("obs_crps: compute.done present, skipping recompute (%s)", done)
        return results_dir

    dates = _expand_dates(eff, params)
    steps = _expand_steps(eff, eval_config, params)
    parameters = list(eval_config.get("parameters") or _DEFAULT_PARAMETERS)
    domains = list(eval_config.get("domains") or _DEFAULT_DOMAINS)
    cache_dir = _cache_dir(eval_config, params, dates, steps)
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        **params,
        "dates": dates,
        "steps": steps,
        "parameters": parameters,
        "domains": domains,
        "cache_dir": str(cache_dir),
    }
    (results_dir / "params.json").write_text(json.dumps(resolved, indent=2))

    args = [
        "--expver", str(params["expver"]),
        "--class", str(params["class_"]),
        "--stream", str(params.get("stream", "enfo")),
        "--database", str(params["database"]),
        "--grid", str(params["grid"]),
        "--nmem", str(params["nmem"]),
        "--dates", ",".join(dates),
        "--steps", ",".join(str(s) for s in steps),
        "--parameters", ",".join(parameters),
        "--domains", ",".join(domains),
        "--cache-dir", str(cache_dir),
        "--out-dir", str(results_dir),
    ]
    if eval_config.get("gross_error_check") is False:
        args.append("--no-gross-error-check")

    _run_compute(args, cwd=results_dir)
    LOG.info(
        "obs_crps: scored %s over %d dates x %d lead times -> %s",
        params["expver"], len(dates), len(steps), results_dir,
    )
    return results_dir
