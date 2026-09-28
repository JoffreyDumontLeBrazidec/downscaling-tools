"""Observation CRPS evaluator — run phase.

Scores an ensemble against surface station observations and reports the result as
a function of lead time, averaged over the forecast start dates in the window.
This is the same measurement quaver makes for surface parameters, and it was
calibrated against quaver before being wired in: over 1440 points it used exactly
the same stations quaver did at every one, and agreed on the score to within
about a tenth of a per cent on the lead-time curve.

Why it exists next to the quaver evaluator: quaver is a multi-hour, FDB-bound job
that returns a PDF, so it cannot be run inside a normal iteration loop and its
numbers cannot reach a scoreboard. This evaluator returns numbers in minutes. It
does not replace quaver, which remains the canonical scorecard and covers upper
air as well.

Three curves, always. The standing rule for a scorecard is that the experiment is
shown against the coarse input that fed the downscaler and against the
high-resolution reference, so that a reader can see what the downscaling added
rather than only what it scored. The three are resolved with the quaver
evaluator's own helpers, so the two evaluators score the same window and the same
baselines by construction rather than by coincidence.

The two baselines reach the stations by different routes, for a reason worth
knowing. STVL, the observation database, also keeps the operational forecasts
already interpolated to the observing stations, and it stores them at the nearest
grid point, which is exactly what quaver's interpolation does; that was verified
directly on 2026-09-04. So the reference curve costs nothing. STVL does not hold
the extended-range stream that feeds the o320 to o1280 lane, so the input curve
has to be retrieved from the archive and interpolated here, which is slow enough
to be worth caching across runs that share a window.

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

import pandas as pd

from eval.evaluators.quaver.runner import (
    _load_effective_config,
    _resolve_lead_range,
    _sorted_ints,
    resolve_input_params,
    resolve_params,
    resolve_reference_params,
)

LOG = logging.getLogger(__name__)

_BACKEND = Path(__file__).resolve().parent.parent.parent / "_backends" / "obs_crps"
_COMPUTE = _BACKEND / "obs_crps_compute.py"

# Surface parameters STVL holds and quaver scores. Precipitation is deliberately
# absent: it needs the SEEPS climatology and an accumulation period, which is a
# separate piece of work.
_DEFAULT_PARAMETERS = ["2t", "2d", "10ff"]
_DEFAULT_DOMAINS = ["n.hem", "tropics", "s.hem", "europe"]

# Streams STVL keeps at the station points, checked on 2026-09-04: `enfo` returns
# all 50 members and `oper` the control, while `eefo` returns nothing at all. A
# curve on any other stream has to be retrieved and interpolated here instead.
_STVL_STREAMS = {"enfo", "oper"}

_DEFAULT_CACHE_ROOT = "~/scratch/eval/_obs_crps_cache"


def _expand_dates(eff: dict, params: dict) -> list[str]:
    """The explicit list of forecast start dates in the window.

    Prefer the dates the run actually produced; fall back to walking the window
    with the resolved stride so a run that only records first and last still works.
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


def _window_tag(params: dict, dates: list[str], steps: list[int]) -> str:
    stride = steps[1] - steps[0] if len(steps) > 1 else 24
    return (
        f"{params['expver']}_{params.get('stream', 'enfo')}_{params['grid']}_"
        f"{dates[0]}-{dates[-1]}_lt{steps[0]}-{steps[-1]}s{stride}_m{params['nmem']}"
    )


def _cache_root(eval_config: dict) -> Path:
    return Path(eval_config.get("cache_root", _DEFAULT_CACHE_ROOT)).expanduser()


def _run_compute(args: list[str], cwd: Path) -> None:
    inner = "python3 " + " ".join(shlex.quote(a) for a in [str(_COMPUTE), *args])
    LOG.info("obs_crps compute: %s (cwd=%s)", inner, cwd)
    # vtb starts metview behind the scenes and the default eight second start-up
    # timeout trips intermittently on a busy login node; the quaver wrapper
    # raises it for the same reason.
    preamble = (
        "module load python3 vtb ecmwf-toolbox && "
        "export METVIEW_PYTHON_START_TIMEOUT=${METVIEW_PYTHON_START_TIMEOUT:-60} && "
    )
    subprocess.run(["bash", "-lc", preamble + inner], check=True, cwd=str(cwd))


def _compute_curve(
    curve: str,
    params: dict,
    dates: list[str],
    steps: list[int],
    parameters: list[str],
    domains: list[str],
    out_dir: Path,
    cache_dir: Path,
    gross_error_check: bool,
    cwd: Path,
) -> Path | None:
    """Score one curve, reusing an earlier identical computation if there is one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "compute.done").exists():
        LOG.info("obs_crps: %s curve already computed, reusing %s", curve, out_dir)
        return out_dir

    stream = str(params.get("stream", "enfo"))
    source = "stvl" if (curve != "experiment" and stream in _STVL_STREAMS) else "grib"
    LOG.info("obs_crps: %s curve, stream %s, source %s", curve, stream, source)

    args = [
        "--expver", str(params["expver"]),
        "--class", str(params["class_"]),
        "--stream", stream,
        "--database", str(params["database"]),
        "--grid", str(params["grid"]),
        "--nmem", str(params["nmem"]),
        "--dates", ",".join(dates),
        "--steps", ",".join(str(s) for s in steps),
        "--parameters", ",".join(parameters),
        "--domains", ",".join(domains),
        "--cache-dir", str(cache_dir),
        "--out-dir", str(out_dir),
        "--source", source,
        "--curve", curve,
    ]
    if not gross_error_check:
        args.append("--no-gross-error-check")
    _run_compute(args, cwd=cwd)
    return out_dir


def _combine(curve_dirs: list[Path], results_dir: Path) -> None:
    """Gather the curves into the two CSVs the scorer and plotter read."""
    for name in ("scores_by_date_and_lead.csv", "summary_by_lead.csv"):
        frames = [pd.read_csv(d / name) for d in curve_dirs if (d / name).exists()]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(results_dir / name, index=False)


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
    gross_error_check = eval_config.get("gross_error_check") is not False
    cache_root = _cache_root(eval_config)

    wanted = {"experiment": params}
    if eval_config.get("three_curve") is not False:
        input_params = resolve_input_params(eff, params, eval_config)
        if input_params is not None:
            wanted["input"] = input_params
        else:
            LOG.warning("obs_crps: input baseline not derivable, curve omitted")
        wanted["reference"] = resolve_reference_params(eff, params, eval_config)

    (results_dir / "params.json").write_text(json.dumps({
        "dates": dates, "steps": steps, "parameters": parameters,
        "domains": domains, "curves": {k: v for k, v in wanted.items()},
    }, indent=2, default=str))

    curve_dirs: list[Path] = []
    for curve, curve_params in wanted.items():
        # The experiment belongs to this run; the baselines are shared by every
        # run over the same window, so they are cached outside the run directory
        # and the archive retrieval is paid once.
        if curve == "experiment":
            out_dir = results_dir / "curves" / curve
        else:
            out_dir = cache_root / "curves" / _window_tag(curve_params, dates, steps)
        cache_dir = cache_root / "fields" / _window_tag(curve_params, dates, steps)
        try:
            produced = _compute_curve(
                curve, curve_params, dates, steps, parameters, domains,
                out_dir, cache_dir, gross_error_check, cwd=results_dir,
            )
        except Exception as exc:
            if curve == "experiment":
                raise
            # A missing baseline should never cost us the experiment's numbers.
            LOG.warning("obs_crps: %s curve failed (%s), continuing without it", curve, exc)
            continue
        if produced is not None:
            curve_dirs.append(produced)

    _combine(curve_dirs, results_dir)
    done.write_text(json.dumps({"curves": [d.name for d in curve_dirs]}, indent=2))
    LOG.info(
        "obs_crps: %s over %d dates x %d lead times, curves %s -> %s",
        params["expver"], len(dates), len(steps),
        [d.name for d in curve_dirs], results_dir,
    )
    return results_dir
