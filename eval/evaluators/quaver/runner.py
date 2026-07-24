"""Quaver evaluator — run phase.

Resolves the FDB scoring window (expver, dates, members, lead times, grid) from
the run's effective_config.json and invokes the quaver probabilistic compute
backend under the `quaver` module binary. Heavy + FDB-bound: keep on the FDB
host (atos_ac) and budget time accordingly. No-op for manual runs.

Beyond the experiment (ML) ensemble, this also computes the INPUT baseline so
the scorecards can show the input's CRPS *and spread* alongside the downscaled
output (the "3-curve" comparison: input -> ML -> reference). The input is the
coarse operational ENS that fed the downscaler; its source (class/stream/grid)
is read lane-agnostically from resolved.prepml.input, scored against the MARS
archive (--database off). Because the input scores are global (keyed by
date/step/grid/vstream in the quaver offline DB), they are computed once per
(window, input-grid) and cached in a shared location, then reused across every
expver/run that shares the window -> no ~47 min MARS recompute per eval.
"""
from __future__ import annotations

import json
from datetime import datetime
import logging
import re
import shlex
import subprocess
from pathlib import Path

LOG = logging.getLogger(__name__)

_BACKEND = Path(__file__).resolve().parent.parent.parent / "_backends" / "quaver"
_Q_COMPUTE = _BACKEND / "q_compute_probabilistic.py"
_PL_GRID = "1.5/1.5"  # upper-air scores are always computed on the regridded 1.5deg grid


def _load_effective_config(results_dir: Path) -> dict:
    # results_dir == <run_root>/evaluators/quaver ; effective_config.json sits at <run_root>.
    for cand in (
        results_dir.parent.parent / "effective_config.json",
        results_dir.parent.parent / "data" / "effective_config.json",
    ):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except Exception:
                LOG.warning("quaver: could not parse %s", cand)
    return {}


def _sorted_ints(values) -> list[int]:
    return sorted({int(v) for v in values})


def _infer_step(vals: list[int], default: int = 24) -> int:
    return (vals[1] - vals[0]) if len(vals) >= 2 else default


def _date_step_hours(dates: list[int], default: int = 24) -> int:
    # dates are YYYYMMDD ints; quaver wants the reference-date stride in HOURS.
    if len(dates) >= 2:
        d0 = datetime.strptime(str(dates[0]), "%Y%m%d")
        d1 = datetime.strptime(str(dates[1]), "%Y%m%d")
        return max(24, (d1 - d0).days * 24)
    return default


def _derive_grid(lane_name: str) -> str:
    # Lane names follow o<in>_o<out>_...; output grid is the SECOND o<N> token.
    nums = re.findall(r"o(\d+)", lane_name or "")
    return f"O{nums[1]}" if len(nums) >= 2 else "O320"


def _derive_input_grid(lane_name: str) -> str:
    # input grid is the FIRST o<N> token of the lane name.
    nums = re.findall(r"o(\d+)", lane_name or "")
    return f"O{nums[0]}" if nums else "O320"


def resolve_params(eff: dict, eval_config: dict) -> dict | None:
    """Return quaver compute params for the experiment, or None if not a prepml/FDB run."""
    if not eff.get("expver"):
        return None
    predict = (eff.get("resolved") or {}).get("predict") or {}
    dates = _sorted_ints(predict.get("dates") or [])
    members = [int(m) for m in (predict.get("members") or [])]
    steps = [s for s in _sorted_ints(predict.get("steps") or []) if s > 0]
    if not (dates and members and steps):
        return None
    return {
        "expver": str(eff["expver"]),
        "nmem": max(members),
        "first_reference_date": dates[0],
        "last_reference_date": dates[-1],
        "date_step": _date_step_hours(dates),
        "first_lead_time": steps[0],
        "last_lead_time": steps[-1],
        "lead_time_step": _infer_step(steps),
        "grid": str(eval_config.get("grid") or _derive_grid(eff.get("lane", ""))),
        "class_": str(eval_config.get("class", eval_config.get("class_", "rd"))),
        "database": str(eval_config.get("database", "fdb")),
    }


def resolve_input_params(eff: dict, exp_params: dict, eval_config: dict) -> dict | None:
    """Return quaver compute params for the INPUT baseline, or None if not derivable.

    The input is the coarse operational ENS that fed the downscaler. Its source
    is recorded lane-agnostically under resolved.prepml.input (class/stream/grid/
    type); it lives in the MARS archive, so it is scored with --database off and
    stored to vstream prepml_<expver>_{ob,an}. Window/members mirror the
    experiment. Surface scores land at the input grid; upper-air at 1.5/1.5.
    """
    cfg = dict(eval_config.get("input") or {})
    prepml_in = ((eff.get("resolved") or {}).get("prepml") or {}).get("input") or {}
    grid = cfg.get("grid") or prepml_in.get("grid") or _derive_input_grid(eff.get("lane", ""))
    if not grid:
        return None
    return {
        "expver": str(cfg.get("expver", "0001")),
        "nmem": exp_params["nmem"],
        "first_reference_date": exp_params["first_reference_date"],
        "last_reference_date": exp_params["last_reference_date"],
        "date_step": exp_params["date_step"],
        "first_lead_time": exp_params["first_lead_time"],
        "last_lead_time": exp_params["last_lead_time"],
        "lead_time_step": exp_params["lead_time_step"],
        "grid": str(grid),
        "class_": str(cfg.get("class", cfg.get("class_", prepml_in.get("class", "od")))),
        # MARS archive route; 'off' resolves to marsod. fdb/mars do not have old od data.
        "database": str(cfg.get("database", "off")),
        "stream": str(cfg.get("stream", prepml_in.get("stream", "enfo"))),
    }


def resolve_reference_params(eff: dict, exp_params: dict, eval_config: dict) -> dict:
    """Compute params for the SELF-COMPUTED high-res reference curve (enfo O1280).

    STANDING RULE (user, reaffirmed 2026-07-07): every scorecard shows THREE curves
    — input (eefo O320) -> ML -> reference (enfo O1280) — ALWAYS. The old code pointed
    the reference at the operational ``oper_{ob,an}`` *standing* dataset, which rotates
    out of the shared DB; when it was empty the fair-mean ``uniformiseMissingValues``
    wiped every curve -> empty plot ("quaver looks broken"). Fix: score the reference
    OURSELVES, exactly like the input baseline but at the model OUTPUT grid, so it is
    ALWAYS present. It is the same operational ENS (class=od, expver=0001, --database
    off) scored at O1280 -> vstream ``prepml_0001_ob`` @ O1280, distinct from the input
    (@ O320) by grid. Surface only: on upper-air both regrid to 1.5/1.5 so the self-
    computed reference coincides with the input; there the plotter keeps the standing
    oper reference (see plotter._patch_threecurve).
    """
    cfg = dict(eval_config.get("reference") or {})
    grid = str(cfg.get("grid") or exp_params["grid"])  # model OUTPUT grid (e.g. O1280)
    return {
        "expver": str(cfg.get("expver", "0001")),
        "nmem": exp_params["nmem"],
        "first_reference_date": exp_params["first_reference_date"],
        "last_reference_date": exp_params["last_reference_date"],
        "date_step": exp_params["date_step"],
        "first_lead_time": exp_params["first_lead_time"],
        "last_lead_time": exp_params["last_lead_time"],
        "lead_time_step": exp_params["lead_time_step"],
        "grid": grid,
        "class_": str(cfg.get("class", cfg.get("class_", "od"))),
        # MARS archive route; 'off' resolves to marsod (same as the input baseline).
        "database": str(cfg.get("database", "off")),
        "stream": str(cfg.get("stream", "enfo")),
        "pl_grid": str(cfg.get("pl_grid") or _PL_GRID),
        "label": str(cfg.get("label", f"enfo {grid}")),
    }


def _input_cache_dir(input_params: dict, eval_config: dict) -> Path:
    root = eval_config.get("input_cache_root") or (
        Path.home() / "perm" / "eval" / "_quaver_input_baseline_cache"
    )
    grid = str(input_params["grid"]).replace("/", "p")
    key = (
        f"{input_params['expver']}_{grid}"
        f"_{input_params['first_reference_date']}_{input_params['last_reference_date']}"
        f"_lt{input_params['first_lead_time']}-{input_params['last_lead_time']}"
        f"s{input_params['lead_time_step']}_n{input_params['nmem']}"
    )
    return Path(root) / key


def _compute_args(params: dict) -> list[str]:
    return [
        "--expver", str(params["expver"]),
        "--nmem", str(params["nmem"]),
        "--first_reference_date", str(params["first_reference_date"]),
        "--last_reference_date", str(params["last_reference_date"]),
        "--date_step", str(params["date_step"]),
        "--first_lead_time", str(params["first_lead_time"]),
        "--last_lead_time", str(params["last_lead_time"]),
        "--lead_time_step", str(params["lead_time_step"]),
        "--grid", str(params["grid"]),
        "--class", str(params["class_"]),
        "--database", str(params["database"]),
        # Ensemble stream: input=eefo (resolved.prepml.input.stream), reference/experiment=enfo.
        # Without this the compute hardcoded enfo -> the "eefo O320 input" curve was silently enfo.
        "--stream", str(params.get("stream", "enfo")),
    ]


def _run_quaver(script: Path, args: list[str], cwd: Path) -> None:
    inner = "quaver " + " ".join(shlex.quote(a) for a in [str(script), *args])
    LOG.info("quaver compute: %s (cwd=%s)", inner, cwd)
    subprocess.run(["bash", "-lc", f"module load quaver && {inner}"], check=True, cwd=str(cwd))


def _compute_input_baseline(input_params: dict, eval_config: dict, results_dir: Path, overwrite: bool) -> None:
    """Compute the input baseline once per (window, input-grid), cached & shared.

    The quaver scores are global (offline DB keyed by date/step/grid/vstream), so
    a single compute serves every expver that shares the window. We guard with a
    shared cache marker to avoid re-paying the ~47 min MARS-archive query.
    """
    cache_dir = _input_cache_dir(input_params, eval_config)
    marker = cache_dir / "compute.done"
    if marker.exists() and not overwrite:
        LOG.info("quaver: input baseline cache hit -> skipping recompute (%s)", marker)
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    _run_quaver(_Q_COMPUTE, _compute_args(input_params), cwd=results_dir)
    marker.write_text(json.dumps(input_params, indent=2))
    LOG.info("quaver: input baseline computed -> %s", marker)


def run(predictions_dir, lane_config, eval_config, *, output_dir=None, overwrite=False, **kwargs):
    results_dir = Path(output_dir) if output_dir else Path(predictions_dir) / "evaluators" / "quaver"
    results_dir.mkdir(parents=True, exist_ok=True)

    eff = _load_effective_config(results_dir)
    eval_config = eval_config or {}
    params = resolve_params(eff, eval_config)
    if params is None:
        reason = (
            "quaver needs a prepml run with an FDB expver (mode==prepml, expver set, "
            "and predict dates/members/steps). This run is not prepml -> skipping."
        )
        LOG.warning("quaver: %s", reason)
        (results_dir / "skipped.json").write_text(json.dumps({"reason": reason}, indent=2))
        return results_dir

    done = results_dir / "compute.done"
    if done.exists() and not overwrite:
        LOG.info("quaver: compute.done present -> skipping experiment recompute (%s)", done)
    else:
        _run_quaver(_Q_COMPUTE, _compute_args(params), cwd=results_dir)
        (results_dir / "params.json").write_text(json.dumps(params, indent=2))
        done.write_text(json.dumps(params, indent=2))

    # --- 3-curve comparison: input baseline (computed) + reference (standing) ---
    # Disable with quaver.three_curve: false to fall back to experiment-only scorecards.
    if eval_config.get("three_curve", True):
        input_params = resolve_input_params(eff, params, eval_config)
        if input_params is None:
            LOG.warning(
                "quaver: could not resolve input baseline (no resolved.prepml.input grid) "
                "-> experiment-only scorecards."
            )
        else:
            try:
                _compute_input_baseline(input_params, eval_config, results_dir, overwrite)
                (results_dir / "input_params.json").write_text(json.dumps(input_params, indent=2))
                # Reference (enfo O1280): SELF-COMPUTE it too — never the rotating oper_ob — so the
                # 3rd curve is ALWAYS present (standing rule: input+target+ML always). Same generic
                # baseline compute + shared cache, keyed by (expver,grid,window) -> the O1280 reference
                # is distinct from the O320 input, no clobber.
                ref_params = resolve_reference_params(eff, params, eval_config)
                _compute_input_baseline(ref_params, eval_config, results_dir, overwrite)
                (results_dir / "reference_params.json").write_text(json.dumps(ref_params, indent=2))
            except subprocess.CalledProcessError:
                # Baselines are best-effort; a MARS hiccup must not fail the eval. The
                # plotter falls back to experiment-only when input_params.json is absent.
                LOG.error("quaver: baseline compute failed -> experiment-only scorecards.", exc_info=True)
    return results_dir
