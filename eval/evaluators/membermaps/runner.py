"""Member maps evaluator: the four-panel case-inspection maps, driven by a lane.

The `eval.cli membermaps` subcommand renders these panels but has to be told
everything: which prediction directories, which date, step, member, extent and
projection. That is the right shape for comparing two ARMS of a campaign against
each other. It is the wrong shape for the ordinary question "show me what this
run looks like", which is what this evaluator answers.

Given one predictions directory it renders, for every region in the lane's own
configuration and every variable asked for, the driving O320 input, the embedded
same-index ENFO member as truth, and this run's prediction. It does that twice:
once as the field itself and once as `--field fine`, the high-pass view that
shows only the detail the O320 driver could not carry.

Diagnostic only. Nothing here scores anything, so there is no `score` function
and nothing reaches a scoreboard.
"""
from __future__ import annotations

import logging
from pathlib import Path

from eval.discovery.predictions import find_predictions

LOG = logging.getLogger(__name__)

DEFAULT_VARIABLES = ["wind10m", "msl", "2t"]
DEFAULT_FIELDS = ["value", "fine"]
# A ceiling on one invocation, so a careless config cannot quietly ask for
# thousands of panels. Each combination opens a multi-gigabyte prediction file.
MAX_COMBINATIONS = 120


def _regions(lane_config: dict, eval_config: dict) -> dict[str, list[float] | None]:
    """Region boxes as [lat_min, lat_max, lon_min, lon_max], the lane YAML order.

    Falls back to the `texture` evaluator's regions so the two diagnostics of the
    fine-scale epic always speak about the same boxes, and finally to a single
    entry with no box, which lets the backend use its own default extent.
    """
    boxes = eval_config.get("regions")
    if not boxes:
        boxes = dict(lane_config.get("texture", {}).get("regions", {}))
    return boxes or {"default": None}


def _extent_args(box: list[float] | None) -> list[str]:
    """Translate a lane region box into the backend's extent and projection flags."""
    if box is None:
        return []
    lat_min, lat_max, lon_min, lon_max = (float(v) for v in box)
    return [
        "--extent", str(lon_min), str(lon_max), str(lat_min), str(lat_max),
        "--proj-lon", str(0.5 * (lon_min + lon_max)),
        "--proj-lat", str(0.5 * (lat_min + lat_max)),
    ]


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    checkpoint: str | None = None,
    run_label: str = "",
    **kwargs,
) -> Path:
    from eval._backends.region_plotting.plot_member_wind_maps import (
        build_arg_parser, run as membermaps_run,
    )

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "membermaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = find_predictions(predictions_dir)
    if not preds:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    available_dates = sorted({p.date for p in preds})
    available_steps = sorted({p.step for p in preds})

    dates = [str(d) for d in eval_config.get("dates") or available_dates[:1]]
    # Default to one step in the middle of the range: the shortest lead is the
    # easiest case and the longest is the one where input and truth have drifted
    # furthest apart, so neither end is representative on its own.
    steps = [int(s) for s in eval_config.get("steps") or [available_steps[len(available_steps) // 2]]]
    members = [int(m) for m in eval_config.get("members") or [1]]
    variables = list(eval_config.get("variables") or DEFAULT_VARIABLES)
    fields = list(eval_config.get("fields") or DEFAULT_FIELDS)
    fine_cut_deg = float(eval_config.get("fine_cut_deg", 0.6))
    arm = str(eval_config.get("arm_label") or run_label or "model")
    regions = _regions(lane_config, eval_config)

    combos = len(dates) * len(steps) * len(members) * len(variables) * len(fields) * len(regions)
    if combos > MAX_COMBINATIONS:
        raise ValueError(
            f"membermaps was asked for {combos} renders "
            f"({len(dates)} dates x {len(steps)} steps x {len(members)} members x "
            f"{len(variables)} variables x {len(fields)} fields x {len(regions)} regions), "
            f"above the {MAX_COMBINATIONS} ceiling. Narrow the membermaps block in the lane "
            f"config, or raise the ceiling deliberately."
        )

    missing = [
        (d, s) for d in dates for s in steps
        if not (predictions_dir / f"predictions_{d}_step{s:03d}.nc").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"membermaps: no prediction file for {missing} in {predictions_dir}. "
            f"Available dates {available_dates}, steps {available_steps}."
        )

    written = 0
    for region_name, box in sorted(regions.items()):
        region_dir = output_dir / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        for date in dates:
            for step in steps:
                for member in members:
                    for variable in variables:
                        for field in fields:
                            argv = [
                                "--run", f"{arm}={predictions_dir}",
                                "--title", f"{arm}=" + (
                                    f"{arm} · O1280" if not checkpoint
                                    else f"{arm} ({Path(str(checkpoint)).name}) · O1280"
                                ),
                                "--variable", variable,
                                "--field", field,
                                "--fine-cut-deg", str(fine_cut_deg),
                                "--date", date,
                                "--step", str(step),
                                "--member", str(member),
                                "--region-tag", region_name,
                                "--output-dir", str(region_dir),
                            ] + _extent_args(box)
                            LOG.info("membermaps: %s", " ".join(argv))
                            membermaps_run(build_arg_parser().parse_args(argv))
                            written += 1

    LOG.info("membermaps: %d renders over %d region(s) into %s",
             written, len(regions), output_dir)
    return output_dir
