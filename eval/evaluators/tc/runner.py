"""TC evaluator — data loading and per-event statistics.

Delegates to eval.tc.workflows for the heavy lifting, but sources
configuration from lane_config instead of hardcoded values, and uses
eval.discovery.predictions for file finding.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from eval.config.loader import load_event
from eval.discovery.predictions import find_predictions
from eval.tc.data_types import BoundingBox
from eval.tc.events import EVENTS, TCEvent
from eval.tc.experiment_config import TCExperimentConfig
from eval.tc.loading_predictions import (
    event_days_steps,
    forecast_dates_for_event,
    select_prediction_files_for_event,
)
from eval.tc.plot_config import PLOT_CONFIGS, TCPlotConfig
from eval.tc.workflows import (
    compute_event_stats,
    load_curves_for_event,
)

LOG = logging.getLogger(__name__)


def _event_from_config(event_name: str) -> TCEvent:
    """Resolve a TCEvent, checking both the YAML config and the legacy registry."""
    if event_name in EVENTS:
        return EVENTS[event_name]
    cfg = load_event(event_name)
    return TCEvent(
        name=cfg["name"],
        year=str(cfg["dates"][0])[:4],
        month=str(cfg["dates"][0])[4:6],
        dates=[str(d)[6:8] for d in cfg["dates"]],
        analysis_dates=cfg.get("analysis_dates", [str(cfg["dates"][0])]),
        bbox=BoundingBox(
            north=cfg["lat_max"],
            south=cfg["lat_min"],
            east=cfg["lon_max"],
            west=cfg["lon_min"],
        ),
    )


def _pred_files_as_tuples(predictions_dir: Path):
    """Convert discovery PredictionFile objects to the (path, ymd, step) tuples
    that the existing TC loading code expects."""
    preds = find_predictions(predictions_dir)
    return [(p.path, int(p.date), p.step) for p in preds]


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    run_label: str = "",
    grib_dir: str | None = None,
    analysis_expid: str | None = None,
    support_mode: str = "native",
    **kwargs,
) -> Path:
    """Run TC evaluation for all configured events.

    Writes per-event stats JSON to output_dir/stats.json.
    Returns the output directory path.
    """
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "tc"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"TC output directory already has content: {output_dir}. Pass overwrite=True to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)

    event_names = eval_config.get("events", [])
    if not event_names:
        LOG.warning("No TC events configured in lane_config['tc']['events']")
        return output_dir

    pred_files = _pred_files_as_tuples(predictions_dir)
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    payload = {"events": {}}

    for event_name in event_names:
        event = _event_from_config(event_name)
        event_pred_files = select_prediction_files_for_event(pred_files, event)
        if not event_pred_files:
            LOG.info("Skipping event=%s: no matching prediction files", event_name)
            continue

        exp_cfg = None
        if analysis_expid:
            exp_cfg = TCExperimentConfig(
                analysis_expid=analysis_expid,
                base_tc_dir=grib_dir,
            )

        curves = load_curves_for_event(
            event,
            prediction_dir=predictions_dir,
            grib_dir=grib_dir,
            experiment_config=exp_cfg,
            support_mode=support_mode,
            run_label=run_label,
            pred_files=event_pred_files,
        )

        if not curves:
            LOG.warning("No curves loaded for event=%s, skipping", event_name)
            continue

        # Determine analysis key
        oper_key = analysis_expid
        if oper_key and oper_key not in curves:
            LOG.warning("Analysis key %s not in curves for event=%s", oper_key, event_name)
            # Fall back: if only prediction curves, skip stats that need analysis
            oper_key = None

        if oper_key:
            plot_cfg = PLOT_CONFIGS.get(event_name, TCPlotConfig())
            event_stats = compute_event_stats(
                curves,
                analysis_key=oper_key,
                plot_config=plot_cfg,
            )
        else:
            # Prediction-only mode: store raw curve metadata without analysis-anchored stats
            days, steps = event_days_steps(event_pred_files)
            event_stats = {
                "event": event_name,
                "selected_days": days,
                "steps_hours": steps,
                "prediction_only": True,
            }

        event_stats["event"] = event_name
        payload["events"][event_name] = event_stats

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)

    LOG.info("TC stats written to %s", stats_path)
    return output_dir


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
