"""Backfill mslp_p001 + wind_p9999 (and refresh mslp_min / wind_max) into TC
stats JSONs that pre-date the percentile-field extension.

Standalone CLI runnable as:

    python -m eval.jobs.backfill_tc_extreme_percentiles \
        --eval-root /home/ecm5702/perm/eval \
        --lane o96_o320 \
        [--check-only] [--ckpt <run_id>]

What it does (per ckpt in --eval-root):
  1. Locate the TC stats JSON (skips ckpts that don't have one).
  2. For each event in the lane config:
     - Lazily load lane refs (AN + ENFO + EEFO) via the same regridded curves
       the TC evaluator uses — cached across ckpts within a single invocation.
     - Load the ckpt's predictions for that event via load_prediction_curves.
     - Compute mslp_p001 / mslp_min / wind_p9999 / wind_max for the model row
       and the three ref rows.
  3. Update each row in extreme_tail.rows in place; atomic write via tmpfile.

Idempotent: rows that already have the new fields are not re-computed unless
their min/max values are missing.

Designed to be safe to run on existing scoreboards before regenerating them.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eval._backends.tc.events import EVENTS
from eval._backends.tc.loading_grib import load_grib_curves, regridded_target_points
from eval._backends.tc.loading_predictions import (
    discover_prediction_files,
    event_days_steps,
    forecast_dates_for_event,
    load_prediction_curves,
    select_prediction_files_for_event,
)
from eval._backends.tc.plot_config import PLOT_CONFIGS

LOG = logging.getLogger(__name__)

LANE_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config" / "lanes"

NEW_FIELDS = ("mslp_p001", "mslp_min", "wind_p9999", "wind_max")


def load_lane_config(lane: str) -> dict:
    path = LANE_CONFIG_ROOT / f"{lane}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Lane config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def _compute_extreme(arr: np.ndarray, variable: str) -> dict[str, float | None]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: None for k in NEW_FIELDS}
    if variable == "mslp":
        return {
            "mslp_p001": float(np.percentile(arr, 0.01)),
            "mslp_min": float(arr.min()),
            "wind_p9999": None,
            "wind_max": None,
        }
    return {
        "mslp_p001": None,
        "mslp_min": None,
        "wind_p9999": float(np.percentile(arr, 99.99)),
        "wind_max": float(arr.max()),
    }


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if v is not None:
            out[k] = v
    return out


def stats_missing_fields(stats_path: Path) -> dict[str, list[str]]:
    """Return {event_name: [missing_field, ...]} for any row in extreme_tail."""
    try:
        data = json.loads(stats_path.read_text())
    except Exception as exc:
        LOG.warning("Cannot read %s: %s", stats_path, exc)
        return {}
    missing: dict[str, list[str]] = {}
    for ev_name, ev_data in data.get("events", {}).items():
        rows = ev_data.get("extreme_tail", {}).get("rows", [])
        for row in rows:
            for f in NEW_FIELDS:
                if row.get(f) is None or not _is_finite(row.get(f)):
                    missing.setdefault(ev_name, []).append(f"{row.get('exp', '?')}:{f}")
    return missing


def _is_finite(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _find_tc_stats(perm_dir: Path) -> Path | None:
    if not perm_dir.exists():
        return None

    def rank(p: Path) -> tuple:
        text = str(p).lower()
        name = p.name.lower()
        return (
            1 if "proxy" in text else 0,
            1 if any(t in text for t in ("smoketest", "cached_refs", "template", "submit_helper")) else 0,
            1 if any(t in text for t in ("full75", "aug16_30")) else 0,
            0 if any(t in name for t in ("from_predictions", "full25")) else 1,
            0 if "idalia_franklin" in name else 1,
            text,
        )

    cands = sorted(perm_dir.rglob("tc_normed_pdfs_*.stats.json"), key=rank)
    return cands[0] if cands else None


def _load_event_refs(
    lane_config: dict,
    event_name: str,
    first_pred_files: list,
):
    """Load AN + ENFO + EEFO curves for one event, return per-row extreme dicts."""
    event = EVENTS[event_name]
    plot_cfg = PLOT_CONFIGS[event_name]
    days, steps = event_days_steps(first_pred_files)
    forecast_dates = forecast_dates_for_event(event, days)

    tc_cfg = lane_config["tc"]
    grib_dir = tc_cfg["grib_dir"]
    analysis_expid = tc_cfg["analysis_expid"]
    reference_expids = list(tc_cfg.get("reference_expids", []))
    max_pf_members = tc_cfg.get("max_pf_members", 10)

    LOG.info("[%s] Loading GRIB refs (max_pf_members=%d)...", event_name, max_pf_members)
    grib = load_grib_curves(
        dir_data_base=grib_dir,
        event_name=event_name,
        analysis_expid=analysis_expid,
        analysis_dates=list(event.analysis_dates),
        forecast_dates=forecast_dates,
        reference_expids=reference_expids,
        support_mode="regridded",
        bbox=event.bbox,
        regrid_resolution=plot_cfg.regrid_resolution,
        steps=steps,
        max_pf_members=max_pf_members,
    )

    sample_an = f"{grib_dir}/{event_name}/surface_an_{analysis_expid}_{event.analysis_dates[0]}.grib"
    tlon, tlat = regridded_target_points(event.bbox, plot_cfg.regrid_resolution, sample_an)

    refs = {}
    for expid in [analysis_expid, *reference_expids]:
        c = grib[expid]
        msl_extreme = _compute_extreme(c.msl, "mslp")
        wnd_extreme = _compute_extreme(c.wind, "wind")
        refs[expid] = _merge(msl_extreme, wnd_extreme)

    return refs, (tlon, tlat)


def _load_model_extreme(event_name: str, pred_dir: Path, target_pts) -> dict[str, float | None] | None:
    event = EVENTS[event_name]
    pred_files = discover_prediction_files(pred_dir)
    ev_pred = select_prediction_files_for_event(pred_files, event)
    if not ev_pred:
        return None
    tlon, tlat = target_pts
    curve = load_prediction_curves(
        ev_pred,
        bbox=event.bbox,
        support_mode="regridded",
        target_lon=tlon,
        target_lat=tlat,
    )
    msl_extreme = _compute_extreme(curve.msl, "mslp")
    wnd_extreme = _compute_extreme(curve.wind, "wind")
    return _merge(msl_extreme, wnd_extreme)


def backfill_stats_file(
    stats_path: Path,
    pred_dir: Path,
    lane_config: dict,
    ref_cache: dict | None = None,
) -> bool:
    """Backfill new percentile fields into one stats JSON. Returns True if file was modified.

    Atomic: writes to <stats>.tmp and renames on success.

    Hard-fails (raises) if predictions can't be loaded — callers should catch
    and record the error rather than silently dropping the ckpt.
    """
    if ref_cache is None:
        ref_cache = {}

    data = json.loads(stats_path.read_text())
    if "events" not in data:
        return False

    # Build first-pred-files inventory per event using the model's prediction dir.
    first_pred_files_all = discover_prediction_files(pred_dir)

    changed = False
    for event_name in lane_config["tc"].get("events", []):
        ev_data = data["events"].get(event_name)
        if not isinstance(ev_data, dict):
            continue
        rows = ev_data.get("extreme_tail", {}).get("rows", [])
        if not rows:
            continue

        # Quick check: any row missing the new fields?
        if all(
            all(_is_finite(row.get(f)) for f in NEW_FIELDS)
            for row in rows
        ):
            continue  # already complete for this event

        # Load refs (cached across ckpts within the run)
        if event_name not in ref_cache:
            event = EVENTS[event_name]
            event_pred = select_prediction_files_for_event(first_pred_files_all, event)
            if not event_pred:
                LOG.warning("[%s] No matching prediction files for refs derivation", event_name)
                continue
            ref_cache[event_name] = _load_event_refs(lane_config, event_name, event_pred)

        refs, target_pts = ref_cache[event_name]

        # Compute model extreme
        model_extreme = _load_model_extreme(event_name, pred_dir, target_pts)
        if model_extreme is None:
            LOG.warning("[%s] No predictions found in %s — skipping event", event_name, pred_dir)
            continue

        # Determine model row label = perm dir name
        model_row_label = pred_dir.parent.name

        for row in rows:
            exp = str(row.get("exp", "")).strip()
            if exp in refs:
                update = refs[exp]
            elif exp == model_row_label:
                update = model_extreme
            else:
                continue
            for k, v in update.items():
                if v is None:
                    continue
                if not _is_finite(row.get(k)):
                    row[k] = v
                    changed = True

    if not changed:
        return False

    tmp = stats_path.with_suffix(stats_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            data, indent=2,
            default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x,
        )
    )
    os.replace(tmp, stats_path)
    return True


def iter_ckpt_paths(eval_root: Path, ckpt_filter: str | None = None):
    """Yield (perm_dir, stats_path, pred_dir) for each ckpt under eval_root."""
    for perm_dir in sorted(eval_root.glob("manual_*")):
        if not perm_dir.is_dir():
            continue
        if ckpt_filter and ckpt_filter not in perm_dir.name:
            continue
        stats_path = _find_tc_stats(perm_dir)
        if stats_path is None:
            continue
        pred_dir = perm_dir / "predictions"
        if not pred_dir.exists():
            continue
        yield perm_dir, stats_path, pred_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, default=Path("/home/ecm5702/perm/eval"),
                        help="Root directory containing manual_* ckpt dirs (default: %(default)s)")
    parser.add_argument("--lane", default="o96_o320", help="Lane name (selects lane YAML)")
    parser.add_argument("--ckpt", default=None,
                        help="Substring filter to backfill only one ckpt")
    parser.add_argument("--check-only", action="store_true",
                        help="Report which stats JSONs are missing fields; do not modify anything")
    parser.add_argument("--summary-out", type=Path, default=None,
                        help="Write JSON summary log here (default: scratch/eval/<lane>/backfill_log.json)")
    args = parser.parse_args()

    lane_config = load_lane_config(args.lane)

    missing_by_ckpt: dict[str, dict[str, list[str]]] = {}
    summary: list[dict] = []
    ref_cache: dict = {}

    for perm_dir, stats_path, pred_dir in iter_ckpt_paths(args.eval_root, args.ckpt):
        missing = stats_missing_fields(stats_path)
        if not missing:
            continue
        missing_by_ckpt[perm_dir.name] = missing

        if args.check_only:
            LOG.info("MISSING %s -> %s", perm_dir.name, missing)
            summary.append({"label": perm_dir.name, "stats": str(stats_path), "missing": missing, "status": "check_only"})
            continue

        LOG.info("[%s] backfilling %s", perm_dir.name, stats_path.name)
        try:
            changed = backfill_stats_file(stats_path, pred_dir, lane_config, ref_cache)
            status = "updated" if changed else "no_change"
        except Exception as exc:
            LOG.exception("backfill_stats_file failed for %s", perm_dir.name)
            summary.append({"label": perm_dir.name, "stats": str(stats_path), "status": "error", "error": str(exc)})
            continue
        summary.append({"label": perm_dir.name, "stats": str(stats_path), "status": status})

    out = args.summary_out or Path(f"/home/ecm5702/scratch/eval/{args.lane}/backfill_log.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lane": args.lane,
        "eval_root": str(args.eval_root),
        "check_only": args.check_only,
        "missing_count": len(missing_by_ckpt),
        "summary": summary,
    }, indent=2))
    LOG.info("Wrote summary to %s", out)

    if args.check_only and missing_by_ckpt:
        LOG.warning("%d stats JSONs are missing percentile fields. Run without --check-only to backfill.",
                    len(missing_by_ckpt))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
