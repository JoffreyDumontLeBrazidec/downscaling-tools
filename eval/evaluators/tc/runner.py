"""TC evaluator — data loading and per-event statistics.

Delegates to eval.tc.workflows for the heavy lifting, but sources
configuration from lane_config instead of hardcoded values, and uses
eval.discovery.predictions for file finding.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from eval.config.loader import load_event
from eval.discovery.predictions import find_predictions
from eval._backends.tc.data_types import BoundingBox
from eval._backends.tc.events import EVENTS, TCEvent
from eval._backends.tc.experiment_config import TCExperimentConfig
from eval._backends.tc.loading_predictions import (
    event_days_steps,
    forecast_dates_for_event,
    load_prediction_curves,
    select_prediction_files_for_event,
)
from eval._backends.tc.loading_grib import regridded_target_points
from eval._backends.tc.plot_config import TCPlotConfig, resolve_plot_config
from eval.evaluators.tc.comparison_contract import (
    build_prediction_contract,
    validate_curve_support_contract,
    validate_comparison_contracts,
)
from eval._backends.tc.workflows import (
    _json_default,
    compute_event_stats,
    load_curves_for_event,
    run_member_maps,
)

LOG = logging.getLogger(__name__)


_GRIB_EXPID_RE = re.compile(r"([A-Za-z]+)_([Oo]\d+)_(\d{4})")


def _expid_from_grib_path(path):
    """Map a bundle source GRIB filename to its expid form.

    e.g. '.../eefo_o320_0001_date20230826_..._sfc.grib' -> 'EEFO_O320_0001'.
    Returns None when the name carries no <stream>_<grid>_<num> token.
    """
    if not path:
        return None
    m = _GRIB_EXPID_RE.search(os.path.basename(str(path)))
    return f"{m.group(1).upper()}_{m.group(2).upper()}_{m.group(3)}" if m else None


# The input curve is drawn by default. Without a default here the dedup below
# and the draw site further down disagreed: the dedup dropped a reference saying
# the input was "already drawn", while the draw site skipped it because the key
# was unset, and the curve disappeared with only a warning asserting otherwise.
_DEFAULT_INPUT_LABEL = "input"


def resolve_bundle_curve_labels(eval_config):
    """Labels for the bundle-derived input and target curves.

    One resolver for both the de-duplication and the drawing, so the two cannot
    drift apart again. The input gets a default because a comparison without the
    model's own input has no baseline for how much the downscaling adds. The
    target deliberately gets none: most lanes already draw their target from
    ``target_grib``, and defaulting this would add a duplicate target curve,
    which is the regression the de-duplication exists to prevent.
    """
    input_label = eval_config.get("input_label", _DEFAULT_INPUT_LABEL)
    target_nc_label = eval_config.get("target_nc_label")
    return input_label, target_nc_label


def _strip_bundle_duplicate_references(reference_expids, lane_config, eval_config):
    """Definitive guard against the duplicate-curve regression.

    The truth-aware bundle already carries the model INPUT (``x_interp``, drawn as
    ``input_label``) and TARGET (``y``, drawn as ``target_nc_label``), both built from the
    GRIBs named in ``prepare.args``. Listing those same products in ``reference_expids``
    re-draws identical data as a second curve on a *different* regrid support -- the exact
    redundancy fixed in 0f45ed7 and silently reverted in 8656a1a. Rather than trust the
    config to stay correct, strip any reference whose product == the bundle's input/target
    source. Non-blocking: warn and continue (correct-default philosophy; never gate the
    scientist).
    """
    if not reference_expids:
        return reference_expids
    prep_args = ((lane_config.get("prepare") or {}).get("args")) or {}
    bundle_products = {
        e for e in (
            _expid_from_grib_path(prep_args.get("lres_sfc_grib")),    # -> input curve
            _expid_from_grib_path(prep_args.get("target_sfc_grib")),  # -> target curve
        ) if e
    }
    if not bundle_products:
        return reference_expids
    input_label, target_nc_label = resolve_bundle_curve_labels(eval_config)
    # Only stand down in favour of a curve that will actually be drawn. The input
    # always is, now that it has a default. The target is drawn from the bundle
    # only when target_nc_label is set, and otherwise usually comes from
    # target_grib; if neither is true, keeping the reference is what preserves
    # the curve.
    target_is_drawn = bool(target_nc_label) or bool(eval_config.get("target_grib"))
    input_product = _expid_from_grib_path(prep_args.get("lres_sfc_grib"))
    target_product = _expid_from_grib_path(prep_args.get("target_sfc_grib"))
    replaceable = set()
    if input_product:
        replaceable.add(input_product)
    if target_product and target_is_drawn:
        replaceable.add(target_product)

    kept, dropped, retained_for_safety = [], [], []
    for expid in reference_expids:
        key = str(expid).upper()
        if key in replaceable:
            dropped.append(expid)
        else:
            kept.append(expid)
            if key in bundle_products:
                retained_for_safety.append(expid)
    if dropped:
        LOG.warning(
            "TC: dropping reference_expids %s -- identical to the bundle's input/target "
            "(drawn instead as '%s' / '%s'); keeping them would plot duplicate curves on a "
            "different regrid support. Non-blocking dedup (remove them from the lane "
            "`reference_expids` to silence).",
            dropped, input_label, target_nc_label or eval_config.get("target_label", "target"),
        )
    if retained_for_safety:
        LOG.warning(
            "TC: KEEPING reference_expids %s even though they duplicate the bundle, because "
            "no bundle curve would be drawn to replace them. Dropping them would remove the "
            "curve entirely rather than de-duplicate it.",
            retained_for_safety,
        )
    return tuple(kept)


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


def _bundle_curve_kwargs(
    *,
    mode: str,
    event: TCEvent,
    grib_dir: str | None,
    analysis_expid: str | None,
    regrid_resolution: float,
) -> dict:
    """Return support arguments for bundle input/target curves.

    In regridded TC mode the model and GRIB reference curves live on the regular
    analysis grid. ``x_interp`` and ``y`` are stored on the native O1280 graph
    grid, so loading them as native curves mixes supports and changes extremes.
    Use the same target grid as ``load_curves_for_event`` for all bundle curves.
    """
    if mode != "regridded":
        return {"support_mode": mode, "target_lon": None, "target_lat": None}
    if not grib_dir or not analysis_expid:
        raise ValueError(
            "Regridded bundle curves require grib_dir and analysis_expid "
            "to define the shared analysis grid"
        )
    sample_path = (
        Path(grib_dir)
        / event.name
        / f"surface_an_{analysis_expid}_{event.analysis_dates[0]}.grib"
    )
    target_lon, target_lat = regridded_target_points(
        event.bbox,
        regrid_resolution,
        str(sample_path),
    )
    return {
        "support_mode": mode,
        "target_lon": target_lon,
        "target_lat": target_lat,
    }


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
    extra_grib_references: dict | None = None,
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

    # Fall back to lane config for reference data not supplied by the caller.
    grib_dir = grib_dir or eval_config.get("grib_dir")
    analysis_expid = analysis_expid or eval_config.get("analysis_expid")
    reference_expids: tuple[str, ...] = tuple(eval_config.get("reference_expids") or ())
    reference_expids = _strip_bundle_duplicate_references(
        reference_expids, lane_config, eval_config
    )
    support_mode = eval_config.get("support_mode", support_mode)

    # run_label: explicit arg > eval_config > grandparent-of-predictions (handles data/ lean layout)
    if not run_label:
        run_label = eval_config.get("run_label", "")
    if not run_label:
        parent = predictions_dir.parent
        run_label = (parent.parent.name if parent.name == "data" else parent.name) or "prediction"
    # target_grib / target_label are the lane-config-level way to specify the high-res
    # truth target (e.g. IEKM for o1280_o2560). They are resolved into extra_grib_references
    # internally so callers never need to use the lower-level dict directly.
    target_grib = eval_config.get("target_grib")
    target_label = eval_config.get("target_label", "TARGET")
    if extra_grib_references is None and target_grib:
        extra_grib_references = {target_label: target_grib}

    event_names = eval_config.get("events", [])
    if not event_names:
        LOG.warning("No TC events configured in lane_config['tc']['events']")
        return output_dir

    # Cap pf-ensemble members for GRIB references so the ENFO/EEFO baselines are
    # sampled at the same ensemble size as ML predictions (typically 10). Without
    # this cap ENFO_O320 contributes 50 members, biasing tail percentiles deeper
    # than ML purely from sample-size effects. Default 10 mirrors run_tc_pdf.
    max_pf_members = eval_config.get("max_pf_members", 10)

    pred_files = _pred_files_as_tuples(predictions_dir)
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    # Fail LOUD preflight (run-trust contract, 2026-06-21): every configured event must
    # (a) resolve in EVENTS / load_event and (b) have at least one matched prediction
    # file. A misconfigured event (typo, missing YAML) or a zero-match selection silently
    # dropping an event from the scoreboard is exactly the non-reproducibility this
    # contract retires — so RAISE here instead of LOG.warning + skip.
    resolved_events: dict[str, TCEvent] = {}
    for event_name in event_names:
        try:
            event = _event_from_config(event_name)
        except Exception as exc:
            raise ValueError(
                f"TC event '{event_name}' does not resolve in EVENTS or as an event "
                f"config (load_event): {exc}. Fix lane_config['tc']['events']."
            ) from exc
        matched = select_prediction_files_for_event(pred_files, event)
        if not matched:
            raise FileNotFoundError(
                f"TC event '{event_name}' has zero matched prediction files under "
                f"{predictions_dir} (selected from {len(pred_files)} files). The event's "
                f"dates/bbox do not match any prediction — refusing to silently drop it."
            )
        resolved_events[event_name] = event

    payload = {"events": {}}

    # When support_mode is "both", run all events in regridded first, then native.
    # Grouping by mode keeps metview operations together and avoids MARS state issues.
    modes = ["regridded", "native"] if support_mode == "both" else [support_mode]

    # Each mode is isolated. With support_mode "both" a failure in one pass used
    # to escape this loop and abort the evaluator, throwing away the results the
    # other pass had already computed. Now a failing mode is recorded and the
    # others still produce output; the evaluator fails only if every mode fails.
    mode_failures = {}
    for mode in modes:
      try:
        for event_name in event_names:
            # Preflighted above: event resolves and has >=1 matched prediction file.
            event = resolved_events[event_name]
            event_pred_files = select_prediction_files_for_event(pred_files, event)
            if not event_pred_files:
                raise FileNotFoundError(
                    f"TC event '{event_name}' has zero matched prediction files "
                    f"(mode={mode}) — refusing to silently drop it."
                )

            event_key = event_name if mode == modes[0] else f"{event_name}__{mode}"

            exp_cfg = None
            if analysis_expid:
                exp_cfg = TCExperimentConfig(
                    analysis_expid=analysis_expid,
                    base_tc_dir=grib_dir,
                    reference_expids=reference_expids,
                )

            plot_cfg = resolve_plot_config(event_name, eval_config)
            prediction_contract = build_prediction_contract(
                prediction_files=[path for path, _ymd, _step in event_pred_files],
                bbox=event.bbox,
                support_mode=mode,
                regrid_resolution=plot_cfg.regrid_resolution,
                analysis_reference=analysis_expid or "",
            )
            reference_contract = dict(prediction_contract)
            reference_contract["ensemble_members"] = max_pf_members
            if analysis_expid:
                validate_comparison_contracts(
                    {"prediction": prediction_contract, "reference": reference_contract}
                )

            curves = None
            max_attempts = 2 if mode == "regridded" else 1
            for attempt in range(max_attempts):
                try:
                    curves = load_curves_for_event(
                        event,
                        prediction_dir=predictions_dir,
                        grib_dir=grib_dir,
                        experiment_config=exp_cfg,
                        extra_grib_references=extra_grib_references,
                        support_mode=mode,
                        run_label=run_label,
                        pred_files=event_pred_files,
                        max_pf_members=max_pf_members,
                        regrid_resolution=plot_cfg.regrid_resolution,
                    )
                    break
                except Exception:
                    if attempt < max_attempts - 1:
                        LOG.warning("Attempt %d failed for event=%s mode=%s, retrying", attempt + 1, event_name, mode)
                    else:
                        LOG.error("Failed to load curves for event=%s mode=%s", event_name, mode, exc_info=True)
                        raise
            if not curves:
                # Reference GRIBs / predictions produced no curves — fail loud rather
                # than silently dropping the event from the scoreboard (run-trust contract).
                raise RuntimeError(
                    f"TC event '{event_name}' (mode={mode}) loaded no curves — no "
                    f"reference GRIBs or matched predictions resolved. Refusing to skip."
                )

            bundle_curve_kwargs = _bundle_curve_kwargs(
                mode=mode,
                event=event,
                grib_dir=grib_dir,
                analysis_expid=analysis_expid,
                regrid_resolution=plot_cfg.regrid_resolution,
            )

            # Load input (x_interp) and target (y) curves directly from NetCDF files
            input_label, target_nc_label = resolve_bundle_curve_labels(eval_config)
            if input_label and event_pred_files:
                try:
                    curves[input_label] = load_prediction_curves(
                        event_pred_files,
                        bbox=event.bbox,
                        **bundle_curve_kwargs,
                        prediction_var="x_interp",
                    )
                except Exception:
                    LOG.warning("Failed to load x_interp curve for event=%s", event_name, exc_info=True)

            if target_nc_label and event_pred_files:
                try:
                    curves[target_nc_label] = load_prediction_curves(
                        event_pred_files,
                        bbox=event.bbox,
                        **bundle_curve_kwargs,
                        prediction_var="y",
                    )
                except Exception:
                    LOG.warning("Failed to load y curve for event=%s", event_name, exc_info=True)

            # Determine analysis key
            oper_key = analysis_expid
            if oper_key and oper_key not in curves:
                LOG.warning("Analysis key %s not in curves for event=%s mode=%s", oper_key, event_name, mode)
                oper_key = None

            # Rename analysis key if a display label is configured
            analysis_display_label = eval_config.get("analysis_display_label")
            if analysis_display_label and oper_key and oper_key in curves:
                curves[analysis_display_label] = curves.pop(oper_key)
                oper_key = analysis_display_label

            # Every curve contributing to one comparison must carry the same,
            # loader-derived support identity. This is deliberately before stats
            # are computed so a mixed-support plot cannot become a scoreboard row.
            validate_curve_support_contract(curves, mode)

            if oper_key:
                plot_cfg = resolve_plot_config(event_name, eval_config)
                event_stats = compute_event_stats(
                    curves,
                    analysis_key=oper_key,
                    plot_config=plot_cfg,
                )
            else:
                days, steps = event_days_steps(event_pred_files)
                event_stats = {
                    "event": event_name,
                    "selected_days": days,
                    "steps_hours": steps,
                    "prediction_only": True,
                }

            event_stats["event"] = event_name
            event_stats["support_mode"] = mode
            event_stats["support_contract"] = {
                "declared_mode": mode,
                "validated": True,
                "curve_supports": {
                    name: {
                        "mode": curve.support_mode,
                        "signature": curve.support_signature,
                    }
                    for name, curve in curves.items()
                },
            }
            event_stats["comparison_contract"] = prediction_contract
            event_stats["reference_comparison_contract"] = reference_contract
            # Provenance for cross-run trust (2026-06-22): the per-event tail is pooled over
            # ALL prediction files whose DD is in event.dates (a date RANGE), so the percentiles
            # drift with WHICH init dates are in predictions_dir. Record what actually contributed
            # so a consumer can gate comparisons on a matched (date set, support). Non-blocking
            # visibility only (no validator) per the project rule.
            _days, _steps = event_days_steps(event_pred_files)
            event_stats["selected_days"] = _days
            event_stats["n_dates"] = len(_days)
            event_stats["steps_hours"] = _steps
            event_stats["n_pred_files"] = len(event_pred_files)
            payload["events"][event_key] = event_stats

      except Exception as exc:
        if len(modes) == 1:
            raise
        mode_failures[mode] = repr(exc)
        LOG.error(
            "TC support mode %r failed and was skipped: %s. The other mode(s) still "
            "produced output; read the results only on the mode(s) that succeeded.",
            mode, exc, exc_info=True,
        )

    if not payload["events"]:
        raise RuntimeError(
            "TC produced no events: every support mode failed. "
            + "; ".join(f"{m}: {e}" for m, e in mode_failures.items())
        )
    if mode_failures:
        payload["mode_failures"] = mode_failures

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)

    LOG.info("TC stats written to %s", stats_path)

    # Member maps (optional, controlled by eval_config["member_maps"])
    mm_cfg = eval_config.get("member_maps") or {}
    if mm_cfg.get("enabled"):
        mm_steps = mm_cfg.get("steps") or [24, 120]
        mm_members = mm_cfg.get("members") or list(range(10))
        mm_outdir = output_dir / "member_maps"

        # Per-event dates (preferred) or global dates (legacy)
        event_dates = mm_cfg.get("event_dates") or {}
        if event_dates:
            for evt, date in event_dates.items():
                try:
                    run_member_maps(
                        predictions_dir=str(predictions_dir),
                        outdir=str(mm_outdir),
                        run_label=run_label,
                        event_names=[evt],
                        date=str(date),
                        steps=mm_steps,
                        members=mm_members,
                    )
                    LOG.info("Member maps written for event=%s date=%s", evt, date)
                except Exception:
                    LOG.error("Member maps failed for event=%s date=%s", evt, date, exc_info=True)
        else:
            mm_dates = mm_cfg.get("dates") or []
            mm_events = mm_cfg.get("events") or list(event_names)
            for date in mm_dates:
                try:
                    run_member_maps(
                        predictions_dir=str(predictions_dir),
                        outdir=str(mm_outdir),
                        run_label=run_label,
                        event_names=mm_events,
                        date=date,
                        steps=mm_steps,
                        members=mm_members,
                    )
                    LOG.info("Member maps written for date=%s", date)
                except Exception:
                    LOG.error("Member maps failed for date=%s", date, exc_info=True)

        # Combined PDF: merge all individual member map PDFs into one
        if mm_cfg.get("combined_pdf") and mm_outdir.exists():
            individual_pdfs = sorted(mm_outdir.glob("tc_members_*.pdf"))
            if len(individual_pdfs) > 1:
                try:
                    from pypdf import PdfReader, PdfWriter
                    writer = PdfWriter()
                    for pdf_path in individual_pdfs:
                        for page in PdfReader(str(pdf_path)).pages:
                            writer.add_page(page)
                    combined_name = "_".join(event_dates.keys()) if event_dates else "combined"
                    safe_label = run_label.replace(" ", "_").replace("/", "_")
                    combined_path = mm_outdir / f"tc_members_{combined_name}_{safe_label}.pdf"
                    with open(combined_path, "wb") as f:
                        writer.write(f)
                    LOG.info("Combined member maps PDF: %s", combined_path)
                except Exception:
                    LOG.error("Failed to merge member map PDFs", exc_info=True)

    return output_dir
