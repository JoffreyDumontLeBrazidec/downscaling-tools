"""precip_events evaluator — find heavy-precip events and render canonical plots.

Reads lane_config[precip_events]: n_events / dlat / dlon / rank_by, and
lane_config[precip] for the truth/baseline GRIB fallbacks (used when the
predictions embed no tp truth / no usable x_interp tp — the o1280->o2560
main-lane case).
Uses eval._backends.region_plotting.plot_precip_events to produce tight,
event-centered local pages.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from eval._backends.region_plotting.precip_events import find_precip_events

LOG = logging.getLogger(__name__)


def _validate_pdf(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 1024:
        raise RuntimeError(f"precip_events did not produce a usable PDF: {path}")


def run(
    predictions_dir,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir=None,
    overwrite: bool = False,
    checkpoint: str | None = None,
    **kwargs,
) -> Path:
    """Render local plots of the top-N heaviest-precip events."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "precip_events"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"precip_events output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_events = int(eval_config.get("n_events", 3))
    dlat = float(eval_config.get("dlat", 2.0))
    dlon = float(eval_config.get("dlon", 2.5))
    rank_by = eval_config.get("rank_by", "truth")
    var = str(eval_config.get("var", "tp"))
    precip_cfg = dict(lane_config.get("precip", {}))
    truth_grib_tpl = str(precip_cfg.get("truth_grib_tpl", ""))
    baseline_grib_tpl = str(precip_cfg.get("baseline_lres_grib_tpl", ""))
    interp_index_cache = str(precip_cfg.get("interp_index_cache", ""))

    events = find_precip_events(
        predictions_dir, n_events=n_events, dlat=dlat, dlon=dlon,
        rank_by=rank_by, var=var, truth_grib_tpl=truth_grib_tpl,
    )

    (output_dir / "events.json").write_text(
        json.dumps([{**asdict(e), "nc_path": str(e.nc_path)} for e in events], indent=2)
    )

    out_pdf = plots_dir / "precip_events_local.pdf"
    cmd = [
        sys.executable, "-m", "eval._backends.region_plotting.plot_precip_events",
        "--predictions-dir", str(predictions_dir),
        "--out", str(out_pdf),
        "--var", var,
        "--n-top", str(n_events),
        "--dlat", f"{dlat:g}",
        "--dlon", f"{dlon:g}",
        "--rank-by", str(rank_by),
    ]
    if truth_grib_tpl:
        cmd += ["--truth-grib-tpl", truth_grib_tpl]
    if baseline_grib_tpl:
        cmd += ["--baseline-grib-tpl", baseline_grib_tpl]
    if interp_index_cache:
        cmd += ["--interp-index-cache", interp_index_cache]
    run_label = str(eval_config.get("run_label") or kwargs.get("run_label") or "")
    if run_label:
        cmd += ["--run-label", run_label]

    LOG.info("precip_events subprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    _validate_pdf(out_pdf)
    return output_dir
