"""precip_events evaluator — find heavy-precip events, render each via plot_regions.

Reads lane_config[precip_events]: n_events / dlat / dlon / rank_by.
Reuses the eval._backends.region_plotting.plot_regions renderer with tp-specific
3-panel layout (x_0, y_0, y_pred_0) and wide bounding boxes; one
subprocess per event, then merges the per-event PDFs into precip_events_local.pdf.
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


def _merge_pdfs(pdf_paths: list[Path], out_pdf: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for p in pdf_paths:
        if p.exists():
            writer.append(str(p))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as fh:
        writer.write(fh)
    writer.close()


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

    n_events = int(eval_config.get("n_events", 8))
    dlat = float(eval_config.get("dlat", 50))
    dlon = float(eval_config.get("dlon", 60))
    rank_by = eval_config.get("rank_by", "truth")

    events = find_precip_events(
        predictions_dir, n_events=n_events, dlat=dlat, dlon=dlon, rank_by=rank_by
    )

    (output_dir / "events.json").write_text(
        json.dumps([{**asdict(e), "nc_path": str(e.nc_path)} for e in events], indent=2)
    )

    per_event_pdfs: list[Path] = []
    for e in events:
        event_dir = plots_dir / e.label
        event_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "eval._backends.region_plotting.plot_regions",
            "--predictions-nc", str(e.nc_path),
            "--out-dir", str(event_dir),
            "--region-boxes-json", json.dumps({e.label: e.bbox}),
            "--model-variables", "x_0,y_0,y_pred_0",
            "--weather-states", "tp",
        ]
        LOG.info("precip_events render %s: %s", e.label, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            per_event_pdfs.append(event_dir / "all_regions_plots.pdf")
        except subprocess.CalledProcessError:
            LOG.error("precip_events: render failed for %s, skipping", e.label)

    _merge_pdfs(per_event_pdfs, plots_dir / "precip_events_local.pdf")
    return output_dir
