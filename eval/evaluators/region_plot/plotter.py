"""Region plot evaluator — plot-only re-rendering of the region map grids.

The runner renders the figures in a subprocess (``eval._backends.region_plotting.plot_regions``)
and records what it drew in ``manifest.json``. ``plot`` reads that manifest and renders the
same figures again with the current plotting code: the same predictions file, regions,
panel keys, weather states, sample and member. Nothing is scored or rewritten except the
figures (and the manifest that lists them, when ``output_dir`` is the results directory).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    regions: list[str] | None = None,
) -> Path:
    """Re-render the region comparison figures described by ``results_dir/manifest.json``.

    ``regions`` restricts the re-render to a subset of the manifest's regions (by name).
    Without a manifest there is nothing to re-render and the plots directory is returned.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = results_dir / "manifest.json"
    if not manifest_path.exists():
        LOG.info("Region plot: no manifest in %s, nothing to re-render", results_dir)
        return plots_dir
    manifest = json.loads(manifest_path.read_text())
    boxes = {r["name"]: list(r["box"]) for r in manifest.get("regions", [])}
    if regions:
        boxes = {name: box for name, box in boxes.items() if name in set(regions)}

    from eval._backends.region_plotting.plot_regions import render_region_suite_from_predictions_file

    LOG.info("Region plot: re-rendering %d region(s) from %s into %s", len(boxes),
             manifest.get("predictions_file"), output_dir)
    render_region_suite_from_predictions_file(
        predictions_nc=manifest["predictions_file"],
        out_dir=output_dir,
        region_boxes=boxes,
        model_variables=manifest.get("model_variables"),
        weather_states=manifest.get("weather_states"),
        sample_index=int(manifest.get("sample_index", 0)),
        ensemble_member_index=int(manifest.get("ensemble_member_index", 0)),
        also_png=False,
        suite_kind=manifest.get("suite_kind", "regions"),
    )
    return plots_dir
