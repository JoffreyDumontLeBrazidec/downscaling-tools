"""ECMWF spectra evaluator — PDF plots."""
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
) -> Path:
    """Build consolidated spectra PDF with prediction + truth + input curves."""
    from ._plotter import build_pdf_ecmwf_with_references

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_amp_dir = results_dir / "spectra"
    if not pred_amp_dir.exists():
        LOG.warning("spectra_ecmwf plotter: spectra dir not found: %s", pred_amp_dir)
        return output_dir

    # References are addressed by evaluation window, so their directory name
    # cannot be reconstructed here. The evaluator records where it put them.
    truth_amp_dir: Path | None = None
    input_amp_dir: Path | None = None
    recorded = ""
    summary_path = results_dir / "spectra_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            recorded = str(summary.get("reference_spectra_dir", "") or "").strip()
        except (OSError, ValueError):
            recorded = ""

    if recorded:
        truth_amp_dir = Path(recorded)
        # truth and input sit side by side under the same window key
        input_amp_dir = Path(recorded.replace("/truth/", "/input/", 1))
        if input_amp_dir == truth_amp_dir:
            input_amp_dir = None
    else:
        reference_dir = eval_config.get("reference_dir", "")
        if reference_dir:
            LOG.info(
                "spectra_ecmwf plotter: no reference_spectra_dir recorded in %s; "
                "falling back to the pre-window-key layout under %s",
                summary_path, reference_dir,
            )
            truth_amp_dir = Path(reference_dir) / "truth" / "spectra"
            input_amp_dir = Path(reference_dir) / "input" / "spectra"

    for label, candidate in (("truth", truth_amp_dir), ("input", input_amp_dir)):
        if candidate is not None and not candidate.exists():
            LOG.warning(
                "spectra_ecmwf plotter: %s reference not found at %s; "
                "that curve will be missing from the PDF",
                label, candidate,
            )
    truth_label = eval_config.get("truth_label", "truth")
    input_label = eval_config.get("input_label", "input")

    out_pdf = output_dir / "spectra_ecmwf.pdf"
    try:
        n = build_pdf_ecmwf_with_references(
            pred_amp_dir,
            out_pdf,
            truth_amp_dir=truth_amp_dir,
            input_amp_dir=input_amp_dir,
            truth_label=truth_label,
            input_label=input_label,
        )
        LOG.info("spectra_ecmwf: wrote %d-page PDF: %s", n, out_pdf)
    except FileNotFoundError as exc:
        LOG.warning("spectra_ecmwf plotter: %s", exc)

    return output_dir
