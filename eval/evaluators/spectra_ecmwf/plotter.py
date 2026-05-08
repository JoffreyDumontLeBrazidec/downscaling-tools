"""ECMWF spectra evaluator — PDF plots."""
from __future__ import annotations

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

    reference_dir = eval_config.get("reference_dir", "")
    truth_amp_dir = Path(reference_dir) / "truth" / "spectra" if reference_dir else None
    input_amp_dir = Path(reference_dir) / "input" / "spectra" if reference_dir else None

    out_pdf = output_dir / "spectra_ecmwf.pdf"
    try:
        n = build_pdf_ecmwf_with_references(
            pred_amp_dir,
            out_pdf,
            truth_amp_dir=truth_amp_dir,
            input_amp_dir=input_amp_dir,
        )
        LOG.info("spectra_ecmwf: wrote %d-page PDF: %s", n, out_pdf)
    except FileNotFoundError as exc:
        LOG.warning("spectra_ecmwf plotter: %s", exc)

    return output_dir
