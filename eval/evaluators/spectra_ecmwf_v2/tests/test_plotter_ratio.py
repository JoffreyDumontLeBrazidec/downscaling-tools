"""Tests for the ratio-to-truth spectra plot of spectra_ecmwf_v2.

The ratio plot divides the prediction curve and the model-input curve by the
truth curve, so that departures of a few percent are readable instead of
being hidden inside the thickness of a log-log line.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.evaluators.spectra_ecmwf_v2 import plotter
from eval.evaluators.spectra_ecmwf_v2._plotter import _ratio_to_reference
from eval.evaluators.spectra_ecmwf_v2._plotter import _ratio_ylim
from eval.evaluators.spectra_ecmwf_v2._plotter import build_pdf_ecmwf_ratio


def _write_curves(base: Path, param: str, wvn: np.ndarray, ampl: np.ndarray, n: int) -> None:
    d = base / param
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        np.save(d / f"ampl_{i:03d}.npy", ampl)
        np.save(d / f"wvn_{i:03d}.npy", wvn)


def _power_law(j: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    wvn = np.arange(j + 1, dtype=float)
    ampl = np.empty_like(wvn)
    ampl[0] = scale
    ampl[1:] = scale * wvn[1:] ** -2.0
    return wvn, ampl


def _count_pdf_pages(path: Path) -> int:
    import re

    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


# ---------------------------------------------------------------------------
# _ratio_to_reference
# ---------------------------------------------------------------------------

def test_ratio_of_a_curve_with_itself_is_one() -> None:
    wvn, ampl = _power_law(64)

    got = _ratio_to_reference(wvn, ampl, wvn, ampl)

    assert got is not None
    out_wvn, ratio = got
    assert np.all(out_wvn > 0)
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-12)


def test_ratio_recovers_a_constant_scale_factor() -> None:
    wvn, truth = _power_law(64, scale=1.0)
    _, pred = _power_law(64, scale=0.9)

    got = _ratio_to_reference(wvn, pred, wvn, truth)

    assert got is not None
    np.testing.assert_allclose(got[1], 0.9, rtol=1e-12)


def test_ratio_is_restricted_to_the_shared_wavenumbers() -> None:
    """A coarse driver stops at its own truncation; the ratio stops there too."""
    twvn, truth = _power_law(200)
    iwvn, inp = _power_law(50)

    got = _ratio_to_reference(iwvn, inp, twvn, truth)

    assert got is not None
    out_wvn, ratio = got
    assert out_wvn.min() >= 1.0
    assert out_wvn.max() == pytest.approx(50.0)
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-12)


def test_ratio_drops_nonpositive_and_nonfinite_samples() -> None:
    wvn, truth = _power_law(32)
    pred = truth.copy()
    pred[5] = np.nan
    pred[7] = 0.0
    pred[9] = -1.0

    got = _ratio_to_reference(wvn, pred, wvn, truth)

    assert got is not None
    out_wvn, ratio = got
    assert np.all(np.isfinite(ratio))
    for bad in (5.0, 7.0, 9.0):
        assert bad not in set(out_wvn.tolist())


def test_ratio_returns_none_without_usable_overlap() -> None:
    a_wvn, a_amp = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])
    b_wvn, b_amp = np.array([100.0, 200.0]), np.array([1.0, 1.0])

    assert _ratio_to_reference(a_wvn, a_amp, b_wvn, b_amp) is None
    assert _ratio_to_reference(a_wvn, a_amp[:2], b_wvn, b_amp) is None


# ---------------------------------------------------------------------------
# _ratio_ylim
# ---------------------------------------------------------------------------

def test_ylim_is_symmetric_about_one_in_the_log_sense() -> None:
    lo, hi = _ratio_ylim([np.full(100, 1.02)])

    assert lo * hi == pytest.approx(1.0, rel=1e-12)
    assert lo < 1.0 < hi


def test_ylim_ignores_a_handful_of_extreme_values() -> None:
    """A prediction that collapses at the truncation must not flatten the rest."""
    curve = np.concatenate([np.full(990, 1.03), np.full(10, 1e-6)])

    lo, hi = _ratio_ylim([curve])

    assert hi < 5.0
    assert lo > 0.2


def test_ylim_falls_back_when_there_is_nothing_to_plot() -> None:
    lo, hi = _ratio_ylim([])

    assert lo < 1.0 < hi


# ---------------------------------------------------------------------------
# build_pdf_ecmwf_ratio
# ---------------------------------------------------------------------------

def test_ratio_pdf_has_one_page_per_parameter(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    truth = tmp_path / "truth"
    inp = tmp_path / "input"
    for param in ("10u", "2t", "z_500"):
        wvn, ampl = _power_law(128)
        _write_curves(pred, param, wvn, ampl * 0.95, 4)
        _write_curves(truth, param, wvn, ampl, 3)
        _write_curves(inp, param, *_power_law(32), 3)

    out = tmp_path / "ratio.pdf"
    n = build_pdf_ecmwf_ratio(pred, out, truth_amp_dir=truth, input_amp_dir=inp)

    assert n == 3
    assert _count_pdf_pages(out) == 3


def test_ratio_pdf_needs_no_input_curve(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    truth = tmp_path / "truth"
    wvn, ampl = _power_law(64)
    _write_curves(pred, "msl", wvn, ampl, 2)
    _write_curves(truth, "msl", wvn, ampl, 2)

    n = build_pdf_ecmwf_ratio(pred, tmp_path / "ratio.pdf", truth_amp_dir=truth)

    assert n == 1


def test_ratio_pdf_is_not_written_without_truth(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    _write_curves(pred, "msl", *_power_law(64), 2)
    out = tmp_path / "ratio.pdf"

    assert build_pdf_ecmwf_ratio(pred, out, truth_amp_dir=None) == 0
    assert not out.exists()


def test_parameter_without_a_truth_curve_is_skipped(tmp_path: Path) -> None:
    pred = tmp_path / "pred"
    truth = tmp_path / "truth"
    wvn, ampl = _power_law(64)
    _write_curves(pred, "10u", wvn, ampl, 2)
    _write_curves(pred, "z_500", wvn, ampl, 2)
    _write_curves(truth, "10u", wvn, ampl, 2)

    n = build_pdf_ecmwf_ratio(pred, tmp_path / "ratio.pdf", truth_amp_dir=truth)

    assert n == 1


# ---------------------------------------------------------------------------
# plotter.plot wiring
# ---------------------------------------------------------------------------

def test_plot_writes_both_the_absolute_and_the_ratio_pdf(tmp_path: Path) -> None:
    results = tmp_path / "results"
    truth = tmp_path / "ref" / "truth" / "win" / "spectra"
    inp = tmp_path / "ref" / "input" / "win" / "spectra"
    wvn, ampl = _power_law(128)
    _write_curves(results / "spectra", "10u", wvn, ampl * 0.95, 3)
    _write_curves(truth, "10u", wvn, ampl, 2)
    _write_curves(inp, "10u", *_power_law(32), 2)
    results.mkdir(parents=True, exist_ok=True)
    (results / "spectra_summary.json").write_text(
        json.dumps({"reference_spectra_dir": str(truth)}), encoding="utf-8"
    )

    out = plotter.plot(results, {}, {"truth_label": "ENFO", "input_label": "EEFO"})

    assert (Path(out) / "spectra_ecmwf.pdf").exists()
    assert (Path(out) / "spectra_ecmwf_ratio.pdf").exists()


def test_plot_skips_the_ratio_pdf_when_no_reference_is_known(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_curves(results / "spectra", "10u", *_power_law(64), 2)

    out = plotter.plot(results, {}, {})

    assert not (Path(out) / "spectra_ecmwf_ratio.pdf").exists()
