"""Tests for ``eval.plotting.spec_helpers`` and the house-style spectra PDF pages."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from eval.plotting import spec_helpers as H  # noqa: E402


def test_power_unit_squares_the_display_unit():
    assert H.power_unit("hPa") == "hPa²"
    assert H.power_unit("m s⁻¹") == "m² s⁻²"
    assert H.power_unit("K") == "K²"
    assert H.power_unit("furlong") == "(furlong)²"
    assert H.power_unit("") == ""


def test_spectral_power_label_uses_the_variable_table():
    assert H.spectral_power_label("msl_sfc") == "Spectral power (hPa²)"
    assert H.spectral_power_label("z_500") == "Spectral power (dam²)"
    assert H.spectral_power_label("not_a_variable") == "Spectral power"
    assert H.spectral_power_label(None) == "Spectral power"


def test_amplitude_to_power_converts_before_squaring():
    # 100 Pa amplitude = 1 hPa -> power 1 hPa²
    np.testing.assert_allclose(H.amplitude_to_power("msl", [100.0, 200.0]), [1.0, 4.0])
    # temperature has no offset in a spectral amplitude
    np.testing.assert_allclose(H.amplitude_to_power("2t", [3.0]), [9.0])


def test_wavelength_axis_is_its_own_inverse():
    ell = np.array([1.0, 10.0, 400.0])
    np.testing.assert_allclose(H._wavelength(H._wavelength(ell)), ell)
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    sec = H.add_wavelength_axis(ax)
    assert sec.get_xlabel() == "Wavelength (km)"
    plt.close(fig)


def test_count_phrase():
    assert H.count_phrase(1, "date") == "1 date"
    assert H.count_phrase(5, "date") == "5 dates"
    assert H.count_phrase(2, "lead time") == "2 lead times"


def test_smooth_series_skips_nans_and_keeps_length():
    v = np.array([1.0, np.nan, 3.0, 3.0])
    out = H.smooth_series(v, span=3)
    assert out.shape == v.shape
    assert out[0] == 1.0 and out[1] == 1.0
    assert 1.0 < out[2] < 3.0
    np.testing.assert_array_equal(H.smooth_series(v, span=1), v)


def test_tint_moves_towards_white():
    r, g, b = H.tint("#000000", 0.5)
    assert (r, g, b) == pytest.approx((0.5, 0.5, 0.5))


# --------------------------------------------------------------------------- spectra pages

def _write_curves(base: Path, param: str, n: int, scale: float = 1.0) -> None:
    d = base / param
    d.mkdir(parents=True, exist_ok=True)
    wvn = np.arange(1, 129, dtype=float)
    for i in range(n):
        date = 20250926 + i
        np.save(d / f"ampl_{date}_24_{param}_1_n1.npy", scale * wvn ** -1.5)
        np.save(d / f"wvn_{date}_24_{param}_1_n1.npy", wvn)


def test_sample_note_counts_dates_and_lead_times(tmp_path: Path):
    from eval.evaluators.spectra_ecmwf_v2._plotter import _sample_note

    _write_curves(tmp_path, "msl_sfc", 3)
    files = sorted((tmp_path / "msl_sfc").glob("ampl_*.npy"))
    assert _sample_note(files) == "n = 3: 3 dates, 1 lead time"
    assert _sample_note([Path("ampl_weird.npy")]) == "n = 1"


def test_reference_pdf_writes_png_pages(tmp_path: Path):
    from eval.evaluators.spectra_ecmwf_v2._plotter import build_pdf_ecmwf_with_references

    _write_curves(tmp_path / "pred", "msl_sfc", 2, 0.9)
    _write_curves(tmp_path / "truth", "msl_sfc", 2)
    out = tmp_path / "spectra_ecmwf.pdf"
    n = build_pdf_ecmwf_with_references(tmp_path / "pred", out, truth_amp_dir=tmp_path / "truth",
                                        truth_label="ENFO")
    assert n == 1
    assert out.exists()
    assert len(list((tmp_path / "spectra_ecmwf_pages").glob("*.png"))) == 1


def test_build_pdf_modes(tmp_path: Path):
    """Both auto-detected modes of build_pdf still produce one page per curve set."""
    from eval.evaluators.spectra_ecmwf_v2._plotter import build_pdf

    ecmwf = tmp_path / "ecmwf"
    _write_curves(ecmwf, "10u_sfc", 3)
    assert build_pdf(ecmwf, tmp_path / "ecmwf.pdf") == 1

    proxy = tmp_path / "proxy"
    proxy.mkdir()
    ell = list(range(64))
    curve = [float(max(i, 1) ** -2.0) for i in ell]
    scope = {"status": "ok", "n_curves": 4, "wavenumbers": ell, "prediction_mean": curve,
             "truth_mean": curve, "relative_l2_mean_curve": 0.05}
    summary = {"run_label": "t", "score_wavenumber_min_exclusive": 10.0,
               "weather_states": {"2t": {"status": "ok",
                                         "scopes": {"full_field": scope, "residual": scope}}}}
    (proxy / "spectra_curve_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert build_pdf(proxy, tmp_path / "proxy.pdf") == 2
