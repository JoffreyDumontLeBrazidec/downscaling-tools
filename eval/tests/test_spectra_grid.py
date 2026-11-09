"""The standard spectra figure (eval.plotting.spectra_grid) and its two feeders."""
from __future__ import annotations

import csv

import numpy as np

from eval.plotting.spectra_grid import (
    RATIO_LIM,
    Curve,
    order_variables,
    plot_spectra_grid,
    ratio_to_reference,
    wavelength_to_wavenumber,
)


def _curve(role, scale=1.0, n=640):
    wvn = np.arange(0, n + 1, dtype=float)
    power = scale * (1.0 + wvn) ** -3.0
    return Curve(role.capitalize(), wvn, power, role=role)


def test_order_follows_the_reference_figure_by_any_spelling():
    got = order_variables(["z_500", "ws", "msl", "2t", "foo", "tp", "tcc"])
    assert got == ["tcc", "tp", "ws", "2t", "msl", "z_500", "foo"]


def test_ratio_is_one_on_identical_curves_and_scales_with_power():
    c = _curve("truth")
    wvn, ratio = ratio_to_reference(c.wavenumber, 2.0 * c.power, c.wavenumber, c.power)
    assert wvn.min() > 0
    np.testing.assert_allclose(ratio, 2.0)


def test_wavelength_maps_to_total_wavenumber():
    np.testing.assert_allclose(wavelength_to_wavenumber([40030.0, 31.3]), [1.0, 40030.0 / 31.3])


def test_grid_writes_png_and_pdf_with_ratio_axis_0p3_to_3(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seen = {}
    real_close = plt.close

    def keep(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            seen["axes"] = list(fig.axes)
        return real_close(fig)

    monkeypatch.setattr(plt, "close", keep)
    panels = {v: [_curve("truth"), _curve("input", 0.8, 320), _curve("model", 1.1)]
              for v in ["2t", "tp", "msl", "10ff", "z_500"]}
    written = plot_spectra_grid(panels, tmp_path / "grid")
    assert sorted(p.suffix for p in written) == [".pdf", ".png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    ratio_axes = [ax for ax in seen["axes"] if np.allclose(ax.get_ylim(), RATIO_LIM)]
    assert len(ratio_axes) == 5
    assert all(np.allclose(ax.get_ylim(), RATIO_LIM) for ax in ratio_axes)


def test_spectra_ecmwf_v2_grid_from_curve_files(tmp_path):
    from eval.evaluators.spectra_ecmwf_v2._plotter import build_spectra_grid

    wvn = np.arange(0, 1281, dtype=float)
    for root, scale, n in (("pred", 1.1, 1281), ("truth", 1.0, 1281), ("input", 0.7, 321)):
        d = tmp_path / root / "2t"
        d.mkdir(parents=True)
        for k in range(2):
            name = f"20230101_0000_024_{k}_2t_sfc.npy"
            np.save(d / f"ampl_{name}", scale * (1 + wvn[:n]) ** -1.5)
            np.save(d / f"wvn_{name}", wvn[:n])
    written = build_spectra_grid(tmp_path / "pred", tmp_path / "out" / "spectra_ecmwf_grid",
                                 truth_amp_dir=tmp_path / "truth", input_amp_dir=tmp_path / "input")
    assert {p.name for p in written} == {"spectra_ecmwf_grid.png", "spectra_ecmwf_grid.pdf"}


def test_box_spectra_grid_from_bins_csv(tmp_path):
    from eval.evaluators.box_spectra.plotter import grid_panels, plot_grid
    from eval.evaluators.box_spectra.runner import BIN_COLS

    rows = []
    for var in ("ws", "msl"):
        for km in (500.0, 250.0, 125.0, 60.0, 31.0):
            r = {c: "" for c in BIN_COLS}
            r.update(scope="all", variable=var, bin_centre_km=km, amp_truth_full=km ** 1.5,
                     amp_interp=0.5 * km ** 1.5, amp_pred_full=0.9 * km ** 1.5, n_draws=120)
            rows.append(r)
    with open(tmp_path / "bins.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=BIN_COLS)
        w.writeheader()
        w.writerows(rows)
    panels = grid_panels([{k: str(v) for k, v in r.items()} for r in rows])
    truth, inp, model = panels["ws"]
    np.testing.assert_allclose(model.power / truth.power, 0.81)
    np.testing.assert_allclose(inp.power / truth.power, 0.25)
    assert np.all(np.diff(truth.wavenumber) > 0)
    written = plot_grid(tmp_path)
    assert {p.name for p in written} == {"box_spectra_grid.png", "box_spectra_grid.pdf"}
