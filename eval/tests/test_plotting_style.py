"""Tests for the shared house plotting style (``eval.plotting``)."""
from __future__ import annotations

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from eval import plotting as P  # noqa: E402


def test_import_has_no_global_side_effects():
    code = (
        "import matplotlib; before = dict(matplotlib.rcParams);"
        "import eval.plotting, sys;"
        "assert dict(matplotlib.rcParams) == before;"
        "assert 'seaborn' not in sys.modules;"
        "assert 'cartopy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_role_colours_and_reference_styles_are_distinct():
    assert P.role_style("truth")["color"] == "#000000"
    assert P.role_style("model")["color"] == "#d62728"
    assert P.role_style("input")["color"] == "#1f77b4"
    assert P.role_style("input")["linestyle"] != "-"
    pairs = [(s.color.lower(), str(s.linestyle)) for s in P.REFERENCE_STYLES]
    assert len(pairs) == len(set(pairs))
    role_colours = {"#000000", "#d62728", "#1f77b4"}
    assert not role_colours & {c.lower() for c in P.SEQUENCE}


def test_reference_styles_never_collide():
    keys = ["ENFO_O320_0001", "ENFO_O1280_0001", "EEFO_O96_0001", "ENFO_O96_0001", "ENFO_O48_0001"]
    styles = P.reference_styles(keys)
    seen = {(s["color"], str(s["linestyle"])) for s in styles.values()}
    assert len(seen) == len(keys)


def test_style_for_key_maps_raw_keys_to_roles():
    assert P.role_of("od_enfo_0001") == "truth"
    assert P.role_of("y_pred_0") == "model"
    assert P.role_of("eval_inputs") == "input"
    assert P.role_of("something_else") is None


def test_variable_conversions():
    assert P.convert("msl", np.array([101325.0]))[0] == pytest.approx(1013.25)
    assert P.convert("z_500", np.array([98.0665]))[0] == pytest.approx(1.0)
    assert P.convert("2t", np.array([280.0]))[0] == 280.0
    assert P.convert("tp", np.array([0.002]))[0] == pytest.approx(2.0)
    assert P.convert("tp", np.array([2.0]), native_unit="mm")[0] == 2.0
    assert P.convert("10ff", np.array([7.0]))[0] == 7.0


def test_variable_lookup_spellings():
    assert P.variable_spec("10u_sfc").key == "10u"
    assert P.variable_spec("wind").key == "10ff"
    assert P.variable_spec("t_850").unit == "K"
    assert P.variable_spec("q_700").unit == "g kg⁻¹"
    assert P.axis_label("msl") == "Mean sea level pressure (hPa)"
    unknown = P.variable_spec("weird_var")
    assert unknown.name == "weird_var" and unknown.cmap == "viridis"


def test_no_forbidden_colormaps_in_table():
    for spec in P.VARIABLES.values():
        assert spec.cmap.lower() not in {"jet", "rainbow", "turbo", "hsv", "gist_rainbow"}
    assert all(s.err_cmap in {"RdBu_r", "BrBG"} for s in P.VARIABLES.values())


def test_readable_labels():
    assert P.readable_label("od_enfo_0001") == "truth (ENFO O1280)"
    assert P.readable_label("od_enfo_0001", P.LabelContext(truth="ENFO O320")) == "truth (ENFO O320)"
    assert P.readable_label("eval_inputs") == "input"
    assert P.readable_label("residuals_pred_0").startswith("predicted residual")
    assert P.readable_label("rmse_ens_mean") == "RMSE of the ensemble mean"
    assert P.readable_label("ENFO_O1280_0001") == "ENFO O1280"
    assert P.readable_label("EEFO_O96_0001", with_id=True) == "EEFO O96 (EEFO_O96_0001)"


def test_projection_rule():
    from cartopy import crs

    assert isinstance(P.select_projection(-20, 30, 30, 60), crs.LambertConformal)
    assert isinstance(P.select_projection(170, -170, 0, 30), crs.PlateCarree)


def test_symmetric_norm_is_centred():
    norm, lim = P.symmetric_norm(np.array([-1.0, 3.0]), np.array([0.5]), q=100)
    assert norm.vmin == -norm.vmax == -3.0 and lim == 3.0


def test_save_figure_writes_png_and_pdf(tmp_path):
    with P.eval_style():
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], **P.role_style("model"))
    paths = P.save_figure(fig, tmp_path / "sub" / "fig.png", close=True)
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_figure_book_writes_pdf_and_pages(tmp_path):
    with P.FigureBook(tmp_path / "book", png=True) as book:
        for name in ("a", "b"):
            fig, ax = plt.subplots()
            ax.plot([0, 1])
            book.add(fig, name=name)
    assert (tmp_path / "book.pdf").exists()
    assert len(list((tmp_path / "book_pages").glob("*.png"))) == 2


def test_probabilistic_figure(tmp_path):
    from eval.plotting.probabilistic import SOURCE_LOCAL, plot_probabilistic_scores

    rows = []
    for metric in ("fcrps", "spread"):
        for role, label, scale in (("model", "model", 1.0), ("input", "input", 1.3)):
            for lead in (24, 48, 72):
                rows.append(dict(metric=metric, variable="msl", domain="europe", lead_h=lead,
                                 series_role=role, series_label=label, value=100.0 * scale * lead,
                                 ci_low=90.0 * scale * lead, ci_high=110.0 * scale * lead))
    written = plot_probabilistic_scores(pd.DataFrame(rows), SOURCE_LOCAL, tmp_path / "prob")
    assert (tmp_path / "prob.pdf") in written
    assert list((tmp_path / "prob_pages").glob("*.png"))
