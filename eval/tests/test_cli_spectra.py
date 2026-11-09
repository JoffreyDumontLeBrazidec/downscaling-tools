"""``eval.cli spectra``: predictions folder -> box read per lead -> standard grid; bins.csv drawn as is."""
from __future__ import annotations

from eval.cli import build_parser, main
from eval.evaluators.box_spectra.tests.test_box_spectra import _write


def test_spectra_is_a_figures_command():
    args = build_parser().parse_args(["spectra", "x", "--out", "o"])
    assert args._command.name == "spectra" and args._command.group == "figures"
    assert args.steps == [24] and args.variables[0] == "tp"


def test_predictions_folder_then_bins_csv(tmp_path):
    pred = tmp_path / "B100e2_m"
    pred.mkdir()
    for i, date in enumerate(("20241101", "20241102")):
        _write(pred / f"predictions_{date}_step024.nc", mode="noise", seed=i)
    main(["spectra", str(pred), "--out", str(tmp_path / "out")])
    row = tmp_path / "out" / "B100e2_m_L024"
    assert (row / "box_spectra_grid.png").exists() and (row / "box_spectra_grid.pdf").exists()
    assert (row / "box_spectra" / "bins.csv").exists()
    # an existing box read is drawn without recomputing
    main(["spectra", str(row / "box_spectra"), "--label", "again", "--out", str(tmp_path / "out2")])
    assert (tmp_path / "out2" / "again" / "box_spectra_grid.png").exists()
