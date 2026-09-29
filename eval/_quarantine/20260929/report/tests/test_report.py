"""Tests of the retired HTML report (eval.report), moved out of
eval/tests/test_plotting_spec_helpers.py when the report was retired on 2026-09-29.
Not collected: pytest.ini skips eval/_quarantine, and eval.report no longer exists."""
from __future__ import annotations


def test_report_tab_names_are_readable():
    from eval.report import _tab_name

    assert _tab_name("spectra_ecmwf.pdf") == "Spectra"
    assert _tab_name("spectra_ecmwf_ratio.pdf") == "Spectra: ratio to truth"
    assert _tab_name("tc_members_idalia_mslp.pdf") == "TC Idalia members"
    assert _tab_name("quaver_crps_scores.pdf") == "Quaver CRPS scores"


def test_report_uses_the_figure_tokens_and_keeps_its_structure(tmp_path):
    from eval.report import generate_report

    run = tmp_path / "run"
    (run / "plots").mkdir(parents=True)
    (run / "plots" / "spectra_ecmwf.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (run / "data" / "scoreboard").mkdir(parents=True)
    (run / "data" / "scoreboard" / "scores.csv").write_text(
        "evaluator,metric,value,unit\ntc,tc_x_wind_max,40.0,m/s\n", encoding="utf-8")
    out = generate_report(run, tmp_path / "out" / "report.html")
    text = out.read_text(encoding="utf-8")
    assert "--model: #d62728" in text and "--truth: #000000" in text
    assert "DejaVu Sans" in text
    assert 'id="pdf-0"' in text and 'data-target="pdf-0"' in text
    assert "m s⁻¹" in text
    assert (tmp_path / "out" / "report_assets" / "spectra_ecmwf.pdf").exists()
