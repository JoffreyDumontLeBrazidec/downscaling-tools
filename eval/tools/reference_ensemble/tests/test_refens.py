import numpy as np
from eval.tools.reference_ensemble import refens


def test_run_date_next_year_and_override():
    assert refens.run_date("20231031") == "20241031"
    assert refens.run_date("20231031", {"20231031": "20241028"}) == "20241028"


def test_tp_window():
    end = np.array([0.010, 0.002, 0.001]); start = np.array([0.004, 0.002, 0.0015])
    np.testing.assert_allclose(refens.tp_window_mm(end, start), [6.0, 0.0, 0.0])
    np.testing.assert_allclose(refens.tp_window_mm(end, None), [10.0, 2.0, 1.0])


def test_mars_request_steps_and_members():
    r = refens.mars_request("truth", "20231031", 24, "pf", "/x.grib")
    assert "stream=enfh" in r and "step=12/24" in r and "number=1/to/10" in r and "hdate=20231031" in r
    r = refens.mars_request("input", "20230111", 12, "cf", "/x.grib")
    assert "stream=eefh" in r and "step=0/12" in r and "number" not in r and "date=20240111" in r


def test_read_cases(tmp_path):
    p = tmp_path / "c.txt"; p.write_text("# x\neurope 20231031 24 42.4 19.1 5.0 3.5\n")
    c = refens.read_cases(p)
    assert refens.case_key(c[0]) == "europe_20231031_L024" and c[0]["hx"] == 5.0
