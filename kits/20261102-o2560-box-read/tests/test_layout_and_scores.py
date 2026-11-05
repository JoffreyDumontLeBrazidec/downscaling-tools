import json

import numpy as np
import pytest
from o2560_common import OUTPUT_VARS, fcrps_terms


def test_fair_crps_matches_brute_force():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(10, 7)); y = rng.normal(size=7)
    fc, sd, se = fcrps_terms(f, y)
    m = f.shape[0]
    brute = np.abs(f - y).mean(0) - np.abs(f[:, None] - f[None]).sum((0, 1)) / (2 * m * (m - 1))
    np.testing.assert_allclose(fc, brute, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, f.var(0, ddof=1)); np.testing.assert_allclose(se, (f.mean(0) - y) ** 2)   # variance, not std
    fc1, sd1, se1 = fcrps_terms(f[:1], y)
    np.testing.assert_allclose(fc1, np.abs(f[0] - y)); assert (sd1 == 0).all()


def test_written_file_reads_in_the_eval_tools(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    from o2560_box_predict import write_nc, file_complete
    P, Mm, S = 50, 3, len(OUTPUT_VARS)
    lat = np.linspace(11, 69, P); lon = np.linspace(-79, 39, P)
    y = np.random.default_rng(1).normal(size=(P, S)).astype(np.float32)
    yp = np.random.default_rng(2).normal(size=(Mm, P, S)).astype(np.float32)
    f = tmp_path / "predictions_20241101_step024.nc"
    write_nc(f, lat, lon, list(OUTPUT_VARS), y, y, yp, 0, 24, dict(seeds=json.dumps([1, 2, 3]), complete="yes"))
    assert file_complete(f, Mm) and not file_complete(f, 10)
    eval_box = pytest.importorskip("eval.evaluators.box_spectra.runner")
    fh = eval_box._File(f)
    assert fh.states == list(OUTPUT_VARS) and fh.members == [1, 2, 3]
    np.testing.assert_allclose(fh.column("y_pred", 2, "tp", np.arange(P)), yp[1, :, OUTPUT_VARS.index("tp")])
    np.testing.assert_allclose(fh.column("y", 2, "2t", np.arange(P)), y[:, OUTPUT_VARS.index("2t")])
    fh.close()
    from eval.discovery.predictions import find_predictions
    assert [(p.date, p.step) for p in find_predictions(tmp_path)] == [("20241101", 24)]
    xr = pytest.importorskip("xarray")
    from eval.evaluators.surface.core.compute import _to_member_point_weather
    ds = xr.open_dataset(f)
    assert _to_member_point_weather(ds["y_pred"], ds, label="y_pred").shape == (Mm, P, S)
    assert _to_member_point_weather(ds["y"], ds, label="y").shape == (1, P, S)
