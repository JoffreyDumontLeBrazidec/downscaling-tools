"""clip_nonnegative: named fields clipped at 0 before scoring (rule 7).

tp/cp are always clipped (house convention); clip_nonnegative adds further fields."""
import numpy as np
import pytest
import xarray as xr

from eval.evaluators.probabilistic.core.scoring import compute_probabilistic_scores


def _write(path, pred_tp, true_tp):
    m, g = pred_tp.shape
    states = np.array(["tp", "2t"])
    y_pred = np.zeros((1, m, g, 2), dtype=np.float32)
    y = np.zeros((1, m, g, 2), dtype=np.float32)
    y_pred[0, :, :, 0] = pred_tp
    y[0, :, :, 0] = true_tp
    y_pred[0, :, :, 1] = 280.0
    y[0, :, :, 1] = 280.0
    lat = np.linspace(-80, 80, g)
    lon = np.linspace(-170, 170, g)
    ds = xr.Dataset(
        {"y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y_pred),
         "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
         "lat_hres": (("grid_point_hres",), lat), "lon_hres": (("grid_point_hres",), lon)},
        coords={"weather_state": states, "ensemble_member": np.arange(1, m + 1)})
    ds.to_netcdf(path)


def _fcrps(out, field):
    import csv
    rows = list(csv.DictReader(open(out / "summary_by_lead.csv")))
    return float(next(r["mean"] for r in rows
                      if r["metric"] == "fcrps" and r["domain"] == "global" and r["weather_state"] == field))


def test_tp_is_always_clipped(tmp_path):
    rng = np.random.default_rng(0)
    pred = rng.normal(0.0, 1e-3, size=(4, 50))
    true = np.abs(rng.normal(0.0, 1e-3, size=(4, 50)))
    d = tmp_path / "pred"
    d.mkdir()
    _write(d / "predictions_20220819_step060.nc", pred, true)
    compute_probabilistic_scores(d, tmp_path / "raw", weather_states=["tp"], domains=["global"])
    compute_probabilistic_scores(d, tmp_path / "clip", weather_states=["tp"], domains=["global"],
                                 clip_nonnegative="tp")
    a, b = _fcrps(tmp_path / "raw", "tp"), _fcrps(tmp_path / "clip", "tp")
    assert a == b
    # same as scoring the clipped members by hand
    _write(d / "predictions_20220819_step060.nc", np.maximum(pred, 0.0), true)
    compute_probabilistic_scores(d, tmp_path / "hand", weather_states=["tp"], domains=["global"])
    assert _fcrps(tmp_path / "hand", "tp") == pytest.approx(a)


def test_clip_nonnegative_adds_fields(tmp_path):
    rng = np.random.default_rng(1)
    d = tmp_path / "pred"
    d.mkdir()
    pred = np.abs(rng.normal(0.0, 1e-3, size=(4, 50)))
    _write(d / "predictions_20220819_step060.nc", pred, pred)
    import xarray as xr2
    path = d / "predictions_20220819_step060.nc"
    ds = xr2.load_dataset(path)
    ds["y_pred"][..., 1] = rng.normal(0.0, 1.0, size=(4, 50))
    ds["y"][..., 1] = np.abs(rng.normal(0.0, 1.0, size=(4, 50)))
    ds.to_netcdf(path)
    compute_probabilistic_scores(d, tmp_path / "raw", weather_states=["2t"], domains=["global"])
    compute_probabilistic_scores(d, tmp_path / "clip", weather_states=["2t"], domains=["global"],
                                 clip_nonnegative=["2t"])
    assert _fcrps(tmp_path / "raw", "2t") != _fcrps(tmp_path / "clip", "2t")
