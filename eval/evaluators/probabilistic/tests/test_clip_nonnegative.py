"""clip_nonnegative: tp members and truth clipped at 0 before scoring (rule 7)."""
import numpy as np
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


def test_clip_changes_tp_only(tmp_path):
    rng = np.random.default_rng(0)
    pred = rng.normal(0.0, 1e-3, size=(4, 50))
    true = np.abs(rng.normal(0.0, 1e-3, size=(4, 50)))
    d = tmp_path / "pred"
    d.mkdir()
    _write(d / "predictions_20220819_step060.nc", pred, true)
    raw = compute_probabilistic_scores(d, tmp_path / "raw", weather_states=["tp"], domains=["global"])
    clip = compute_probabilistic_scores(d, tmp_path / "clip", weather_states=["tp"], domains=["global"],
                                        clip_nonnegative=["tp"])
    import csv
    def fcrps(out):
        rows = list(csv.DictReader(open(out / "summary_by_lead.csv")))
        return float(next(r["mean"] for r in rows if r["metric"] == "fcrps" and r["domain"] == "global"))
    a, b = fcrps(tmp_path / "raw"), fcrps(tmp_path / "clip")
    assert a != b
    clipped = compute_probabilistic_scores(d, tmp_path / "c2", weather_states=["tp"], domains=["global"],
                                           clip_nonnegative="tp")
    assert fcrps(tmp_path / "c2") == b
    assert raw and clip and clipped
