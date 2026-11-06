"""Regional (E2 box) files in wind_extremes and texture: synthetic grids."""
import numpy as np
import pytest
import scipy.sparse as sps

xr = pytest.importorskip("xarray")


def _grid(step=1.0):
    lat, lon = np.meshgrid(np.arange(-89.5, 90, step), np.arange(0, 360, step), indexing="ij")
    return lat.ravel(), lon.ravel()


def test_wind_extremes_regional_uses_x_interp_and_clips_boxes(tmp_path):
    from eval.evaluators.wind_extremes import runner as W

    glat, glon = _grid(1.0)
    n_full = glat.size
    up = sps.identity(n_full, format="csr")
    up_path = tmp_path / "up.npz"
    sps.save_npz(up_path, up)
    lon180 = np.where(glon > 180, glon - 360, glon)
    inbox = (glat > 30) & (glat < 70) & (lon180 > -20) & (lon180 < 40)
    lat, lon = glat[inbox], glon[inbox]
    n = lat.size
    rng = np.random.default_rng(0)
    states = ["10u", "10v"]
    y = rng.normal(5, 2, (1, 2, n, 2)).astype(np.float32)
    ds = xr.Dataset(
        {"x": (("sample", "ensemble_member", "grid_point_lres", "weather_state"),
               np.zeros((1, 2, 7, 2), np.float32)),
         "x_interp": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y * 0.5),
         "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y),
         "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), y * 1.1),
         "lat_hres": (("grid_point_hres",), lat), "lon_hres": (("grid_point_hres",), lon),
         "weather_state": (("weather_state",), np.array(states)),
         "ensemble_member": (("ensemble_member",), np.array([1, 2]))})
    pdir = tmp_path / "pred"
    pdir.mkdir()
    ds.to_netcdf(pdir / "predictions_20230101_step024.nc")
    import eval.discovery.predictions as P
    P_orig = P.valid_prediction_files
    P.valid_prediction_files = lambda d, steps=None: sorted(d.glob("predictions_*.nc"))
    try:
        out = W.run(pdir, {}, {"paths": {"up_matrix": str(up_path)}, "radii_km": [100.0]},
                    output_dir=tmp_path / "out")
    finally:
        P.valid_prediction_files = P_orig
    import json
    js = json.loads((out / "wind_extremes.json").read_text())
    assert set(js["boxes"]) == {"europe", "open_north_atlantic"}   # tropical box dropped
    assert js["boxes"]["open_north_atlantic"]["box"][2] > -20      # clipped inside the data
    assert js["config"]["point_area_km2"] == pytest.approx(W.SPHERE_AREA_KM2 / n_full)
    s = js["samples"][0]
    assert s["input"]["peak"] == pytest.approx(s["truth"]["peak"] * 0.5, rel=1e-5)


def test_texture_regional_embedding_valid_interior():
    from eval.evaluators.texture.runner import _regional_embedding

    glat, glon = _grid(2.0)
    n = glat.size
    nxt = np.arange(n).reshape(90, 180)
    nxt = np.roll(nxt, -1, axis=1).ravel()
    static = {"lat": glat, "lon": glon, "nxt": nxt, "nn": nxt.copy()}
    eye = sps.identity(n, format="csr")
    box = (glat > 20) & (glat < 60) & (glon > 10) & (glon < 50)
    gidx, valid = _regional_embedding(glat[box], glon[box], static, eye, eye)
    assert np.array_equal(np.sort(gidx), np.flatnonzero(box))
    assert valid.sum() > 0 and not valid[~box].any()
    assert not valid[box & (glon > 47)].any()        # last column: successor outside the box
