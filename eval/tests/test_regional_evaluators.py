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


def test_texture_regional_subset_valid_interior():
    """The regional texture path (one implementation: _subset_static, shared with
    texture/tests/test_regional_mode.py) keeps only box points, leaves the edge margin out
    of every stratum and rebuilds the zonal successor inside the box."""
    from eval.evaluators.texture.runner import _subset_static

    glat, glon = _grid(2.0)
    n = glat.size
    static = {"lat": glat, "lon": glon, "masks": {"all": None}, "strata": {"all": {}},
              "grid": {"n_points": n}}
    eye = sps.identity(n, format="csr")
    box = (glat > 20) & (glat < 60) & (glon > 10) & (glon < 50)
    idx = np.flatnonzero(box)
    sub, up_s, down_s = _subset_static(static, idx, eye, eye, nn_count=4, edge_margin_deg=2.5,
                                       min_points=10)
    assert sub["lat"].size == idx.size and up_s.shape == (idx.size, n) and down_s.shape == (n, idx.size)
    keep = sub["masks"]["all"]
    assert keep.sum() > 0 and not keep[sub["lon"] > 47].any() and not keep[sub["lat"] < 23].any()
    assert sub["nxt"].max() < idx.size                 # successors are indices into the box
    assert sub["grid"]["regional"]["n_global"] == n
