"""--layout scales and --layout fronts of zoom_maps: the split, the gradient and the render."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("scipy")

from eval.evaluators.probabilistic.core.scale_split import LowPassOperator
from eval.evaluators.zoom_maps.core import scale_maps
from eval.evaluators.zoom_maps.core.plot_member_wind_maps import build_arg_parser, run


def _lattice(step: float = 0.2):
    lat1 = np.arange(35.0, 65.0, step)
    lon1 = np.arange(-40.0, 10.0, step)
    lon_g, lat_g = np.meshgrid(lon1, lat1)
    return lat_g.ravel(), lon_g.ravel()


def _km_x(lat, lon):
    return (lon + 15.0) * 111.2 * np.cos(np.deg2rad(lat))


def _large(lat, lon):
    """A 2,000 km wave: almost all of it in the large part of the 400 km split."""
    return np.sin(2 * np.pi * _km_x(lat, lon) / 2000.0) * np.cos(2 * np.pi * (lat - 50.0) * 111.2 / 2000.0)


def _fine(lat, lon, rng):
    """Independent noise on the 22 km lattice: almost all of it in the fine part."""
    return rng.standard_normal(lat.size)


def test_split_reads_a_half_amplitude_large_scale_change_and_independent_detail():
    rng = np.random.default_rng(0)
    lat, lon = _lattice()
    inp = 1000.0 + 10.0 * np.tanh(_km_x(lat, lon) / 100.0)
    truth = inp + 4.0 * _large(lat, lon) + _fine(lat, lon, rng)
    model = inp + 2.0 * _large(lat, lon) + _fine(lat, lon, rng)
    op = LowPassOperator(lat, lon, None, cut_km=400.0)
    mask = scale_maps.in_box(lat, lon, (-25.0, -5.0, 43.0, 57.0))
    out = scale_maps.split_residuals(inp, truth, model, op, mask)
    s, f = out["stats"], out["fields"]
    assert 0.9 < s["large_corr"] <= 1.0                       # the model's large change goes towards the truth
    assert abs(s["model_over_truth_large"] - 0.5) < 0.08       # at half the amplitude
    assert abs(s["fine_corr"]) < 0.05                          # its detail is its own
    assert s["truth_large_share"] > 0.7                        # 4 x wave against unit noise
    # the parts add up, and "aligned" is the input plus the fine part
    np.testing.assert_allclose(f["truth_large"] + f["truth_fine"], truth - inp, atol=1e-9)
    np.testing.assert_allclose(f["model_aligned"], inp + f["model_fine"], atol=1e-9)


def test_temperature_gradient_peaks_on_the_front():
    lat, lon = _lattice(0.1)
    x = _km_x(lat, lon)
    temp = 280.0 - 5.0 * np.tanh(x / 100.0)                   # 10 K across a 200 km zone: 5 K / 100 km at x = 0
    gx, gy, grad = scale_maps.temperature_gradient(lat, lon, temp, (-30.0, 0.0, 45.0, 55.0))
    row = grad[np.argmin(np.abs(gy - 50.0))]
    peak_lon = gx[np.argmax(row)]
    assert abs(peak_lon + 15.0) < 0.6                          # on the front line
    assert 3.5 < row.max() < 5.5                               # about 5 K per 100 km, smoothing lowers it a little
    assert row[np.abs(gx + 15.0) > 8.0].max() < 0.2            # flat far from it


def test_subset_box_widens_longitude_with_latitude():
    lat, lon = _lattice()
    sub = scale_maps.subset_box(lat, lon, (-20.0, -10.0, 45.0, 50.0), 5.0)
    assert lat[sub].min() >= 40.0 - 1e-9 and lat[sub].max() <= 55.0 + 1e-9
    assert lon[sub].min() < -20.0 - 5.0                        # longitude margin > 5 degrees at 55 N
    with pytest.raises(SystemExit):
        scale_maps.subset_box(lat, lon, (170.0, -170.0, 45.0, 50.0), 5.0)


def _write(path: Path, *, with_interp: bool = True, n_members: int = 2) -> None:
    rng = np.random.default_rng(1)
    lat, lon = _lattice(0.25)
    n = lat.size
    x = _km_x(lat, lon)
    weather = np.array(["10u", "10v", "2t", "msl", "t_850"], dtype=object)
    inp = np.stack([5 * np.tanh(x / 150.0), 10 + 0 * x, 280 - 5 * np.tanh(x / 150.0),
                    100500.0 + 800 * _large(lat, lon), 275 - 5 * np.tanh(x / 150.0)], axis=-1)
    shift = np.stack([2 * _large(lat, lon), 0 * x, 0 * x, 400 * _large(lat, lon), 0 * x], axis=-1)
    truth = inp + shift + 0.5 * rng.standard_normal(inp.shape)
    members = np.stack([inp + 0.5 * shift + 0.5 * rng.standard_normal(inp.shape) for _ in range(n_members)])
    data = {
        "y_pred": (("sample", "ensemble_member", "grid_point_hres", "weather_state"), members[None]),
        "y": (("sample", "ensemble_member", "grid_point_hres", "weather_state"),
              np.broadcast_to(truth, (1, n_members, n, 5)).copy()),
        "lat_hres": (("grid_point_hres",), lat),
        "lon_hres": (("grid_point_hres",), lon),
    }
    coords = {"sample": [0], "ensemble_member": np.arange(1, n_members + 1), "grid_point_hres": np.arange(n),
              "weather_state": weather}
    if with_interp:
        data["x_interp"] = (("sample", "grid_point_hres", "weather_state"), inp[None])
    else:
        keep = np.arange(0, n, 4)
        data["x"] = (("sample", "ensemble_member", "grid_point_lres", "weather_state"),
                     np.broadcast_to(inp[keep], (1, n_members, keep.size, 5)).copy())
        data["lat_lres"] = (("grid_point_lres",), lat[keep])
        data["lon_lres"] = (("grid_point_lres",), lon[keep])
        coords["grid_point_lres"] = np.arange(keep.size)
    xr.Dataset(data, coords=coords).to_netcdf(path)


@pytest.fixture
def no_geography(monkeypatch):
    pytest.importorskip("cartopy")
    import eval.plotting as plotting
    monkeypatch.setattr(plotting, "add_geography", lambda *a, **k: None)


@pytest.mark.parametrize("with_interp", [True, False])
def test_scales_layout_writes_figures_and_stats(tmp_path, no_geography, with_interp):
    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / "predictions_20231031_step024.nc", with_interp=with_interp)
    out = tmp_path / "out"
    args = build_arg_parser().parse_args([
        "--layout", "scales", "--run", f"CA300={pred}", "--date", "20231031", "--step", "24",
        "--member", "2", "--extent", "-25", "-5", "43", "57", "--output-dir", str(out), "--region-tag", "toy"])
    assert run(args) == 0
    names = sorted(p.name for p in out.glob("*.png"))
    assert names == ["CA300_scales-full_init20231031_n002_toy_f024.png",
                     "CA300_scales-resid-10mwind_init20231031_n002_toy_f024.png",
                     "CA300_scales-resid-msl_init20231031_n002_toy_f024.png"]
    manifest = json.loads(next(out.glob("zoom_maps_manifest_scales_*.json")).read_text())
    msl = manifest["stats"]["CA300"]["msl"]
    assert manifest["scale_cut_km"] == 400.0 and "never filtered" in manifest["note"]
    assert msl["large_corr"] > 0.9 and 0.35 < msl["model_over_truth_large"] < 0.65


def test_fronts_layout_writes_one_figure(tmp_path, no_geography):
    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / "predictions_20231031_step024.nc")
    out = tmp_path / "out"
    args = build_arg_parser().parse_args([
        "--layout", "fronts", "--run", f"CA300={pred}", "--date", "20231031", "--step", "24",
        "--extent", "-25", "-5", "43", "57", "--output-dir", str(out), "--region-tag", "toy"])
    assert run(args) == 0
    assert [p.name for p in out.glob("*.png")] == ["CA300_fronts_init20231031_n001_toy_f024.png"]
    manifest = json.loads(next(out.glob("zoom_maps_manifest_fronts_*.json")).read_text())
    assert manifest["temperature"] == "t_850" and manifest["gradient_vmax_k_per_100km"] > 0


def test_unknown_scale_row_is_refused(tmp_path):
    args = build_arg_parser().parse_args([
        "--layout", "scales", "--run", f"A={tmp_path}", "--date", "20231031", "--step", "24",
        "--scale-rows", "wind10m,tp", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="unknown variable"):
        run(args)


def test_lane_runner_draws_the_layouts_once_per_region(tmp_path, no_geography):
    from eval.evaluators.zoom_maps.runner import run as runner_run

    pred = tmp_path / "pred"
    pred.mkdir()
    _write(pred / "predictions_20231031_step024.nc", with_interp=False)    # the panels layout reads x on o320
    out = runner_run(pred, {}, {"variables": ["msl"], "fields": ["value"],
                                "regions": {"toy": [43.0, 57.0, -25.0, -5.0]}}, output_dir=tmp_path / "zm")
    names = sorted(p.name for p in (out / "toy").glob("*.png"))
    assert any(n.startswith("model_scales-full") for n in names)
    assert any(n.startswith("model_fronts") for n in names)
    assert sum(n.startswith("model_scales-resid") for n in names) == 2      # wind10m and msl
