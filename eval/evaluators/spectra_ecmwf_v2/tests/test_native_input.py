"""Rule 41 (owner, 9 Nov 2026): the input spectrum comes from the native input x on its own grid.

Tiny synthetic prediction files; the gptosp stages are mocked, so these run anywhere.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eval.evaluators.spectra_ecmwf_v2 import runner

N_LRES = 8                       # O8: 4 * 8^2 + 36 * 8 = 544 points, TCo7
N_POINTS = 4 * N_LRES**2 + 36 * N_LRES


def _write_predictions(pred_dir: Path, *, native: bool, date="20230826", step=120) -> Path:
    import netCDF4

    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"predictions_{date}_step{step:03d}.nc"
    rng = np.random.default_rng(0)
    n_h, n_m, states = 200, 2, ["2t", "10u"]
    with netCDF4.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", n_m)
        d.createDimension("grid_point_hres", n_h)
        d.createDimension("weather_state", len(states))
        d.createVariable("lat_hres", "f8", ("grid_point_hres",))[:] = np.linspace(-80, 80, n_h)
        d.createVariable("lon_hres", "f8", ("grid_point_hres",))[:] = np.linspace(0, 359, n_h)
        v = d.createVariable("weather_state", str, ("weather_state",))
        for i, s in enumerate(states):
            v[i] = s
        d.createVariable("ensemble_member", "i4", ("ensemble_member",))[:] = np.arange(1, n_m + 1)
        for name in ("y", "x_interp"):
            d.createVariable(name, "f4", ("sample", "grid_point_hres", "weather_state"))[:] = \
                rng.standard_normal((1, n_h, len(states)))
        d.createVariable("y_pred", "f4", ("sample", "ensemble_member", "grid_point_hres", "weather_state"))[:] = \
            rng.standard_normal((1, n_m, n_h, len(states)))
        if native:
            d.createDimension("grid_point_lres", N_POINTS)
            d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = np.linspace(-89, 89, N_POINTS)
            d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = np.linspace(0, 359, N_POINTS)
            d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))[:] = \
                rng.standard_normal((1, n_m, N_POINTS, len(states)))
    return path


def _run(tmp_path: Path, eval_extra: dict | None = None):
    pred = tmp_path / "predictions"
    out = tmp_path / "spectra_ecmwf"
    lane = {"predict": {"dates": ["20230826"], "steps": [120]}}
    ev = {"weather_states": ["2t", "10u"], "members": [1, 2], "truncation": 31,
          "reference_dir": str(tmp_path / "ref"), **(eval_extra or {})}
    with (
        patch.object(runner.socket, "gethostname", return_value="ac-test"),
        patch.object(runner, "_run_pipeline") as pipe,
        patch.object(runner, "_run_input_bundle_pipeline") as native_pipe,
    ):
        runner.run(pred, lane, ev, output_dir=out)
    return out, pipe, native_pipe


def test_input_spectrum_from_the_native_grid_at_its_own_truncation(tmp_path: Path) -> None:
    _write_predictions(tmp_path / "predictions", native=True)
    out, pipe, native_pipe = _run(tmp_path)

    # the model and the truth go through the output-grid pipeline; x_interp is never staged
    assert [c.kwargs["prediction_var"] for c in pipe.call_args_list] == ["y_pred", "y"]
    assert native_pipe.call_count == 1
    kw = native_pipe.call_args.kwargs
    assert kw["predictions_dir"] == (tmp_path / "predictions").resolve()
    assert kw["truncation"] == N_LRES - 1                 # TCo7, the input's own truncation, not T31
    assert kw["template_grib"].endswith("o8-template.grib")
    assert "/input_native/" in str(kw["output_dir"])
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["input_source"] == ["x (native input grid)"]
    assert meta["input_grid"] == "O8" and meta["input_truncation"] == N_LRES - 1
    assert meta["input_reference_dir"] == str(kw["output_dir"] / "spectra")


def test_without_native_x_the_input_falls_back_to_x_interp_with_a_warning(tmp_path: Path, caplog) -> None:
    _write_predictions(tmp_path / "predictions", native=False)
    with caplog.at_level(logging.WARNING):
        out, pipe, native_pipe = _run(tmp_path)

    assert [c.kwargs["prediction_var"] for c in pipe.call_args_list] == ["y_pred", "y", "x_interp"]
    assert native_pipe.call_count == 0
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["input_source"] == ["x_interp (input interpolated to the output grid; no native x in the file)"]
    assert "/input/" in meta["input_reference_dir"]
    assert any("rule 41" in r.getMessage() for r in caplog.records)


def test_native_input_false_reproduces_the_x_interp_curve(tmp_path: Path) -> None:
    _write_predictions(tmp_path / "predictions", native=True)
    out, pipe, native_pipe = _run(tmp_path, {"native_input": False})

    assert [c.kwargs["prediction_var"] for c in pipe.call_args_list] == ["y_pred", "y", "x_interp"]
    assert native_pipe.call_count == 0
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["input_source"] == ["x_interp (input interpolated to the output grid; native_input: false)"]


def test_native_grid_is_read_off_the_point_count() -> None:
    assert runner._octahedral_n(421120) == 320
    assert runner._octahedral_n(40320) == 96
    assert runner._native_input_grid({}, 421120) == ("O320", 319)
    assert runner._native_input_grid({"prepml": {"input": {"grid": "o1280"}}}, 4 * 1280**2 + 36 * 1280) == \
        ("O1280", 1279)
    with pytest.raises(ValueError, match="no full octahedral"):
        runner._native_input_grid({}, 55871)              # a regional (box) input
    with pytest.raises(ValueError, match="prepml.input.grid"):
        runner._native_input_grid({"prepml": {"input": {"grid": "O96"}}}, 421120)


def test_native_point_order_never_interpolates() -> None:
    from eval.evaluators.spectra_ecmwf_v2._input_bundle_stager import native_point_order

    lat = np.repeat(np.linspace(-80, 80, 20), 10)
    lon = np.tile(np.linspace(-170, 170, 10), 20)
    assert native_point_order(lat, lon, lat, lon) is None
    perm = np.random.default_rng(1).permutation(lat.size)
    idx = native_point_order(lat[perm], np.mod(lon[perm], 360.0), lat, lon)
    np.testing.assert_array_equal(lat[perm][idx], lat)
    with pytest.raises(ValueError, match="not a permutation"):
        native_point_order(lat + 0.3, lon, lat, lon)
    with pytest.raises(ValueError, match="not on the template grid"):
        native_point_order(lat[:-1], lon[:-1], lat, lon)


O96_TEMPLATE = Path("/home/ecm5702/hpcperm/data/o96-template.grib")


@pytest.mark.skipif(not O96_TEMPLATE.exists(), reason="needs the O96 template GRIB on hpc-login")
def test_stager_writes_the_native_x_on_its_own_grid(tmp_path: Path) -> None:
    """x stored in a shuffled point order comes out on the O96 template in template order, unchanged."""
    import netCDF4
    from eccodes import codes_get_values, codes_grib_new_from_file, codes_release

    from eval.evaluators.spectra_ecmwf_v2 import _input_bundle_stager as st

    tlat, tlon = st.load_template_grib(O96_TEMPLATE)
    perm = np.random.default_rng(2).permutation(tlat.size)
    pred = tmp_path / "pred"
    path = _write_predictions(pred, native=False)
    field = np.cos(np.radians(tlat)) * 30.0 + 250.0
    with netCDF4.Dataset(str(path), "a") as d:
        d.createDimension("grid_point_lres", tlat.size)
        d.createVariable("lat_lres", "f8", ("grid_point_lres",))[:] = tlat[perm]
        d.createVariable("lon_lres", "f8", ("grid_point_lres",))[:] = np.mod(tlon[perm], 360.0)
        x = d.createVariable("x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))
        x[0, :, :, 0] = np.stack([field[perm], field[perm] + 1.0])
        x[0, :, :, 1] = 0.0
    summary = st.stage_native_predictions(pred, tmp_path / "grb", O96_TEMPLATE, ["2t"])
    assert summary["interpolated_to_template"] is False and summary["point_order"] == "exact permutation"
    assert summary["grib_files_written"] == 2
    with open(tmp_path / "grb" / "2t_sfc" / "1_20230826_120_2_nopoles.grb", "rb") as fh:
        gid = codes_grib_new_from_file(fh)
        vals = np.asarray(codes_get_values(gid))
        codes_release(gid)
    np.testing.assert_allclose(vals, field.astype(np.float32) + 1.0, atol=0.05)   # GRIB packing precision


def test_plotter_reads_the_recorded_native_input_curve(tmp_path: Path) -> None:
    """The standard figure and the ratio PDF draw an input curve that ends at the input's truncation."""
    from eval.evaluators.spectra_ecmwf_v2 import plotter

    def curves(base: Path, j: int, scale: float, n: int = 2) -> None:
        d = base / "2t_sfc"
        d.mkdir(parents=True, exist_ok=True)
        wvn = np.arange(1, j + 1, dtype=float)
        for i in range(n):
            np.save(d / f"ampl_{i:03d}.npy", scale * wvn ** -1.5)
            np.save(d / f"wvn_{i:03d}.npy", wvn)

    results = tmp_path / "run"
    truth = tmp_path / "ref" / "truth" / "dKEY_T31" / "spectra"
    native = tmp_path / "ref" / "input_native" / "dKEY_T7" / "spectra"
    curves(results / "spectra", 31, 1.0)
    curves(truth, 31, 1.0)
    curves(native, 7, 0.8)
    (results / "spectra_summary.json").write_text(json.dumps({"reference_spectra_dir": str(truth)}))
    (results / "run_meta.json").write_text(json.dumps({"input_reference_dir": str(native),
                                                       "input_source": ["x (native input grid)"]}))
    with patch("eval.evaluators.spectra_ecmwf_v2._plotter.build_pdf_ecmwf_with_references") as mock_build:
        mock_build.return_value = 1
        plotter.plot(results, {}, {})
    assert mock_build.call_args.kwargs["input_amp_dir"] == native
    plotter.plot(results, {}, {})           # real drawing with a T7 input beside T31 truth and model
    assert (results / "spectra_ecmwf_ratio.pdf").exists()
    assert list(results.glob("spectra_ecmwf_grid.*"))
