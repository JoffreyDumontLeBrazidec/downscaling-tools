"""spectra_coherence: residual ratio, aligned part, curves.csv and nside auto (o96->o320 big eval)."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("xarray")

from eval.evaluators.spectra_coherence import runner as R  # noqa: E402
from eval.tests.synthetic_predictions import octahedral_grid, write_predictions_file  # noqa: E402


def _files(tmp_path, n_hres=48, n_lres=16):
    grids = (octahedral_grid(n_hres), octahedral_grid(n_lres))
    d = tmp_path / "predictions"
    for i, date in enumerate(("20220101", "20220819")):
        write_predictions_file(d / f"predictions_{date}_step060.nc", grids=grids,
                               states=("10u", "2t", "tp"), members=(1, 2), seed=i)
    return d


def test_residual_columns_and_per_date_rows(tmp_path):
    d = _files(tmp_path)
    out = tmp_path / "out"
    R.run(d, {}, {"weather_states": ["10u", "2t"], "nside": "auto", "steps": [60],
                  "per_date": True}, output_dir=out)
    payload = json.loads((out / "coherence.json").read_text())
    assert payload["nside"] == 16          # O48 fills nside 16 completely, not 32
    assert payload["healpix_coverage"] >= 0.999
    c = payload["curves"]["10u"]
    for key in ("residual_amplitude_ratio", "residual_coherence", "aligned"):
        assert len(c[key]) == len(c["ell"])
    np.testing.assert_allclose(np.asarray(c["aligned"]),
                               np.asarray(c["coherence"]) * np.asarray(c["amplitude_ratio"]))
    with open(out / "curves.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert set(rows[0]) == set(R.CURVE_COLS)
    scopes = {r["scope"] for r in rows}
    assert scopes == {"all", "20220101", "20220819"}
    allrows = [r for r in rows if r["scope"] == "all" and r["state"] == "10u"]
    assert int(allrows[0]["n_samples"]) == 4          # 2 files x 2 members
    assert float(allrows[0]["wavelength_km"]) == pytest.approx(2 * np.pi * 6371.0088 / np.sqrt(2), rel=1e-4)


def test_residual_ratio_is_the_spectrum_of_the_difference(tmp_path):
    """sum_m |a(pred) - a(interp)|^2 equals the spectrum of the map pred - interp (linearity)."""
    import healpy as hp
    import netCDF4

    d = _files(tmp_path)
    out = tmp_path / "out"
    R.run(d, {}, {"weather_states": ["2t"], "nside": 16, "lmax": 32, "steps": [60]},
          output_dir=out)
    rows = [r for r in csv.DictReader(open(out / "curves.csv")) if r["state"] == "2t"]
    # direct computation on the difference fields
    p_rp = p_rt = 0.0
    n = 0
    for f in sorted(d.glob("*.nc")):
        with netCDF4.Dataset(f) as ds:
            lat, lon = ds["lat_hres"][:], ds["lon_hres"][:]
            j = list(ds["weather_state"][:]).index("2t")
            pix, counts, valid, npix, cov = R._healpix_binner(lat, lon, 16)
            for k in range(2):
                yp, y, xi = (np.asarray(ds[v][0, k, :, j], dtype=np.float64)
                             for v in ("y_pred", "y", "x_interp"))
                mp = R._healpix_map(yp - xi, pix, counts, valid, npix)
                mt = R._healpix_map(y - xi, pix, counts, valid, npix)
                p_rp = p_rp + hp.alm2cl(R._alm(mp, 32), lmax=32) / cov
                p_rt = p_rt + hp.alm2cl(R._alm(mt, 32), lmax=32) / cov
                n += 1
    want = np.sqrt((p_rp / n) / (p_rt / n))
    got = np.array([float(r["residual_amplitude_ratio"]) for r in rows])
    np.testing.assert_allclose(got[:10], want[1:11], rtol=1e-5)  # csv holds 6 digits


def test_auto_nside_on_the_real_o320_grid():
    lat, lon = octahedral_grid(320)
    assert lat.size == 421120
    assert R.auto_nside(lat, lon) == 128
    # the default nside 512 leaves most of the map empty on O320
    assert R._healpix_binner(lat, lon, 512)[4] < 0.2
