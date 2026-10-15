"""Synthetic prediction files in the harness schema, on real octahedral grids (CPU tests).

``octahedral_grid(N)`` returns the latitudes and longitudes of the ECMWF octahedral reduced
Gaussian grid O<N> (4 N (N + 9) points: O96 has 40,320, O320 has 421,120), north to south and
west to east inside each row, longitudes in -180..180 as the prediction files store them. Tests
of code that depends on the grid (HEALPix coverage, neighbour distances, box point counts) can
therefore run on the real O320 point set without any file from Atos.

``write_predictions_file`` writes one ``predictions_YYYYMMDD_stepNNN.nc`` with the variables
and attributes of ``manual_inference.prediction.dataset.build_predictions_dataset``:
``x`` [sample, ensemble_member, grid_point_lres, weather_state], ``y``, ``y_pred`` and
``x_interp`` [sample, ensemble_member, grid_point_hres, weather_state], ``lat/lon_hres``,
``lat/lon_lres``, ``date``, ``weather_state`` names, ``ensemble_member`` labels and the
global attributes ``grid``, ``checkpoint_id``, ``sampling_config_json``, ``lead_step_hours``.
Fields are a smooth large-scale pattern (the "input") plus a seeded small-scale part (truth:
one per date, or one per member with ``per_member_truth``; prediction: one per member).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_STATES = ("10u", "10v", "2t", "msl", "sp", "tp", "cp", "t_850", "z_500", "q_850")
_BASE = {"10u": 0.0, "10v": 0.0, "2t": 285.0, "msl": 101300.0, "sp": 98000.0, "tp": 0.0,
         "cp": 0.0, "t_850": 275.0, "z_500": 55000.0, "q_850": 0.004}
_AMP = {"10u": 8.0, "10v": 8.0, "2t": 10.0, "msl": 1500.0, "sp": 1500.0, "tp": 0.002,
        "cp": 0.001, "t_850": 8.0, "z_500": 1500.0, "q_850": 0.002}


def octahedral_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) in degrees of the octahedral grid O<n>, lon in -180..180."""
    x, _ = np.polynomial.legendre.leggauss(2 * n)
    lats = np.degrees(np.arcsin(x))[::-1]                     # north to south
    lat_rows, lon_rows = [], []
    for i in range(2 * n):
        j = i if i < n else 2 * n - 1 - i
        nlon = 20 + 4 * j
        lon = np.arange(nlon) * (360.0 / nlon)
        lat_rows.append(np.full(nlon, lats[i]))
        lon_rows.append(lon)
    lat = np.concatenate(lat_rows)
    lon = np.concatenate(lon_rows)
    return lat, ((lon + 180.0) % 360.0) - 180.0


def _pattern(lat, lon, phase: float, scale: float) -> np.ndarray:
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    k = 1.0 / scale
    return (np.sin(3 * la * k + phase) * np.cos(4 * lo * k - phase)
            + 0.5 * np.cos(7 * la * k) * np.sin(9 * lo * k + 2 * phase))


def _field(state: str, lat, lon, *, phase: float, fine: np.ndarray | None) -> np.ndarray:
    base = _pattern(lat, lon, phase, 1.0)
    v = _BASE[state] + _AMP[state] * base
    if fine is not None:
        v = v + 0.15 * _AMP[state] * fine
    if state in ("tp", "cp"):
        v = np.maximum(v - 0.5 * _AMP[state], 0.0)
        if fine is not None:
            v = v - 1e-5 * (fine < -1.2)    # a few small negative raw values, as raw tp has
    return v.astype(np.float32)


def write_predictions_file(
    path: str | Path,
    *,
    n_hres: int = 320,
    n_lres: int = 96,
    states: Sequence[str] = DEFAULT_STATES,
    members: Sequence[int] = (1, 2, 3),
    per_member_truth: bool = False,
    nan_truth_states: Sequence[str] = (),
    seed: int = 0,
    lead_step_hours: int = 60,
    grids: tuple | None = None,
) -> Path:
    """Write one synthetic prediction file and return its path.

    ``grids`` = ((lat_h, lon_h), (lat_l, lon_l)) reuses grids already built (O320 takes a
    second or two to build). ``nan_truth_states`` sets those states of ``y`` to NaN (the
    Idalia bundles carry no tp/cp truth).
    """
    import netCDF4

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if grids is None:
        grids = (octahedral_grid(n_hres), octahedral_grid(n_lres))
    (lat_h, lon_h), (lat_l, lon_l) = grids
    rng = np.random.default_rng(seed)
    nm, nh, nl, ns = len(members), lat_h.size, lat_l.size, len(states)
    phase = float(rng.uniform(0, np.pi))

    def fine_h():
        return _pattern(lat_h, lon_h, float(rng.uniform(0, np.pi)), 0.05) + 0.3 * rng.standard_normal(nh)

    x = np.empty((1, nm, nl, ns), dtype=np.float32)
    xi = np.empty((1, nm, nh, ns), dtype=np.float32)
    y = np.empty((1, nm, nh, ns), dtype=np.float32)
    yp = np.empty((1, nm, nh, ns), dtype=np.float32)
    truth_fine = fine_h()
    for k in range(nm):
        mphase = phase + (0.05 * k if per_member_truth else 0.0)
        if per_member_truth:
            truth_fine = fine_h()
        pred_fine = fine_h()
        for j, s in enumerate(states):
            x[0, k, :, j] = _field(s, lat_l, lon_l, phase=mphase, fine=None)
            xi[0, k, :, j] = _field(s, lat_h, lon_h, phase=mphase, fine=None)
            y[0, k, :, j] = np.nan if s in nan_truth_states else _field(
                s, lat_h, lon_h, phase=mphase, fine=truth_fine)
            yp[0, k, :, j] = _field(s, lat_h, lon_h, phase=mphase, fine=pred_fine)

    stem = path.name
    date = stem.split("_")[1] if stem.startswith("predictions_") else "20220819"
    with netCDF4.Dataset(str(path), "w") as d:
        d.createDimension("sample", 1)
        d.createDimension("ensemble_member", nm)
        d.createDimension("grid_point_hres", nh)
        d.createDimension("grid_point_lres", nl)
        d.createDimension("weather_state", ns)
        v = d.createVariable("weather_state", str, ("weather_state",))
        for j, s in enumerate(states):
            v[j] = s
        v = d.createVariable("ensemble_member", "i8", ("ensemble_member",))
        v[:] = np.asarray(members, dtype=np.int64)
        v = d.createVariable("sampler_seed", "i8", ("ensemble_member",))
        v[:] = 756000 + np.asarray(members, dtype=np.int64)
        for name, arr in (("lat_hres", lat_h), ("lon_hres", lon_h)):
            d.createVariable(name, "f8", ("grid_point_hres",))[:] = arr
        for name, arr in (("lat_lres", lat_l), ("lon_lres", lon_l)):
            d.createVariable(name, "f8", ("grid_point_lres",))[:] = arr
        d.createVariable("date", "i8", ("sample",))[:] = int(date)
        d.createVariable("init_date", "i8", ("sample",))[:] = int(date)
        d.createVariable("lead_step_hours", "i8", ("sample",))[:] = lead_step_hours
        dims_h = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
        dims_l = ("sample", "ensemble_member", "grid_point_lres", "weather_state")
        for name, arr, dims in (("x", x, dims_l), ("x_interp", xi, dims_h),
                                ("y", y, dims_h), ("y_pred", yp, dims_h)):
            var = d.createVariable(name, "f4", dims, zlib=True, complevel=1,
                                   chunksizes=(1, 1, arr.shape[2], 1))
            var[:] = arr
            var.setncattr("lon", "lon_lres" if name == "x" else "lon_hres")
            var.setncattr("lat", "lat_lres" if name == "x" else "lat_hres")
        d.setncattr("grid", f"O{n_lres}->O{n_hres}")
        d.setncattr("checkpoint_id", "synthetic")
        d.setncattr("sampling_config_json", json.dumps({"sampler": "synthetic", "num_steps": 1}))
    return path
