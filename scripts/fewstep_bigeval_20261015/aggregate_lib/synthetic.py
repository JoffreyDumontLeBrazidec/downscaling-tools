"""Small synthetic prediction files and evaluator outputs, for the tests and a smoke run.

The evaluator outputs follow the real writers' columns (box_spectra BIN_COLS, shape CSV_COLS,
tc_structure CASE_COLS, the texture / wind_extremes / tc JSON layouts); the prediction files
follow the harness schema (y_pred, y, x_interp [sample, ensemble_member, grid_point_hres,
weather_state], lat_hres/lon_hres, weather_state names).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

STATES = ["10u", "10v", "2t", "msl", "tp", "t_850", "z_500"]


def grid(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon = rng.uniform(-180, 180, n)
    # make sure Europe holds points
    lat[:150] = rng.uniform(36, 74, 150)
    lon[:150] = rng.uniform(-24, 44, 150)
    return lat, lon


def write_prediction(path: Path, *, lat, lon, members: int = 5, seed: int = 0, bias: float = 0.0,
                     spread: float = 1.0, truth_seed: int = 99, states=STATES) -> Path:
    """One predictions_YYYYMMDD_stepNNN.nc file."""
    import xarray as xr

    n = lat.size
    rt = np.random.default_rng(truth_seed)
    truth = np.stack([rt.normal(0, 1, n) + (280 if s == "2t" else 0) for s in states], axis=-1)
    tp_i = states.index("tp") if "tp" in states else None
    if tp_i is not None:
        truth[:, tp_i] = np.maximum(rt.gamma(0.4, 0.002, n) - 0.0003, -0.00005)  # m, a few negatives
    rng = np.random.default_rng(seed)
    pred = truth[None] + bias + spread * rng.normal(0, 1, (members, n, len(states)))
    if tp_i is not None:
        pred[:, :, tp_i] = truth[None, :, tp_i] * rng.lognormal(0, 0.5, (members, n)) \
            + rng.normal(0, 0.0002, (members, n))
    y = np.broadcast_to(truth[None], pred.shape).copy()
    xi = y + 0.3 * np.random.default_rng(truth_seed + 1).normal(0, 1, (1, n, len(states)))
    dims = ("sample", "ensemble_member", "grid_point_hres", "weather_state")
    ds = xr.Dataset(
        {"y_pred": (dims, pred[None].astype(np.float32)), "y": (dims, y[None].astype(np.float32)),
         "x_interp": (dims, xi[None].astype(np.float32)),
         "lat_hres": (("grid_point_hres",), lat), "lon_hres": (("grid_point_hres",), lon)},
        coords={"weather_state": np.array(states, dtype=object),
                "ensemble_member": np.arange(1, members + 1)})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def write_csv(path: Path, cols, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
    return path


BIN_COLS = ["scope", "variable", "bin_lo_km", "bin_hi_km", "bin_centre_km",
            "amp_resid_pred", "amp_resid_true", "ratio_resid",
            "amp_pred_full", "amp_truth_full", "ratio_full",
            "amp_interp", "ratio_interp_to_truth", "coherence", "aligned", "n_draws"]


def box_bins(dirpath: Path, *, dates, variables=("10u", "2t"), bins_km=(400.0, 200.0, 100.0, 60.0),
             ratio=None, box=(33.0, 70.0, -15.0, 40.0)) -> Path:
    """bins.csv: ``ratio(var, km, scope)`` gives (ratio_full, ratio_resid) for scope 'all' or a date."""
    ratio = ratio or (lambda v, km, s: (1.0, 1.0))
    rows = []
    for scope in ["all"] + list(dates):
        for v in variables:
            for km in bins_km:
                rf, rr = ratio(v, km, scope)
                coh = 0.8
                rows.append({"scope": scope, "variable": v, "bin_lo_km": km * 0.9, "bin_hi_km": km * 1.1,
                             "bin_centre_km": km, "amp_resid_pred": rr, "amp_resid_true": 1.0,
                             "ratio_resid": rr, "amp_pred_full": rf, "amp_truth_full": 1.0,
                             "ratio_full": rf, "amp_interp": 0.9, "ratio_interp_to_truth": 0.9,
                             "coherence": coh, "aligned": coh * rf, "n_draws": 10})
    write_csv(dirpath / "bins.csv", BIN_COLS, rows)
    write_json(dirpath / "run_meta.json", {"box": list(box), "box_chosen_by": "given"})
    return dirpath


def coherence_json(dirpath: Path, *, lmax: int = 200, amp: float = 1.0, coh: float = 0.9,
                   states=("10u", "2t")) -> Path:
    ell = np.arange(lmax + 1, dtype=float)
    pt = np.where(ell > 0, 1.0 / np.maximum(ell, 1) ** 2, 0.0)
    curves = {}
    for s in states:
        pp = pt * amp ** 2
        curves[s] = {"n_samples": 10, "ell": ell.tolist(), "P_true": pt.tolist(), "P_pred": pp.tolist(),
                     "amplitude_ratio": np.full(ell.size, amp).tolist(),
                     "coherence": np.full(ell.size, coh).tolist(),
                     "P_interp": (pt * 0.81).tolist(), "interp_amplitude_ratio": np.full(ell.size, 0.9).tolist(),
                     "interp_coherence": np.full(ell.size, 0.95).tolist()}
    return write_json(dirpath / "coherence.json", {"curves": curves, "lmax": lmax})
