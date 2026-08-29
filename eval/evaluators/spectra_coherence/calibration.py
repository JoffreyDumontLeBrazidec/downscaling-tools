"""Is the model's own uncertainty consistent with the skill it achieves?

A diffusion model is supposed to look wrong on unpredictable scales: it invents a
plausible detail instead of a blurred average, so correlating one sample against
one truth penalises it twice. That means a low correlation is NOT by itself
evidence of a defect, and the earlier coherence numbers cannot distinguish
"the model is wrong" from "the atmosphere is unpredictable here".

This does distinguish them, using only draws from the SAME input.

Write the truth as y = m + e, where m is the predictable part given the coarse
input and e is what the input cannot determine. A correct sample is m + e', with
e' an independent draw of the same statistics. Then, per spherical-harmonic degree,

    f = Var(m) / (Var(m) + Var(e))          the predictable fraction

and a single correct sample correlates with the truth at exactly f, while the mean
of N samples correlates at Var(m)/sqrt((Var(m)+Var(e)/N)(Var(m)+Var(e))), which
tends to sqrt(f).

Crucially f can be estimated from the draws ALONE, with no truth involved: the
scatter between draws is Var(e), and the variance of their mean is
Var(m) + Var(e)/N. So we get the model's OWN claim about what is predictable, and
can then ask whether reality agrees:

    C_single ~= f_model   -> honest. The low correlation is real unpredictability
                             and there is nothing here to fix.
    C_single <  f_model   -> OVERCONFIDENT. The draws agree with each other on
                             something that is not true. A genuine defect that the
                             double-penalty argument does not excuse.
    C_single >  f_model   -> OVER-SPREAD. The draws disagree more than they need
                             to; skill is being thrown away as excess randomness.

Caveat recorded honestly: on this lane the truth is an independent ENFO forecast
rather than the paired outcome of the EEFO input, so part of what lands in Var(e)
is forecast divergence rather than model uncertainty. That inflates the gap
between f_model and C_single in the OVERCONFIDENT direction, so a finding of
over-spread is safe, while a finding of overconfidence needs care.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

BANDS = [
    ("planetary", 1, 20),
    ("synoptic", 20, 100),
    ("meso", 100, 300),
    ("fine", 300, 500),
    ("very_fine", 500, 700),
    ("near_grid", 700, 100000),
]


def run_calibration(draw_dirs, *, output_dir, states, nside=512, lmax=1024,
                    step=120, member=0, run_label=""):
    import healpy as hp
    from eval.evaluators.spectra_coherence.runner import (
        _alm, _healpix_binner, _healpix_map, _select_member,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alms = {s: [] for s in states}
    alm_truth = {}
    pix = counts = valid = npix = None

    for d in draw_dirs:
        f = sorted(Path(d).glob("predictions/predictions_*_step%03d.nc" % step))
        if not f:
            LOG.warning("no prediction in %s", d)
            continue
        with xr.open_dataset(f[0]) as ds:
            ws = [str(v) for v in ds["weather_state"].values.tolist()]
            idx_of = {s: i for i, s in enumerate(ws)}
            lat = ds["lat_hres"].values
            lon = ds["lon_hres"].values
            if lat.ndim > 1:
                lat = lat[0]
            if lon.ndim > 1:
                lon = lon[0]
            if pix is None:
                pix, counts, valid, npix, cov = _healpix_binner(lat, lon, nside)
            pred = _select_member(ds["y_pred"], member)
            truth = _select_member(ds["y"], member)
            for s in states:
                si = idx_of.get(s)
                if si is None:
                    continue
                alms[s].append(_alm(_healpix_map(pred[:, si], pix, counts, valid, npix), lmax))
                if s not in alm_truth:
                    alm_truth[s] = _alm(_healpix_map(truth[:, si], pix, counts, valid, npix), lmax)
        LOG.info("loaded %s", d)

    ell_of_alm, _ = hp.Alm.getlm(lmax)
    rows = []
    for s in states:
        A = np.array(alms[s])            # (N, n_alm) complex
        N = A.shape[0]
        if N < 3:
            continue
        T = alm_truth[s]
        abar = A.mean(axis=0)
        dev = A - abar                   # deviations from the draw-mean

        for name, lo, hi in BANDS:
            sel = (ell_of_alm >= lo) & (ell_of_alm < hi)
            if not np.any(sel):
                continue
            # alm2cl-style power: m=0 terms count once, m>0 twice. Using a plain
            # squared modulus sum is proportional to that for a fixed band and
            # cancels in every ratio below, so it is used directly.
            def pw(x):
                return float(np.sum(np.abs(x[sel]) ** 2))

            P_t = pw(T)
            P_bar = pw(abar)
            spread = float(np.mean([pw(dev[i]) for i in range(N)])) * N / (N - 1.0)
            var_m = max(P_bar - spread / N, 0.0)
            f_model = var_m / max(var_m + spread, 1e-300)

            c_single = float(np.mean([
                np.sum(np.real(A[i][sel] * np.conj(T[sel])))
                / np.sqrt(max(pw(A[i]) * P_t, 1e-300)) for i in range(N)
            ]))
            c_mean = float(np.sum(np.real(abar[sel] * np.conj(T[sel])))
                           / np.sqrt(max(P_bar * P_t, 1e-300)))
            c_mean_expected = var_m / np.sqrt(
                max((var_m + spread / N) * (var_m + spread), 1e-300))

            rows.append({
                "state": s, "band": name, "n_draws": N,
                "f_model": f_model,
                "C_single": c_single,
                "C_mean": c_mean,
                "C_mean_expected_if_calibrated": float(c_mean_expected),
                "spread_over_total": float(spread / max(P_bar + spread * (1 - 1.0 / N), 1e-300)),
                "amplitude_ratio_single": float(np.sqrt(
                    np.mean([pw(A[i]) for i in range(N)]) / max(P_t, 1e-300))),
            })

    payload = {"run_label": run_label, "n_draws": N, "nside": nside, "lmax": lmax,
               "step": step, "states": states, "rows": rows}
    (output_dir / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    LOG.info("wrote %s", output_dir / "calibration.json")
    return output_dir
