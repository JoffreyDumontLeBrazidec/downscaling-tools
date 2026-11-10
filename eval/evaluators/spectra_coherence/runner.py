"""Spectra-coherence evaluator: split the per-scale error into amplitude and phase.

The existing `spectra` evaluator compares POWER only. Power is blind to where the
energy sits: a field with the right amount of small-scale variance in the wrong
places scores as well as one that is actually correct. That is exactly the
ambiguity we hit on 10u/10v, where the fine band looks spectrally healthy while
the pointwise nMSE is ~40x worse than 2t.

For every spherical-harmonic degree l we compute, against the truth field y:

    P_true(l)  = sum_m |a_lm(y)|^2
    P_pred(l)  = sum_m |a_lm(y_pred)|^2
    X(l)       = sum_m Re[ a_lm(y_pred) conj(a_lm(y)) ]

    R(l) = sqrt( P_pred / P_true )          amplitude ratio   (1 = right amount of energy)
    C(l) = X / sqrt( P_pred * P_true )      coherence         (1 = perfectly in phase)

These two give an EXACT decomposition of the normalised per-degree error:

    E(l) = sum_m |a_pred - a_true|^2 / P_true = 1 + R^2 - 2 R C

E is minimised over amplitude at R = C, where it takes the value

    E_floor(l) = 1 - C^2

so E_floor is the best error attainable at that scale by ANY rescaling of the
prediction. If E_floor is close to 1 in the fine band, the fine-scale content is
phase-random texture and no amplitude/sharpness knob (guidance, churn, sigma
schedule) can help -- only a different model or loss can. That is the single
number this evaluator exists to produce.

The same quantities are computed for the low-resolution input, which is the honest
baseline: it tells us at which scales the model is actually adding skill over its own
input. These curves (keys ``interp_*``, ``P_interp``, ``X_interp_true``; the names are
kept so the scorer's ``coh_*_interp_coherence`` rows stay comparable) score the INPUT AS A
COMPETITOR: the coherence and amplitude ratio of the input with the truth per degree. They
are not a residual quantity; this evaluator defines no residual against x_interp. So under
rule 41 (owner, 9 Nov 2026: the input is scored on its own grid, never interpolated to the
output grid first) they come from the native input ``x`` on ``lat_lres/lon_lres`` whenever
the file carries it, sampled at the analysis-map pixel centres with the operator
box_spectra uses for its native input (linear, barycentric over the native points'
triangulation; see _input_sampler.py), not from ``x_interp``. Without a native x they fall
back to ``x_interp``, with a warning. ``native_input: false`` reproduces the old curves.
coherence.json records ``input_source``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

DEFAULT_STATES = ["10u", "10v", "2t", "msl"]

# Degree bands. o1280 is ~9 km; its spectral truncation is l=1279, so the top
# band is at/above the grid scale and is where a sampler noise floor shows up.
DEFAULT_BANDS: list[tuple[str, int, int]] = [
    ("planetary", 1, 20),
    ("synoptic", 20, 100),
    ("meso", 100, 300),
    ("fine", 300, 500),
    ("very_fine", 500, 700),
    ("near_grid", 700, 100000),
]


def _healpix_binner(lat, lon, nside):
    """Precompute the unstructured -> HEALPix scatter for one grid.

    lat/lon are identical for every weather state and every ensemble member in a
    file, so this is done once per file rather than once per field. np.bincount is
    used instead of np.add.at because the latter is an unbuffered scatter-add and
    is roughly two orders of magnitude slower on 2.6M points.
    """
    import healpy as hp

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - lat), np.deg2rad(lon), nest=False)
    npix = hp.nside2npix(nside)
    counts = np.bincount(pix, minlength=npix).astype(np.float64)
    valid = counts > 0
    coverage = float(valid.sum()) / float(npix)
    return pix, counts, valid, npix, coverage


def _healpix_map(values, pix, counts, valid, npix):
    """Bin one field onto the precomputed HEALPix grid; mean-remove over valid pixels.

    Matches eval.tools.spectra_analysis.calibrate_fast_spectra_proxy.build_healpix_mean_map
    so this evaluator sits on exactly the same support as the `spectra` evaluator.
    """
    sums = np.bincount(pix, weights=np.asarray(values, dtype=np.float64), minlength=npix)
    m = np.zeros(npix, dtype=np.float64)
    m[valid] = sums[valid] / counts[valid]
    m[valid] -= np.mean(m[valid])
    return m


def _alm(m, lmax):
    import healpy as hp
    # iter=0: no Jacobi refinement. Every map in a comparison gets identical
    # treatment, so ratios and coherences are unaffected, and it is 4x cheaper.
    return hp.map2alm(m, lmax=lmax, iter=0)


def _auto_cross(alm_a, alm_b, lmax):
    import healpy as hp
    return hp.alm2cl(alm_a, alm_b, lmax=lmax)


def _input_member(da: xr.DataArray, member_idx: int) -> np.ndarray:
    """The input of draw member_idx: its own input member, or the only one when the input has one."""
    if "ensemble_member" in da.dims and int(da.sizes["ensemble_member"]) == 1:
        member_idx = 0
    return _select_member(da, member_idx)


def _native_sampler(ds, nside: int, valid: np.ndarray, cache: dict):
    """Linear sampler of the file's native input at the centres of the valid pixels (rule 41); cached per grid."""
    import healpy as hp
    from eval.evaluators.spectra_coherence._input_sampler import SphereLinearSampler

    lat_in = np.asarray(ds["lat_lres"].values, dtype=np.float64).reshape(-1)
    lon_in = np.asarray(ds["lon_lres"].values, dtype=np.float64).reshape(-1)
    key = (nside, int(valid.sum()), lat_in.size, float(lat_in[0]), float(lat_in[-1]),
           float(lon_in[0]), float(lon_in[-1]))
    if key not in cache:
        theta, phi = hp.pix2ang(nside, np.flatnonzero(valid), nest=False)
        cache[key] = SphereLinearSampler(lat_in, lon_in, 90.0 - np.degrees(theta), np.degrees(phi))
    return cache[key]


def _sampled_map(values, sampler, valid, npix):
    """The native input on the analysis map: sampled at the valid pixels, mean-removed like _healpix_map."""
    m = np.zeros(npix, dtype=np.float64)
    m[valid] = sampler(values)
    m[valid] -= np.mean(m[valid])
    return m


INPUT_NATIVE = "x (native input grid)"
INPUT_INTERP = "x_interp (input interpolated to the output grid; no native x in the file)"
INPUT_INTERP_OFF = "x_interp (input interpolated to the output grid; native_input: false)"


def _select_member(da: xr.DataArray, member_idx: int) -> np.ndarray:
    d = da.isel(sample=0) if "sample" in da.dims else da
    if "ensemble_member" in d.dims:
        d = d.isel(ensemble_member=member_idx)
    arr = d.values.astype(np.float64)
    if d.dims and d.dims[0] == "weather_state" and arr.ndim == 2:
        arr = arr.T
    return arr


def run(
    predictions_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    run_label: str = "",
    **kwargs,
) -> Path:
    from eval.discovery.predictions import valid_prediction_files

    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir) if output_dir else predictions_dir / "evaluators" / "spectra_coherence"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default to the SAME support as the power-spectra evaluator so the two are
    # directly comparable; anything explicitly set on spectra_coherence wins.
    spectra_cfg = lane_config.get("spectra", {}) or {}
    states = eval_config.get("weather_states") or DEFAULT_STATES
    if isinstance(states, str):
        states = [s.strip() for s in states.split(",") if s.strip()]
    # NOT inherited from lane spectra.nside. The o320->o1280 lane runs power
    # spectra at nside=742, chosen so npix (6.61M) matches the o1280 point count
    # (6.60M). Matching the COUNT does not give complete coverage: the two grids
    # are quasi-uniform but not aligned, so 12.2% of HEALPix pixels receive no
    # source point and are zero-filled. That hole pattern is IDENTICAL in the
    # prediction, the truth and the interpolated driver, so its spectral leakage
    # is perfectly correlated between them and manufactures coherence at high
    # degree. Measured on this lane (2026-08-28): at nside=742 the interpolated
    # O320 driver scores C=0.59-0.70 for l in 300-1000, which is impossible --
    # O320 is truncated near l=320 and carries no information there at all. At
    # nside=512 (0.01% empty) the same baseline correctly collapses to C=0.04-0.08.
    # nside=512 caps us at l<=1024 against the O1280 truncation of 1279, which is
    # the honest trade: 80% of the resolved range, measured on a gap-free map.
    nside = int(eval_config.get("nside", 512))
    lmax = int(eval_config.get("lmax", 2 * nside))
    steps = eval_config.get("steps", spectra_cfg.get("steps"))
    max_members = eval_config.get("max_members")
    native_input = bool(eval_config.get("native_input", True))

    files = valid_prediction_files(predictions_dir, steps=[int(s) for s in steps] if steps else None)
    if not files:
        raise RuntimeError(f"No prediction files under {predictions_dir} (steps={steps})")

    LOG.info(
        "spectra_coherence: %d file(s), states=%s nside=%d lmax=%d max_members=%s",
        len(files), states, nside, lmax, max_members,
    )

    # Accumulators: sum of spectra over every (file, member) sample, per state.
    acc: dict[str, dict[str, np.ndarray]] = {}
    n_samples: dict[str, int] = {s: 0 for s in states}
    have_interp = False
    input_sources: set[str] = set()
    samplers: dict = {}

    for file_path in files:
        with xr.open_dataset(file_path) as ds:
            for required in ("y", "y_pred"):
                if required not in ds:
                    raise RuntimeError(f"{file_path} is missing {required!r}")
            ws = [str(v) for v in ds["weather_state"].values.tolist()]
            idx_of = {s: i for i, s in enumerate(ws)}
            lat = ds["lat_hres"].values
            lon = ds["lon_hres"].values
            if lat.ndim > 1:
                lat = lat[0]
            if lon.ndim > 1:
                lon = lon[0]

            pix, counts, valid, npix, cov = _healpix_binner(lat, lon, nside)

            # the input curves: native x on its own grid when the file has it (rule 41)
            file_native = native_input and "x" in ds.variables and "lat_lres" in ds.variables
            file_has_interp = file_native or "x_interp" in ds
            have_interp = have_interp or file_has_interp
            sampler = _native_sampler(ds, nside, valid, samplers) if file_native else None
            if file_native:
                input_sources.add(INPUT_NATIVE)
            elif file_has_interp:
                input_sources.add(INPUT_INTERP if native_input else INPUT_INTERP_OFF)

            n_mem = int(ds.sizes["ensemble_member"]) if "ensemble_member" in ds.dims else 1
            if max_members:
                n_mem = min(n_mem, int(max_members))

            for member_idx in range(n_mem):
                pred = _select_member(ds["y_pred"], member_idx)
                truth = _select_member(ds["y"], member_idx)
                if file_native:
                    interp = _input_member(ds["x"], member_idx)
                else:
                    interp = _select_member(ds["x_interp"], member_idx) if file_has_interp else None

                for state in states:
                    si = idx_of.get(state)
                    if si is None:
                        continue

                    m_t = _healpix_map(truth[:, si], pix, counts, valid, npix)
                    m_p = _healpix_map(pred[:, si], pix, counts, valid, npix)
                    a_t = _alm(m_t, lmax)
                    a_p = _alm(m_p, lmax)

                    p_true = _auto_cross(a_t, a_t, lmax) / max(cov, 1e-6)
                    p_pred = _auto_cross(a_p, a_p, lmax) / max(cov, 1e-6)
                    x_pt = _auto_cross(a_p, a_t, lmax) / max(cov, 1e-6)

                    entry = acc.setdefault(state, {})
                    for key, arr in (("P_true", p_true), ("P_pred", p_pred), ("X_pred_true", x_pt)):
                        entry[key] = arr if key not in entry else entry[key] + arr

                    if interp is not None:
                        m_i = (_sampled_map(interp[:, si], sampler, valid, npix) if file_native
                               else _healpix_map(interp[:, si], pix, counts, valid, npix))
                        a_i = _alm(m_i, lmax)
                        p_int = _auto_cross(a_i, a_i, lmax) / max(cov, 1e-6)
                        x_it = _auto_cross(a_i, a_t, lmax) / max(cov, 1e-6)
                        for key, arr in (("P_interp", p_int), ("X_interp_true", x_it)):
                            entry[key] = arr if key not in entry else entry[key] + arr

                    n_samples[state] += 1
        LOG.info("spectra_coherence: done %s", file_path.name)

    if INPUT_INTERP in input_sources:
        LOG.warning("spectra_coherence: input curves from x_interp (no native x in the files); rule 41: they are "
                    "then a smoothed-input diagnostic, not the input's own coherence with the truth")

    curves: dict[str, Any] = {}
    npz_payload: dict[str, np.ndarray] = {}
    for state, entry in acc.items():
        n = max(n_samples[state], 1)
        mean = {k: v / n for k, v in entry.items()}
        ell = np.arange(mean["P_true"].shape[0], dtype=np.float64)

        def _ratio_coh(p_other, x_other):
            eps = 1e-300
            R = np.sqrt(np.maximum(p_other, 0.0) / np.maximum(mean["P_true"], eps))
            C = x_other / np.sqrt(np.maximum(p_other * mean["P_true"], eps))
            C = np.clip(C, -1.0, 1.0)
            E = 1.0 + R**2 - 2.0 * R * C
            return R, C, E, 1.0 - C**2

        R, C, E, F = _ratio_coh(mean["P_pred"], mean["X_pred_true"])
        state_out = {
            "n_samples": n,
            "ell": ell.tolist(),
            "P_true": mean["P_true"].tolist(),
            "P_pred": mean["P_pred"].tolist(),
            "amplitude_ratio": R.tolist(),
            "coherence": C.tolist(),
            "normalised_error": E.tolist(),
            "error_floor_phase_only": F.tolist(),
        }
        npz_payload[f"{state}__ell"] = ell
        npz_payload[f"{state}__P_true"] = mean["P_true"]
        npz_payload[f"{state}__P_pred"] = mean["P_pred"]
        npz_payload[f"{state}__R"] = R
        npz_payload[f"{state}__C"] = C

        if "P_interp" in mean:
            Ri, Ci, Ei, Fi = _ratio_coh(mean["P_interp"], mean["X_interp_true"])
            state_out.update({
                "P_interp": mean["P_interp"].tolist(),
                "interp_amplitude_ratio": Ri.tolist(),
                "interp_coherence": Ci.tolist(),
                "interp_normalised_error": Ei.tolist(),
                "interp_error_floor_phase_only": Fi.tolist(),
            })
            npz_payload[f"{state}__P_interp"] = mean["P_interp"]
            npz_payload[f"{state}__Ri"] = Ri
            npz_payload[f"{state}__Ci"] = Ci

        curves[state] = state_out

    bands = eval_config.get("bands")
    if bands:
        bands = [(b["name"], int(b["lo"]), int(b["hi"])) for b in bands]
    else:
        bands = DEFAULT_BANDS

    # Band aggregation sums POWER and CROSS inside the band before forming the
    # ratio, which is the correct way round: a mean of per-degree ratios would be
    # dominated by the many near-empty high degrees.
    band_rows: list[dict[str, Any]] = []
    for state, entry in acc.items():
        n = max(n_samples[state], 1)
        mean = {k: v / n for k, v in entry.items()}
        ell = np.arange(mean["P_true"].shape[0])
        for name, lo, hi in bands:
            sel = (ell >= lo) & (ell < hi)
            if not np.any(sel):
                continue
            pt = float(np.sum(mean["P_true"][sel]))
            pp = float(np.sum(mean["P_pred"][sel]))
            xp = float(np.sum(mean["X_pred_true"][sel]))
            if pt <= 0:
                continue
            R = float(np.sqrt(pp / pt))
            C = float(np.clip(xp / np.sqrt(max(pp * pt, 1e-300)), -1.0, 1.0))
            row = {
                "state": state,
                "band": name,
                "ell_lo": lo,
                "ell_hi": min(hi, int(ell[-1]) + 1),
                "amplitude_ratio": R,
                "coherence": C,
                "normalised_error": 1.0 + R**2 - 2.0 * R * C,
                "error_floor_phase_only": 1.0 - C**2,
                "share_of_truth_variance": pt / float(np.sum(mean["P_true"][ell >= 1])),
            }
            if "P_interp" in mean:
                pi = float(np.sum(mean["P_interp"][sel]))
                xi = float(np.sum(mean["X_interp_true"][sel]))
                Ri = float(np.sqrt(pi / pt))
                Ci = float(np.clip(xi / np.sqrt(max(pi * pt, 1e-300)), -1.0, 1.0))
                row.update({
                    "interp_amplitude_ratio": Ri,
                    "interp_coherence": Ci,
                    "interp_normalised_error": 1.0 + Ri**2 - 2.0 * Ri * Ci,
                })
            band_rows.append(row)

    payload = {
        "run_label": run_label or predictions_dir.name,
        "predictions_dir": str(predictions_dir),
        "nside": nside,
        "lmax": lmax,
        "steps": steps,
        "states": states,
        "has_interp_baseline": have_interp,
        "native_input": native_input,
        "input_source": sorted(input_sources),
        "bands": [{"name": n_, "lo": lo, "hi": hi} for n_, lo, hi in bands],
        "band_summary": band_rows,
        "curves": curves,
    }
    (output_dir / "coherence.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(output_dir / "coherence_curves.npz", **npz_payload)
    LOG.info("spectra_coherence: wrote %s", output_dir / "coherence.json")
    return output_dir
