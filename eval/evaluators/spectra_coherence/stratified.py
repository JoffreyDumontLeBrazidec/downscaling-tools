"""Surface-type stratification of the amplitude/phase diagnostic.

Why this is not just "mask the region and run a spherical-harmonic transform":
masking BEFORE the transform is exactly the mistake that made nside=742 invent
coherence -- a shared hole pattern leaks correlated power into every degree. So
the order is inverted here. The transform is always done on the FULL sphere, the
band-pass is applied in spectral space, and only the band-passed MAPS are then
masked and correlated in grid space. No mask ever touches a transform.

For each band and each surface class we report, over the pixels of that class,

    C_region = sum(p*t) / sqrt(sum(p^2) sum(t^2))     phase agreement, uncentred
    R_region = sqrt( sum(p^2) / sum(t^2) )            amplitude ratio

using the uncentred inner product because a band-passed field has no meaningful
regional mean to subtract. Over the whole sphere these reduce exactly to the
global coherence and amplitude ratio, so the "all" row is a built-in consistency
check against coherence.json.

Surface classes come from the O1280 constant forcings (lsm, z), which are the very
fields the model does and does not receive -- the point of the test.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

FORCINGS_NPZ = "/home/ecm5702/hpcperm/data/o1280-forcings.npz"
G = 9.80665  # m s^-2, to turn surface geopotential into metres

# Terrain-variability class edges in metres of orographic standard deviation,
# measured inside ~0.5 degree cells (nside 128), which is the scale at which
# sub-grid orographic drag is parameterised.
FLAT_MAX_M = 50.0
COMPLEX_MIN_M = 200.0

STRAT_BANDS: list[tuple[str, int, int]] = [
    ("synoptic", 20, 100),
    ("meso", 100, 300),
    ("fine", 300, 500),
    ("very_fine", 500, 700),
    ("near_grid", 700, 100000),
]


def build_surface_classes(lat, lon, nside, *, forcings_npz=FORCINGS_NPZ):
    """Label every HEALPix pixel at `nside` by surface type.

    Returns {class_name: boolean pixel mask} plus a small diagnostic dict.
    """
    import healpy as hp

    z = np.load(forcings_npz)
    lsm_pts = np.asarray(z["lsm"], dtype=np.float64)
    orog_pts = np.asarray(z["z"], dtype=np.float64) / G

    if lsm_pts.size != np.asarray(lat).size:
        raise RuntimeError(
            "forcings/prediction grid size mismatch: forcings=%d predictions=%d"
            % (lsm_pts.size, np.asarray(lat).size)
        )

    # Gate: the forcings file carries no lat/lon, only their trig pair, so the
    # ordering against the prediction grid must be proven, never assumed.
    lat_f = np.degrees(np.arctan2(z["sin_latitude"], z["cos_latitude"]))
    lon_f = np.degrees(np.arctan2(z["sin_longitude"], z["cos_longitude"]))
    dlat = float(np.max(np.abs(lat_f - np.asarray(lat))))
    dlon = float(np.max(np.abs((np.mod(lon_f - np.asarray(lon) + 180.0, 360.0)) - 180.0)))
    if dlat > 1e-3 or dlon > 1e-3:
        raise RuntimeError(f"grid order gate FAILED: max dlat={dlat} dlon={dlon} deg")
    LOG.info("surface classes: grid order gate PASSED (dlat=%.2e dlon=%.2e deg)", dlat, dlon)

    theta = np.deg2rad(90.0 - np.asarray(lat))
    phi = np.deg2rad(np.mod(np.asarray(lon), 360.0))

    # Land-sea fraction on the analysis map.
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    npix = hp.nside2npix(nside)
    cnt = np.bincount(pix, minlength=npix).astype(np.float64)
    lsm_map = np.divide(np.bincount(pix, weights=lsm_pts, minlength=npix), cnt,
                        out=np.zeros(npix), where=cnt > 0)

    # Orographic variability at ~0.5 deg (nside 128), then upsampled. That cell
    # holds ~130 O1280 points, enough for a meaningful standard deviation; the
    # analysis map itself has only ~2 points per pixel and cannot supply one.
    ns_c = 128
    pix_c = hp.ang2pix(ns_c, theta, phi, nest=False)
    npix_c = hp.nside2npix(ns_c)
    cnt_c = np.bincount(pix_c, minlength=npix_c).astype(np.float64)
    s1 = np.bincount(pix_c, weights=orog_pts, minlength=npix_c)
    s2 = np.bincount(pix_c, weights=orog_pts**2, minlength=npix_c)
    ok = cnt_c > 1
    mean_c = np.divide(s1, cnt_c, out=np.zeros(npix_c), where=ok)
    var_c = np.divide(s2, cnt_c, out=np.zeros(npix_c), where=ok) - mean_c**2
    std_c = np.sqrt(np.maximum(var_c, 0.0))
    std_map = hp.ud_grade(std_c, nside_out=nside, order_in="RING", order_out="RING")

    ocean = lsm_map < 0.2
    coast = (lsm_map >= 0.2) & (lsm_map <= 0.8)
    land = lsm_map > 0.8
    flat_land = land & (std_map < FLAT_MAX_M)
    complex_land = land & (std_map > COMPLEX_MIN_M)
    mid_land = land & ~flat_land & ~complex_land

    classes = {
        "all": np.ones(npix, dtype=bool),
        "ocean": ocean,
        "coast": coast,
        "flat_land": flat_land,
        "rolling_land": mid_land,
        "complex_land": complex_land,
    }
    info = {
        name: {
            "pixel_fraction": float(np.mean(m)),
            "median_orog_std_m": float(np.median(std_map[m])) if np.any(m) else None,
        }
        for name, m in classes.items()
    }
    return classes, info


def _band_map(alm, lo, hi, lmax, nside):
    """Zero the alm outside [lo, hi) and transform back to a map."""
    import healpy as hp

    ell, _ = hp.Alm.getlm(lmax)
    keep = (ell >= lo) & (ell < hi)
    a = np.where(keep, alm, 0.0)
    return hp.alm2map(a, nside, lmax=lmax)


def run_stratified(
    predictions_dir,
    *,
    output_dir,
    states,
    nside,
    lmax,
    steps,
    max_members=None,
    run_label="",
    bands=None,
):
    from eval.evaluators.spectra.proxy_runner import valid_prediction_files
    from eval.evaluators.spectra_coherence.runner import (
        _alm, _healpix_binner, _healpix_map, _select_member,
    )

    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bands = bands or STRAT_BANDS

    files = valid_prediction_files(predictions_dir, steps=[int(s) for s in steps] if steps else None)

    classes = None
    class_info = None
    # acc[(state, band, cls)] = [sum p*t, sum p^2, sum t^2, sum i*t, sum i^2]
    acc: dict[tuple, np.ndarray] = {}
    n_samples = 0

    for file_path in files:
        with xr.open_dataset(file_path) as ds:
            ws = [str(v) for v in ds["weather_state"].values.tolist()]
            idx_of = {s: i for i, s in enumerate(ws)}
            lat = ds["lat_hres"].values
            lon = ds["lon_hres"].values
            if lat.ndim > 1:
                lat = lat[0]
            if lon.ndim > 1:
                lon = lon[0]

            if classes is None:
                classes, class_info = build_surface_classes(lat, lon, nside)

            pix, counts, valid, npix, cov = _healpix_binner(lat, lon, nside)
            has_interp = "x_interp" in ds
            n_mem = int(ds.sizes["ensemble_member"]) if "ensemble_member" in ds.dims else 1
            if max_members:
                n_mem = min(n_mem, int(max_members))

            for member_idx in range(n_mem):
                pred = _select_member(ds["y_pred"], member_idx)
                truth = _select_member(ds["y"], member_idx)
                interp = _select_member(ds["x_interp"], member_idx) if has_interp else None

                for state in states:
                    si = idx_of.get(state)
                    if si is None:
                        continue
                    a_p = _alm(_healpix_map(pred[:, si], pix, counts, valid, npix), lmax)
                    a_t = _alm(_healpix_map(truth[:, si], pix, counts, valid, npix), lmax)
                    a_i = (_alm(_healpix_map(interp[:, si], pix, counts, valid, npix), lmax)
                           if interp is not None else None)

                    for bname, lo, hi in bands:
                        mp = _band_map(a_p, lo, hi, lmax, nside)
                        mt = _band_map(a_t, lo, hi, lmax, nside)
                        mi = _band_map(a_i, lo, hi, lmax, nside) if a_i is not None else None
                        for cls, mask in classes.items():
                            m = mask & valid
                            p = mp[m]
                            t = mt[m]
                            vals = np.array([
                                float(np.dot(p, t)), float(np.dot(p, p)), float(np.dot(t, t)),
                                float(np.dot(mi[m], t)) if mi is not None else 0.0,
                                float(np.dot(mi[m], mi[m])) if mi is not None else 0.0,
                            ])
                            key = (state, bname, cls)
                            acc[key] = vals if key not in acc else acc[key] + vals
                n_samples += 1
        LOG.info("spectra_coherence/stratified: done %s", file_path.name)

    rows = []
    for (state, bname, cls), v in acc.items():
        pt, pp, tt, it, ii = v
        if tt <= 0 or pp <= 0:
            continue
        row = {
            "state": state, "band": bname, "surface_class": cls,
            "correlation": float(np.clip(pt / np.sqrt(pp * tt), -1.0, 1.0)),
            "amplitude_ratio": float(np.sqrt(pp / tt)),
            "truth_band_rms": float(np.sqrt(tt)),
        }
        if ii > 0:
            row["interp_correlation"] = float(np.clip(it / np.sqrt(ii * tt), -1.0, 1.0))
            row["interp_amplitude_ratio"] = float(np.sqrt(ii / tt))
        rows.append(row)

    payload = {
        "run_label": run_label,
        "predictions_dir": str(predictions_dir),
        "nside": nside, "lmax": lmax, "steps": steps,
        "n_member_files": n_samples,
        "surface_classes": class_info,
        "class_definition": {
            "ocean": "land-sea fraction < 0.2",
            "coast": "0.2 <= land-sea fraction <= 0.8",
            "flat_land": "land-sea fraction > 0.8 and orographic std < %g m" % FLAT_MAX_M,
            "rolling_land": "land, orographic std between %g and %g m" % (FLAT_MAX_M, COMPLEX_MIN_M),
            "complex_land": "land-sea fraction > 0.8 and orographic std > %g m" % COMPLEX_MIN_M,
            "orographic_std": "std of O1280 surface geopotential/g inside ~0.5 deg (nside 128) cells",
        },
        "bands": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in bands],
        "rows": rows,
    }
    out = output_dir / "coherence_by_surface.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    LOG.info("spectra_coherence/stratified: wrote %s", out)
    return output_dir
