# PORT NOTE (2026-09-23, eval/evaluators/shape): verbatim copy of
# PORT /home/ecm5702/agent-work/20260917-shape-probes/scripts/shape_fullrung.py
# PORT (md5 41264a155ed6c37847aac30e630054cb) without main() and its argument parsing, and with
# PORT the sibling import of shape_stats replaced by the package import of instrument.py.
# PORT Every definition kept is unchanged.

"""Shape statistics of the fine-scale texture, one row per ensemble member of a full-rung file.

This is the full-rung companion of ``shape_stats.py``. The apparatus (the neighbour graph, the
seven-Gaussian kernel ladder, the box masks, the 0.09 degree regrid) and all three shape
statistics (the flow-relative anisotropy index, the two-point correlation ellipse, the
connected-feature morphology) are taken unchanged from the verbatim copy of that script that
sits next to this file, so that every number here is produced by exactly the same code as the
matched-seed numbers of 2026-09-16. Only two things are new: the reader, and the loop.

The reader. The full-rung prediction files are global (6,599,680 O1280 points) and hold ten
ensemble members in which every member has its own input. So member k carries its own
interpolated coarse driver ``x_interp_k``, its own truth member ``y_k`` and its own draw
``y_pred_k``. The identity assertions of ``shape_stats.read_arm`` (truth and driver bit-identical
across draws) therefore must NOT be applied here; instead this reader asserts the opposite, that
``x_interp`` really does differ between members, which is the sanity check that these are ten
different inputs rather than ten draws on one input. The box of the epic (10-40N, 100-58W) is cut
out of the global field by a bounding-box selection on ``lat_hres``/``lon_hres``; that selection
was verified to reproduce the 185,146 box points of the matched-seed files exactly, in the same
order, with zero coordinate difference.

The loop. For each member k the two fields of interest are

    d_k = (y_pred_k - x_interp_k) / sd     the residual predicted by member k
    r_k = (y_k      - x_interp_k) / sd     the truth residual of the same member

plus, as a reference, the driver ``x_k = x_interp_k / sd`` itself. The per-variable scale ``sd``
is the training standard deviation used by ``shape_stats.py``; all three statistics are scale
invariant, so this only keeps the numbers comparable. The local wind direction that makes the
anisotropy index and the angle-to-wind flow-relative is taken from that member's own
``x_interp_k``. One CSV row is written per (arm, date, step, variable, member, field, band).
"""

import argparse
import json
import os
import sys
import time

import netCDF4 as nc
import numpy as np

# PORT: sys.path.insert(...) dropped; the instrument is imported from the package
from . import instrument as S  # PORT: was `import shape_stats as S` (the verbatim copy)

EVAL = "/home/ecm5702/scratch/eval/o320_o1280"
PROBE = ("/home/ecm5702/agent-work/20260915-matched-feature-draws/outputs/R47k/"
         "seed_2026091600/predictions/predictions_20230829_step024.nc")

BANDS = ("mid", "b12")
FIELDS = ("draw", "truth", "driver")

CSV_COLS = ["arm", "dirname", "date", "step", "var", "member", "field", "band",
            "A_open_ocean", "A_all_interior",
            "ell_major_km_median", "ell_minor_km_median", "ell_ratio_median",
            "ell_angle_to_wind_deg_median", "n_windows",
            "elong_frac_gt3", "elong_median", "major_km_median", "minor_km_median",
            "area_km2_median", "angle_to_wind_median", "n_components"]


def log(msg):
    print(f"[{time.time()-S.T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# the new reader
# --------------------------------------------------------------------------------------


def box_selection(path):
    """Indices of the epic's box inside the global grid of a full-rung file, plus its lat/lon."""
    d = nc.Dataset(PROBE)
    blat = np.asarray(d["lat_hres"][:], dtype=np.float64)
    blon = np.asarray(d["lon_hres"][:], dtype=np.float64)
    d.close()
    d = nc.Dataset(path)
    glat = np.asarray(d["lat_hres"][:], dtype=np.float64)
    glon = np.asarray(d["lon_hres"][:], dtype=np.float64)
    d.close()
    if glat.size == blat.size and np.array_equal(glat, blat):
        return np.arange(glat.size), blat, blon, "file is already the box"
    e = 1e-9
    sel = np.flatnonzero((glat >= blat.min() - e) & (glat <= blat.max() + e)
                         & (glon >= blon.min() - e) & (glon <= blon.max() + e))
    assert sel.size == blat.size, f"box selection gave {sel.size}, expected {blat.size}"
    assert np.array_equal(glat[sel], blat), "box latitudes do not match the probe file"
    assert np.array_equal(glon[sel], blon), "box longitudes do not match the probe file"
    return sel, blat, blon, f"box cut out of {glat.size} global points by bounding box"


def read_fullrung(path, var_names, sel, members=None):
    """Read a full-rung file: returns (yp, yy, xi) each (nmember, nbox, nvar), plus notes.

    Unlike the matched-seed reader, every member has its own input, so the check here is that
    x_interp DIFFERS between members.
    """
    notes = []
    d = nc.Dataset(path)
    st = [str(x) for x in d["weather_state"][:]]
    si = [st.index(v) for v in var_names]
    nm = d.dimensions["ensemble_member"].size
    ks = list(range(nm)) if members is None else list(members)
    yp = np.empty((len(ks), sel.size, len(si)), dtype=np.float32)
    yy = np.empty_like(yp)
    xi = np.empty_like(yp)
    # the weather-state indices of 10u/10v are adjacent, so one strided slice per field per
    # member is enough; the files are contiguous and uncompressed, so this costs about 2 s each
    contiguous = (max(si) - min(si) + 1) == len(si) and si == sorted(si)
    for a, k in enumerate(ks):
        for dest, name in ((yp, "y_pred"), (yy, "y"), (xi, "x_interp")):
            if contiguous:
                blk = np.asarray(d[name][0, k, :, min(si):max(si) + 1], dtype=np.float32)
                dest[a] = blk[sel]
            else:
                for b, j in enumerate(si):
                    dest[a, :, b] = np.asarray(d[name][0, k, :, j], dtype=np.float32)[sel]
        log(f"    read member {k} ({a+1}/{len(ks)})")
    d.close()

    # sanity: these really are ten different inputs, not ten draws on one input
    if len(ks) > 1:
        same = [a for a in range(1, len(ks)) if np.array_equal(xi[a], xi[0])]
        assert not same, (f"x_interp is identical to member {ks[0]} for members "
                          f"{[ks[a] for a in same]}; this does not look like a full-rung file")
        samey = [a for a in range(1, len(ks)) if np.array_equal(yy[a], yy[0])]
        assert not samey, f"y is identical across members {[ks[a] for a in samey]}"
        notes.append(f"per-member input check passed: x_interp and y both differ between all "
                     f"{len(ks)} members")
    assert np.isfinite(yp).all() and np.isfinite(yy).all() and np.isfinite(xi).all()
    return yp, yy, xi, ks, notes


# --------------------------------------------------------------------------------------
# the new loop
# --------------------------------------------------------------------------------------


def wind_of(ap, xi_k, var_names):
    """Local wind direction and its regridded components, from one member's own driver."""
    nbw = ap.idx[:, :S.NWIND_SMOOTH].astype(np.int64)
    uw = xi_k[:, var_names.index("10u")][nbw].mean(axis=1)
    vw = xi_k[:, var_names.index("10v")][nbw].mean(axis=1)
    sp = np.sqrt(uw ** 2 + vw ** 2)
    sp[sp < 1e-6] = 1e-6
    uhat = (uw / sp).astype(np.float32)
    vhat = (vw / sp).astype(np.float32)
    return uhat, vhat, ap.to_grid(uw.astype(np.float32)), ap.to_grid(vw.astype(np.float32))


def stats_of_field(ap, band_field, uhat, vhat, gu, gv):
    """All three statistics of one band-passed field, as a dict of the CSV column names."""
    out = {}
    out["A_open_ocean"] = S.anisotropy(ap, band_field, uhat, vhat, "open_ocean_interior")[0]
    out["A_all_interior"] = S.anisotropy(ap, band_field, uhat, vhat, "all_interior")[0]
    g = ap.to_grid(band_field)
    ells = S.correlation_ellipses(ap, g, gu, gv)
    for key, ki in (("ell_major_km_median", 0), ("ell_minor_km_median", 1),
                    ("ell_ratio_median", 2), ("ell_angle_to_wind_deg_median", 3)):
        out[key] = S.quart(ells, ki)[0]
    out["n_windows"] = float(len(ells))
    m = S.morph_summary(S.feature_morphology(ap, g, gu, gv))
    for key in ("elong_median", "elong_frac_gt3", "major_km_median", "minor_km_median",
                "area_km2_median", "angle_to_wind_median", "n_components"):
        out[key] = m[key]
    return out


def run_file(ap, arm, dirname, date, step, path, var_names, sel, stdev, rows,
             members=None, bands=BANDS):
    yp, yy, xi, ks, notes = read_fullrung(path, var_names, sel, members)
    for n in notes:
        log(f"  note: {n}")
    for a, k in enumerate(ks):
        uhat, vhat, gu, gv = wind_of(ap, xi[a], var_names)
        for var in var_names:
            sd = stdev[var]
            vi = var_names.index(var)
            raw = {
                "draw": ((yp[a, :, vi] - xi[a, :, vi]) / sd).astype(np.float32),
                "truth": ((yy[a, :, vi] - xi[a, :, vi]) / sd).astype(np.float32),
                "driver": (xi[a, :, vi] / sd).astype(np.float32),
            }
            for field in FIELDS:
                bp = ap.bandpass(raw[field], bands)
                for b in bands:
                    st = stats_of_field(ap, bp[b], uhat, vhat, gu, gv)
                    rows.append(dict(arm=arm, dirname=dirname, date=date,
                                     step=f"{int(step):03d}", var=var, member=k,
                                     field=field, band=b, **st))
        log(f"  member {k} done ({a+1}/{len(ks)})")
    return len(ks)


def write_csv(path, rows):
    exists = os.path.exists(path)
    with open(path, "w") as fh:
        fh.write(",".join(CSV_COLS) + "\n")
        for r in rows:
            vals = []
            for c in CSV_COLS:
                v = r[c]
                if isinstance(v, str):
                    vals.append(v)
                elif v is None or not np.isfinite(float(v)):
                    vals.append("")
                else:
                    vals.append(f"{float(v):.8g}")
            fh.write(",".join(vals) + "\n")
    log(f"wrote {path} with {len(rows)} rows (overwrote={exists})")
