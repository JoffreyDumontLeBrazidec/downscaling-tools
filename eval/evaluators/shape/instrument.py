# PORT NOTE (2026-09-23, eval/evaluators/shape): this module is a verbatim copy of
# PORT /home/ecm5702/agent-work/20260917-shape-probes/scripts/shape_stats.py
# PORT (md5 7c76aff1c4d61c7ab1575034cbcaa3b4). Only the parts the full-rung instrument uses are kept:
# PORT the lines up to the end of "statistic 3" and the summary helpers quart / morph_summary.
# PORT The matched-seed reader (read_arm), the matched-seed loop (run_one), the sanity checks and
# PORT main() were dropped. No kept line was changed; the module constants (cache paths) are
# PORT overridden, when asked, by the evaluator runner through attribute assignment.

"""Shape of the fine-scale texture of residual-diffusion draws: is it filaments or blobs?

The spectrum of these fields is already known to be about right, so energy per scale cannot
separate a filament elongated along the flow from an isotropic blob carrying the same energy.
This script measures shape directly, with the three statistics pre-registered in
docs/epics/fine-scale-o320-o1280/in-progress/20260916_shape_instrument_stochastic_part.md:

  1. a flow-relative anisotropy index A = var(gradient along the wind) / var(gradient across),
  2. the ellipse of the two-point autocorrelation in 4x4 degree open-ocean windows,
  3. the morphology of the connected components of the field above its own 90th percentile.

Fields. For each arm, date, lead and variable, with y_pred_k the k-th draw, y the truth and
x_interp the interpolated coarse driver:
    d_k = y_pred_k - x_interp        (the residual predicted by draw k)
    m   = mean_k d_k                 (the draw mean, the model's own conditional mean)
    s_k = d_k - m                    (the stochastic part of draw k)
    r   = y - x_interp               (the truth residual)
    r - m                            (what the draw mean fails to explain)
All are divided by the training standard deviation of the variable, exactly as
condmean_seedrep.py does; every statistic here is scale invariant, so this only keeps the
numbers comparable with that script.

Band pass. The seven-Gaussian kernel ladder of the fine-scale epic, rebuilt here on the
185,146 box points with the neighbour rule copied verbatim from condmean_seedrep.py. With S_j
the j-th smoother (j = 0 the narrowest), the two bands used are
    band 1-2 = S_0 f - S_2 f   (about 21 to 40 km)
    mid band = S_2 f - S_6 f   (about 40 to 109 km)
which are the sums of the epic's band-pass fields 1 and 2, and 3 to 6, respectively.

Error bars. Every draw-dependent number is recomputed on the first half of the draws and on
the second half; the two halves are reported next to the full-sample value, and a difference
between two arms counts only when it is larger than the gap between the halves. The truth has
a single realisation and therefore gets no error bar.
"""

import argparse
import json
import os
import sys
import time

import netCDF4 as nc
import numpy as np
import zarr
from scipy import ndimage
from scipy.sparse import load_npz
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------------------
# paths and constants (kernel rule copied from condmean_seedrep.py)
# --------------------------------------------------------------------------------------

WORK = "/home/ecm5702/agent-work/20260916-shape-instrument"
MATCHED = "/home/ecm5702/agent-work/20260915-matched-feature-draws/outputs"
SEEDREP = "/home/ecm5702/scratch/eval/o320_o1280_ft400k_gmass_wsweep_20260822/seedrep_5date"
STATS = "/home/ecm5702/hpcperm/data/residuals_statistics/o1280_dict_0_72.npy"
KNN = "/home/ecm5702/scratch/agent-work/20260901-sigma-scale-map/o1280_knn128.npz"
IM = "/home/ecm5702/hpcperm/data/inter_mat"
FORC = "/home/mlx/ai-ml/datasets/downscaling-od-cf-enfh-0001-mars-o1280-2003-2023-12h-v3-forcings.zarr"
DONOR_MASKCACHE = "/home/ecm5702/agent-work/20260914-audit-branch-review/outputs/condmean_box_masks.npz"
LOCAL_MASKCACHE = f"{WORK}/outputs/box_masks.npz"
GEOMCACHE = f"{WORK}/outputs/box_geometry.npz"

SIGMAS = [4.12, 5.25, 6.75, 8.62, 11.06, 14.24, 18.37]
GLOBAL_DX = 8.791259463563057
R_EARTH = 6371.0
G = 9.80665
INTERIOR_MARGIN_DEG = 0.7
COASTAL_LSM_SMOOTH_MIN = 0.01
CH = 200_000

# band definitions, as indices into the list of smoothers
BAND_DEF = {"b12": (0, 2), "mid": (2, 6)}

REGRID_DEG = 0.09
WINDOW_DEG = 4.0
WINDOW_OCEAN_MIN = 0.80
MIN_COMPONENT_PIX = 6
FEATURE_PCTL = 90.0
GRAD_H_KM = 8.0    # Gaussian half-width of the weights of the least-squares gradient
GRAD_K = 64        # neighbours offered to that fit (the weights truncate it well before 64)
NWIND_SMOOTH = 3   # neighbours averaged to define the local wind direction

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# geometry, kernels, gradients  (kernel rule identical to condmean_seedrep.py)
# --------------------------------------------------------------------------------------


def unit_vectors(lat_deg, lon_deg):
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lo = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    cl = np.cos(la)
    return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=1)


def build_local_graph(lat, lon, kmax=128):
    xyz = unit_vectors(lat, lon)
    tree = cKDTree(xyz)
    chord, idx = tree.query(xyz, k=kmax, workers=-1)
    chord = np.clip(chord, 0.0, 2.0)
    dist = (2.0 * R_EARTH * np.arcsin(chord / 2.0)).astype(np.float32)
    return idx.astype(np.int32), dist


def build_kernels(dist, n_points, dx):
    kmax = dist.shape[1]
    radius = dist.mean(axis=0)
    kk, W, f = [], [], []
    for s in SIGMAS:
        k = int(np.clip(np.searchsorted(radius, 3.2 * s) + 1, 7, kmax))
        w = np.exp(-0.5 * (dist[:, :k].astype(np.float32) / np.float32(s)) ** 2)
        w /= w.sum(axis=1, keepdims=True)
        kk.append(k)
        W.append(w)
        f.append(float(np.einsum("ij,ij->", w, w, dtype=np.float64) / n_points))
    lam = [2 * dx / np.sqrt(min(1.0, 1.386 * ff)) for ff in f]
    return kk, W, f, lam, radius


def box_spacing_km(lat, lon, n_points):
    la1, la2 = np.deg2rad(float(lat.min())), np.deg2rad(float(lat.max()))
    lo1, lo2 = np.deg2rad(float(lon.min())), np.deg2rad(float(lon.max()))
    area = R_EARTH**2 * (lo2 - lo1) * (np.sin(la2) - np.sin(la1))
    return float(np.sqrt(area / n_points))


def build_gradient_operator(lat, lon, idx, h_km=GRAD_H_KM, kk=GRAD_K):
    """Gaussian-weighted least-squares gradient on the graph, in east/north kilometres.

    The first version of this operator fitted a plane to the eight nearest neighbours. On the
    reduced Gaussian O1280 grid those eight points do not form a symmetric ring: they come in
    pairs at 8.0, 9.7, 11.4 and 13.6 km, the rows above and below are offset in longitude, and
    the resulting stencil is skewed. Pushed through the pipeline, an isotropic random field
    then returned an anisotropy index of 0.66 instead of 1, which would have swamped the
    signal being measured. The fix is to fit the plane to the 64 nearest neighbours with
    Gaussian weights of fixed half-width h_km, so that the effective stencil is a smooth
    isotropic disc that no longer remembers the grid. Sanity check (a) in the log verifies it.

    With A the matrix of east/north offsets and w the weights, the fit is
    g = (A^T W A)^-1 A^T W df, and the operator P = (A^T W A)^-1 A^T W is precomputed here.
    """
    nb = idx[:, :kk].astype(np.int64)          # neighbour 0 is the point itself, offset zero
    la0 = np.deg2rad(lat)[:, None]
    lo0 = np.deg2rad(lon)[:, None]
    dlon = (np.deg2rad(lon)[nb] - lo0 + np.pi) % (2 * np.pi) - np.pi
    de = R_EARTH * np.cos(la0) * dlon
    dn = R_EARTH * (np.deg2rad(lat)[nb] - la0)
    wt = np.exp(-0.5 * ((de ** 2 + dn ** 2) / h_km ** 2))
    A = np.stack([de, dn], axis=2)                      # (n, kk, 2)
    Aw = A * wt[:, :, None]
    ATA = np.einsum("nij,nik->njk", Aw, A)              # (n, 2, 2)
    det = ATA[:, 0, 0] * ATA[:, 1, 1] - ATA[:, 0, 1] * ATA[:, 1, 0]
    inv = np.empty_like(ATA)
    inv[:, 0, 0] = ATA[:, 1, 1] / det
    inv[:, 1, 1] = ATA[:, 0, 0] / det
    inv[:, 0, 1] = -ATA[:, 0, 1] / det
    inv[:, 1, 0] = -ATA[:, 1, 0] / det
    P = np.einsum("nij,nkj->nik", inv, Aw)              # (n, 2, kk)
    return nb.astype(np.int32), P.astype(np.float32)


def gradient(field, nb, P):
    df = field[nb] - field[:, None]
    g = np.einsum("nij,nj->ni", P, df.astype(np.float32))
    return g[:, 0], g[:, 1]      # east, north components, per kilometre


# --------------------------------------------------------------------------------------
# masks (copied from condmean_seedrep.py, reading its cache when it exists)
# --------------------------------------------------------------------------------------


def build_box_masks(lat, lon):
    for cache in (DONOR_MASKCACHE, LOCAL_MASKCACHE):
        if os.path.exists(cache):
            z = np.load(cache)
            log(f"mask cache loaded from {cache}")
            return {k: z[k] for k in z.files}, cache

    log("building box masks from the global forcings")
    zk = np.load(KNN)
    glat = zk["lat"].astype(np.float64)
    glon = zk["lon"].astype(np.float64)
    tree = cKDTree(unit_vectors(glat, glon))
    d, gidx = tree.query(unit_vectors(lat, lon), k=1, workers=-1)
    del tree
    gnn32 = zk["idx"][:, :32]
    z = zarr.open(FORC, mode="r")
    v = json.load(open(FORC + "/.zattrs"))["variables"]
    lsm = np.asarray(z["data"][0, v.index("lsm"), 0, :], dtype=np.float32)
    oro = np.asarray(z["data"][0, v.index("z"), 0, :], dtype=np.float32) / G
    rough = oro[gnn32].std(axis=1)
    del gnn32, zk
    up = load_npz(f"{IM}/interpol_O320_to_O1280_linear.mat.npz").tocsr()
    down = load_npz(f"{IM}/interpol_o1280_to_o320_linear.mat.npz").tocsr()
    lsm_c = np.asarray(up @ (down @ lsm.astype(np.float64)))
    del up, down
    b_lsm = lsm[gidx]
    b_lsm_c = lsm_c[gidx]
    b_rough = rough[gidx]
    lat_lo = float(lat.min()) + INTERIOR_MARGIN_DEG
    lat_hi = float(lat.max()) - INTERIOR_MARGIN_DEG
    lon_lo = float(lon.min()) + INTERIOR_MARGIN_DEG
    lon_hi = float(lon.max()) - INTERIOR_MARGIN_DEG
    interior = (lat >= lat_lo) & (lat <= lat_hi) & (lon >= lon_lo) & (lon <= lon_hi)
    ocean = b_lsm < 0.05
    out = {
        "interior": interior,
        "all_interior": interior,
        "ocean_interior": interior & ocean,
        "open_ocean_interior": interior & ocean & (b_lsm_c < COASTAL_LSM_SMOOTH_MIN),
        "coastal_interior": interior & ocean & (b_lsm_c >= COASTAL_LSM_SMOOTH_MIN),
        "flat_land_interior": interior & (b_lsm > 0.95) & (b_rough < 30.0),
        "mountain_interior": interior & (b_lsm > 0.95) & (b_rough > 150.0),
    }
    os.makedirs(os.path.dirname(LOCAL_MASKCACHE), exist_ok=True)
    np.savez(LOCAL_MASKCACHE, **out)
    return out, LOCAL_MASKCACHE


# --------------------------------------------------------------------------------------
# apparatus assembled once and cached on disk
# --------------------------------------------------------------------------------------


class Apparatus:
    """Graph, kernels, gradient operator, masks and regrid, built once for the box."""

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.n = lat.size
        if os.path.exists(GEOMCACHE):
            z = np.load(GEOMCACHE)
            same = (z["n"] == self.n and np.allclose(z["lat"], lat)
                    and np.allclose(z["lon"], lon)
                    and "grad_h" in z.files and float(z["grad_h"]) == GRAD_H_KM
                    and int(z["grad_k"]) == GRAD_K)
            if same:
                log(f"geometry cache loaded from {GEOMCACHE}")
                self.idx = z["idx"]
                dist = z["dist"]
                self.nb = z["nb"]
                self.P = z["P"]
            else:
                raise SystemExit(f"{GEOMCACHE} was built with different settings; delete it")
        else:
            log("building the 128-neighbour graph of the box")
            self.idx, dist = build_local_graph(lat, lon)
            self.nb, self.P = build_gradient_operator(lat, lon, self.idx)
            os.makedirs(os.path.dirname(GEOMCACHE), exist_ok=True)
            np.savez(GEOMCACHE, n=self.n, lat=lat, lon=lon, idx=self.idx, dist=dist,
                     nb=self.nb, P=self.P, grad_h=GRAD_H_KM, grad_k=GRAD_K)
            log(f"geometry cache written to {GEOMCACHE}")

        self.dx_box = box_spacing_km(lat, lon, self.n)
        self.kk, self.W, self.f, self.lam, _ = build_kernels(dist, self.n, self.dx_box)
        del dist
        log(f"kernel k = {self.kk}")
        log(f"lam (box dx {self.dx_box:.4f} km) = {[round(float(x), 3) for x in self.lam]}")
        self.masks, self.maskcache = build_box_masks(lat, lon)
        self.midx = {m: np.flatnonzero(self.masks[m])
                     for m in ("all_interior", "open_ocean_interior")}
        for m, ii in self.midx.items():
            log(f"mask {m:22s}: {ii.size:7d} points")
        self._gather = np.empty((self.n, self.idx.shape[1]), dtype=np.float32)
        self._build_regrid()

    # --- smoothing ---------------------------------------------------------------------

    def smooth(self, x, js):
        """Apply the smoothers listed in js to x; returns a dict j -> smoothed field."""
        outs = {j: np.empty(self.n, dtype=np.float32) for j in js}
        for aa in range(0, self.n, CH):
            bb = min(aa + CH, self.n)
            np.take(x, self.idx[aa:bb], out=self._gather[aa:bb])
            for j in js:
                k = self.kk[j]
                outs[j][aa:bb] = np.einsum("ij,ij->i", self.W[j][aa:bb],
                                           self._gather[aa:bb, :k])
        return outs

    def bandpass(self, x, bands=("b12", "mid")):
        js = sorted({j for b in bands for j in BAND_DEF[b]})
        sm = self.smooth(x, js)
        return {b: sm[BAND_DEF[b][0]] - sm[BAND_DEF[b][1]] for b in bands}

    def all_bands(self, x):
        """The epic's eight band-pass fields, for the reproduction check."""
        sm = self.smooth(x, list(range(len(SIGMAS))))
        out = [x - sm[0]]
        for j in range(len(SIGMAS) - 1):
            out.append(sm[j] - sm[j + 1])
        out.append(sm[len(SIGMAS) - 1])
        return out

    # --- regrid ------------------------------------------------------------------------

    def _build_regrid(self):
        lat_lo = float(self.lat.min()) + INTERIOR_MARGIN_DEG
        lat_hi = float(self.lat.max()) - INTERIOR_MARGIN_DEG
        lon_lo = float(self.lon.min()) + INTERIOR_MARGIN_DEG
        lon_hi = float(self.lon.max()) - INTERIOR_MARGIN_DEG
        self.glat = np.arange(lat_lo, lat_hi + 1e-9, REGRID_DEG)
        self.glon = np.arange(lon_lo, lon_hi + 1e-9, REGRID_DEG)
        LA, LO = np.meshgrid(self.glat, self.glon, indexing="ij")
        tree = cKDTree(unit_vectors(self.lat, self.lon))
        _, gi = tree.query(unit_vectors(LA.ravel(), LO.ravel()), k=1, workers=-1)
        del tree
        self.reg_idx = gi.reshape(LA.shape).astype(np.int32)
        self.reg_ocean = self.masks["open_ocean_interior"][self.reg_idx]
        self.reg_lat = LA
        self.dlat_km = REGRID_DEG * np.pi / 180.0 * R_EARTH
        self.dlon_km = self.dlat_km * np.cos(np.deg2rad(self.glat))
        log(f"regrid: {self.glat.size} x {self.glon.size} at {REGRID_DEG} deg, "
            f"{100*self.reg_ocean.mean():.1f}% open-ocean interior")

    def to_grid(self, field):
        return field[self.reg_idx]


# --------------------------------------------------------------------------------------
# statistic 1: flow-relative anisotropy index
# --------------------------------------------------------------------------------------


def anisotropy(ap, field, uhat, vhat, mask_name):
    ii = ap.midx[mask_name]
    ge, gn = gradient(field, ap.nb, ap.P)
    along = ge[ii] * uhat[ii] + gn[ii] * vhat[ii]
    across = -ge[ii] * vhat[ii] + gn[ii] * uhat[ii]
    va = float(np.var(along, dtype=np.float64))
    vc = float(np.var(across, dtype=np.float64))
    return (va / vc if vc > 0 else np.nan), va, vc


# --------------------------------------------------------------------------------------
# statistic 2: two-point correlation ellipse
# --------------------------------------------------------------------------------------


def ellipse_from_region(region, cy, cx, dy_km, dx_km, full_axis=False):
    ys, xs = np.nonzero(region)
    if ys.size < 4:
        return None
    dy = (ys - cy) * dy_km
    dx = (xs - cx) * dx_km
    C = np.cov(np.vstack([dx, dy]))
    if not np.all(np.isfinite(C)):
        return None
    ev, evec = np.linalg.eigh(C)
    ev = np.clip(ev, 0.0, None)
    scale = 4.0 if full_axis else 2.0
    minor = scale * float(np.sqrt(ev[0]))
    major = scale * float(np.sqrt(ev[1]))
    vmaj = evec[:, 1]      # (east, north)
    ang = np.arctan2(vmaj[1], vmaj[0])
    return major, minor, ang


def angle_to_wind(ax_angle, u, v):
    """Angle in degrees between an undirected axis and the wind vector, folded to [0, 90]."""
    wa = np.arctan2(v, u)
    d = np.degrees(np.abs(((ax_angle - wa) + np.pi / 2) % np.pi - np.pi / 2))
    return float(min(d, 180.0 - d))


def correlation_ellipses(ap, gfield, gu, gv):
    """One ellipse per 4x4 degree window that is at least 80% open ocean."""
    ny = int(round(WINDOW_DEG / REGRID_DEG))
    nx = ny
    out = []
    for y0 in range(0, ap.glat.size - ny + 1, ny):
        for x0 in range(0, ap.glon.size - nx + 1, nx):
            oc = ap.reg_ocean[y0:y0 + ny, x0:x0 + nx]
            if oc.mean() < WINDOW_OCEAN_MIN:
                continue
            w = gfield[y0:y0 + ny, x0:x0 + nx].astype(np.float64)
            if not np.isfinite(w).all():
                continue
            w = w - w.mean()
            taper = np.outer(np.hanning(ny), np.hanning(nx))
            w = w * taper
            F = np.fft.rfft2(w, s=(2 * ny, 2 * nx))
            ac = np.fft.irfft2(np.abs(F) ** 2, s=(2 * ny, 2 * nx))
            ac = np.fft.fftshift(ac)
            pk = ac[ny, nx]
            if pk <= 0:
                continue
            ac = ac / pk
            # keep lags up to half the window so the taper tail is not fitted
            h = ny // 2
            sub = ac[ny - h:ny + h + 1, nx - h:nx + h + 1]
            reg = sub > (1.0 / np.e)
            lab, nlab = ndimage.label(reg, structure=np.ones((3, 3), int))
            if nlab == 0 or lab[h, h] == 0:
                continue
            reg = lab == lab[h, h]
            latc = float(ap.glat[y0 + ny // 2])
            dxkm = ap.dlat_km * np.cos(np.deg2rad(latc))
            e = ellipse_from_region(reg, h, h, ap.dlat_km, dxkm, full_axis=False)
            if e is None:
                continue
            major, minor, ang = e
            if minor <= 0:
                continue
            um = float(gu[y0:y0 + ny, x0:x0 + nx].mean())
            vm = float(gv[y0:y0 + ny, x0:x0 + nx].mean())
            out.append((major, minor, major / minor, angle_to_wind(ang, um, vm)))
    return out


# --------------------------------------------------------------------------------------
# statistic 3: connected-feature morphology
# --------------------------------------------------------------------------------------


def feature_morphology(ap, gfield, gu, gv):
    a = np.abs(gfield)
    vals = a[ap.reg_ocean]
    if vals.size == 0:
        return []
    thr = float(np.percentile(vals, FEATURE_PCTL))
    binary = (a > thr) & ap.reg_ocean
    lab, nlab = ndimage.label(binary, structure=np.ones((3, 3), int))
    if nlab == 0:
        return []
    out = []
    objs = ndimage.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        npix = int(sub.sum())
        if npix < MIN_COMPONENT_PIX:
            continue
        ys, xs = np.nonzero(sub)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        latc = float(ap.glat[int(round(ys.mean()))])
        dxkm = ap.dlat_km * np.cos(np.deg2rad(latc))
        area = npix * ap.dlat_km * dxkm
        dy = (ys - ys.mean()) * ap.dlat_km
        dx = (xs - xs.mean()) * dxkm
        C = np.cov(np.vstack([dx, dy]))
        if not np.all(np.isfinite(C)):
            continue
        ev, evec = np.linalg.eigh(C)
        ev = np.clip(ev, 0.0, None)
        minor = 4.0 * float(np.sqrt(ev[0]))
        major = 4.0 * float(np.sqrt(ev[1]))
        if minor <= 1e-6:
            minor = ap.dlat_km          # a one-pixel-wide feature
        vmaj = evec[:, 1]
        ang = np.arctan2(vmaj[1], vmaj[0])
        um = float(gu[ys, xs].mean())
        vm = float(gv[ys, xs].mean())
        out.append((area, major, minor, major / minor, angle_to_wind(ang, um, vm)))
    return out




# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------


def quart(vals, k):
    a = np.asarray([v[k] for v in vals], dtype=np.float64)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)))


def morph_summary(comps):
    if not comps:
        return dict(elong_median=np.nan, elong_frac_gt3=np.nan, major_km_median=np.nan,
                    minor_km_median=np.nan, area_km2_median=np.nan,
                    angle_to_wind_median=np.nan, n_components=np.nan)
    el = np.asarray([c[3] for c in comps])
    return dict(elong_median=float(np.median(el)),
                elong_frac_gt3=float(np.mean(el > 3.0)),
                major_km_median=float(np.median([c[1] for c in comps])),
                minor_km_median=float(np.median([c[2] for c in comps])),
                area_km2_median=float(np.median([c[0] for c in comps])),
                angle_to_wind_median=float(np.median([c[4] for c in comps])),
                n_components=len(comps))
