"""Shared pieces of the T1d dense-trajectory DP package (numpy + scipy, CPU only).

- dense and reference schedules (the fork's piecewise scheduler re-implemented in numpy,
  checked against the fork in tests/test_dp.py when torch is importable);
- reading one trajectory_states_s<seed>.npz written by
  `interp/tools/trajectory.py --save-trajectory-states`;
- the band splitter: the box cells are resampled to a regular lat/lon grid the way
  plot_sampler_texture_v3.py does (its Sampler class, 0.07 deg, linear barycentric by
  default), Hann-windowed, FFT'd, and the energy is summed inside sharp wavelength cuts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

KM_PER_DEG = 111.195            # plot_sampler_texture_v3.py
SPEC_RES_DEG = 0.07             # plot_sampler_texture_v3.py (regular grid step of the spectra)
SIGMA_MAX = 1.0e5
SIGMA_MIN = 0.03
COST_VARS = ("10u", "10v", "2t", "msl")          # the four variables of the FIXED costs
CUTS_KM = (100.0, 300.0)                          # 100 km = the cost's cut; 300 km reported
W_FINE, W_COARSE = 0.5, 0.5                       # C_band weights, FIXED before any read


# ---------------------------------------------------------------------------
# the storm box: Idalia, inside the Franklin-Idalia cut graph
# ---------------------------------------------------------------------------

CUT_BOX = (10.0, 40.0, 260.0, 302.0)        # lat0, lat1, lon0, lon1 (0..360) of the cut graph (10-40N, 100-58W)
EDGE_MARGIN_DEG = 1.0                        # the 500 km disc must stay this far inside the cut edges
BOX_RADIUS_KM = 500.0
# msl-minimum search windows per (date, lead), chosen to contain Idalia and exclude Franklin (which sits at
# about 289-295E on 27-31 Aug). Idalia (NHC track, valid times): 27 Aug 00Z TD near 20.5N 86W (274E, NW
# Caribbean); 29 Aug 00Z near 23N 85W (275E); 31 Aug 00Z near 32.5N 80W (280E, SE US coast); 2 Sep 00Z
# post-tropical near 32N 65W (295E, near Bermuda). Each window keeps the disc at least 1 deg inside the cut.
IDALIA_WINDOWS = {
    ("20230826", "024"): (17.0, 26.0, 270.0, 280.0),
    ("20230826", "120"): (27.0, 34.0, 272.0, 285.0),
    ("20230828", "024"): (19.0, 28.0, 270.0, 280.0),
    ("20230828", "120"): (27.0, 34.0, 284.0, 295.6),
}


def disc_extent(clat, clon, radius_km=BOX_RADIUS_KM):
    dlat = radius_km / KM_PER_DEG
    dlon = radius_km / (KM_PER_DEG * math.cos(math.radians(clat)))
    return clat - dlat, clat + dlat, clon - dlon, clon + dlon


def box_checks(clat, clon, radius_km, window=None, cut=CUT_BOX, margin=EDGE_MARGIN_DEG):
    """[(ok, message)]: the disc stays `margin` deg inside the cut graph, and the centre lies strictly
    inside the Idalia window (a centre on the window edge means the minimum there was cut off: another
    low, or the storm outside the window)."""
    clon = clon % 360.0
    la0, la1, lo0, lo1 = disc_extent(clat, clon, radius_km)
    out = [(la0 >= cut[0] + margin and la1 <= cut[1] - margin and lo0 >= cut[2] + margin and lo1 <= cut[3] - margin,
            f"disc {la0:.2f}..{la1:.2f}N {lo0:.2f}..{lo1:.2f}E at least {margin} deg inside the cut "
            f"{cut[0]:.0f}-{cut[1]:.0f}N {cut[2]:.0f}-{cut[3]:.0f}E")]
    if window is not None:
        w0, w1, w2, w3 = window
        e = 0.15
        out.append((w0 + e < clat < w1 - e and w2 + e < clon < w3 - e,
                    f"centre {clat:.2f}N {clon:.2f}E strictly inside the Idalia window {window} (not on its edge)"))
    return out


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------

def dense_levels(n: int = 240, sigma_max: float = SIGMA_MAX, sigma_min: float = SIGMA_MIN) -> np.ndarray:
    """n log-uniform positive levels, endpoints exact."""
    s = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), n))
    s[0], s[-1] = sigma_max, sigma_min
    return s


def custom_scheduler_json(sigmas) -> dict:
    """noise-scheduler block for the fork's `schedule_type: custom` (positive levels only;
    the scheduler appends the terminal zero). num_steps = number of positive levels."""
    vals = [float("%.10g" % v) for v in sigmas]
    vals[0], vals[-1] = float(sigmas[0]), float(sigmas[-1])
    return {"schedule_type": "custom", "sigmas": vals, "sigma_max": vals[0],
            "sigma_min": vals[-1], "num_steps": len(vals)}


def _karras(a, b, n, rho):
    if n <= 1:
        return np.array([a], dtype=np.float64)
    i = np.arange(n, dtype=np.float64)
    return (a ** (1 / rho) + i / (n - 1.0) * (b ** (1 / rho) - a ** (1 / rho))) ** rho


def _expo(a, b, n):
    if n <= 1:
        return np.array([a], dtype=np.float64)
    return np.exp(np.linspace(math.log(a), math.log(b), n))


def piecewise_levels(sigma_max, sigma_min, n_high, n_low, sigma_transition=10.0, rho=7.0,
                     high="exponential", low="karras") -> np.ndarray:
    """Positive levels of the fork's ExperimentalSamplerScheduler (experimental_piecewise):
    high segment n_high+1 points sigma_max -> transition, low segment n_low points
    transition -> sigma_min, joined without repeating the transition."""
    seg = {"karras": lambda a, b, n: _karras(a, b, n, rho), "exponential": _expo}
    h = seg[high](sigma_max, sigma_transition, n_high + 1)
    lo = seg[low](sigma_transition, sigma_min, n_low)
    return np.concatenate([h, lo[1:]])


REFERENCE_SCHEDULES = {
    # quality reference: 30 steps, 10 + 20, 1e5, transition 10, exp above / Karras rho 7 below
    "c0_30": dict(sigma_max=1e5, sigma_min=0.03, n_high=10, n_low=20),
    # cost reference: 16 steps, 5 + 11, sigma_max 1e3
    "c0_pw16_s1k": dict(sigma_max=1e3, sigma_min=0.03, n_high=5, n_low=11),
}


def reference_levels(name: str) -> np.ndarray:
    return piecewise_levels(**REFERENCE_SCHEDULES[name])


def nearest_level_indices(levels: np.ndarray, sigmas) -> np.ndarray:
    """Nearest dense level in ln(sigma) for each sigma (monotone; duplicates are reported
    by the caller)."""
    ll = np.log(np.asarray(levels, dtype=np.float64))
    return np.array([int(np.argmin(np.abs(ll - math.log(s)))) for s in sigmas], dtype=np.int64)


# ---------------------------------------------------------------------------
# trajectory_states npz
# ---------------------------------------------------------------------------

class Trajectory:
    """One draw (one seed of one bundle). Reference states x_i and D_i are the FIRST Heun
    evaluations (call 2i), the sampler state at the start of step i on level i."""

    def __init__(self, path):
        self.path = Path(path)
        z = np.load(self.path, allow_pickle=False)
        self.vars = [str(v) for v in z["vars"]]
        sig = z["sigma"].astype(np.float64)
        he = z["heun_eval"].astype(int)
        first = np.flatnonzero(he == 1)
        self.sigma = sig[first]                          # (N,) levels actually used
        if np.any(np.diff(self.sigma) >= 0):
            raise ValueError(f"{path}: first-evaluation sigmas are not strictly decreasing")
        self.x = z["x_in"][first].astype(np.float64)     # (N, V, n)
        self.D = z["D"][first].astype(np.float64)        # (N, V, n)
        second = np.flatnonzero(he == 2)
        self.x2 = z["x_in"][second].astype(np.float64)   # Euler-predicted points (N-1, V, n)
        self.D2 = z["D"][second].astype(np.float64)
        self.sigma2 = sig[second]
        self.final = z["final"].astype(np.float64)       # (V, n)
        self.truth = z["truth_residual"].astype(np.float64)
        self.lat = z["lat"].astype(np.float64)
        self.lon = z["lon"].astype(np.float64) % 360.0
        self.meta = {k[5:]: z[k].item() if z[k].ndim == 0 else z[k].tolist()
                     for k in z.files if k.startswith("meta_")}
        self.stride = int(z["stride"]) if "stride" in z.files else 1
        z.close()

    def var_index(self, name):
        return self.vars.index(name)

    @property
    def seed(self):
        return int(self.meta.get("seed", -1))


def draw_label(path: Path) -> str:
    """<bundle dir name>/<file stem>, e.g. 20230826_024/trajectory_states_s1000."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def draw_date(tr: Trajectory, path: Path) -> str:
    """YYYYMMDD of the draw from the bundle file name in the metadata (`..._date20230826_...`),
    else from the output directory name (`d20230826_l024`). Raises when neither carries it:
    a bare 8-digit run is never trusted (the bundle root itself contains `_20260818`)."""
    import re
    m = re.search(r"date(\d{8})", Path(str(tr.meta.get("bundle", ""))).name)
    if m:
        return m.group(1)
    for part in Path(path).parts[::-1]:
        m = re.fullmatch(r"d(\d{8})_l\d{3}", part)
        if m:
            return m.group(1)
    raise ValueError(f"{path}: no draw date in the bundle name {tr.meta.get('bundle')!r} or a d<YYYYMMDD>_l<LLL> dir")


# ---------------------------------------------------------------------------
# band splitter (v3 resampling + sharp spectral cuts)
# ---------------------------------------------------------------------------

class GridSampler:
    """plot_sampler_texture_v3.Sampler: unstructured points -> regular lat/lon grid,
    nearest (lat scaled by 1.2 in the KD-tree) or linear barycentric (Delaunay)."""

    def __init__(self, lat, lon, extent, res=SPEC_RES_DEG, lat_scale=1.2, method="linear"):
        from scipy.spatial import cKDTree, Delaunay
        lat_min, lat_max, lon_min, lon_max = extent
        m = (lat >= lat_min - 0.5) & (lat <= lat_max + 0.5) & (lon >= lon_min - 0.5) & (lon <= lon_max + 0.5)
        if not np.any(m):
            raise ValueError(f"no source points inside extent {extent}")
        self.src_idx = np.flatnonzero(m)
        self.gx = np.arange(lon_min, lon_max + 1e-9, res)
        self.gy = np.arange(lat_min, lat_max + 1e-9, res)
        GX, GY = np.meshgrid(self.gx, self.gy)
        self.shape = GY.shape
        self.method, self.res = method, res
        self.lat0 = 0.5 * (lat_min + lat_max)
        pts = np.column_stack([GX.ravel(), GY.ravel()])
        if method == "nearest":
            tree = cKDTree(np.column_stack([lon[m], lat[m] * lat_scale]))
            _, idx = tree.query(np.column_stack([pts[:, 0], pts[:, 1] * lat_scale]))
            self.idx = self.src_idx[idx].reshape(self.shape)
        elif method == "linear":
            tri = Delaunay(np.column_stack([lon[m], lat[m]]))
            simplex = tri.find_simplex(pts)
            if np.any(simplex < 0):
                raise ValueError(f"{int((simplex < 0).sum())} grid points outside the triangulation")
            T = tri.transform[simplex]
            b = np.einsum("nij,nj->ni", T[:, :2, :], pts - T[:, 2, :])
            self.w = np.column_stack([b, 1.0 - b.sum(axis=1)])
            self.verts = self.src_idx[tri.simplices[simplex]]
        else:
            raise ValueError(method)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """values (..., n_points) -> (..., ny, nx)."""
        if self.method == "nearest":
            return values[..., self.idx]
        out = np.sum(values[..., self.verts] * self.w, axis=-1)
        return out.reshape(values.shape[:-1] + self.shape)

    def dx_dy_km(self):
        return (self.res * KM_PER_DEG * math.cos(math.radians(self.lat0)), self.res * KM_PER_DEG)


def box_extent(lat, lon, center_lat=None, center_lon=None, radius_km=None, margin_km=10.0):
    """Regular-grid extent for the box: the square inscribed in the storm circle when the
    centre and radius are known (the box is a disc), else the points' bounding box less
    one grid step."""
    if radius_km and radius_km > 0 and center_lat is not None:
        h = radius_km / math.sqrt(2.0) - margin_km
        dlat = h / KM_PER_DEG
        dlon = h / (KM_PER_DEG * math.cos(math.radians(center_lat)))
        return (center_lat - dlat, center_lat + dlat, center_lon - dlon, center_lon + dlon)
    return (lat.min() + SPEC_RES_DEG, lat.max() - SPEC_RES_DEG,
            lon.min() + SPEC_RES_DEG, lon.max() - SPEC_RES_DEG)


class BandSplitter:
    """Linear map box field -> windowed spectrum, and band energies with sharp cuts.

    energy(F, mask) = sum_{k in mask} |F_k|^2 / (nx ny)^2 / mean(win^2), so that the
    energies of all bands sum to the (window-weighted) mean square of the field. The
    fields are NOT mean-removed here (an error's box mean is a real coarse error); the
    normaliser removes the mean of the final state (it is a variance)."""

    def __init__(self, sampler: GridSampler, cuts_km=CUTS_KM):
        self.sampler = sampler
        ny, nx = sampler.shape
        dx, dy = sampler.dx_dy_km()
        self.ny, self.nx, self.dx, self.dy = ny, nx, dx, dy
        self.win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
        self.norm = 1.0 / ((nx * ny) ** 2 * float(np.mean(self.win ** 2)))
        kx = np.fft.fftfreq(nx, d=dx)
        ky = np.fft.fftfreq(ny, d=dy)
        k = np.hypot(*np.meshgrid(kx, ky))                     # cycles per km, (ny, nx)
        self.k = k
        self.cuts_km = tuple(float(c) for c in cuts_km)
        # fine = wavelength < cut  <=>  k > 1/cut ; coarse = the rest (k = 0 included)
        self.masks = {}
        for c in self.cuts_km:
            fine = k > 1.0 / c
            self.masks[f"fine{int(c)}"] = fine.ravel()
            self.masks[f"coarse{int(c)}"] = (~fine).ravel()
        self.band_names = list(self.masks)
        self.mask_mat = np.stack([self.masks[b] for b in self.band_names], axis=1).astype(np.float64)
        self.nyquist_km = 2.0 * max(dx, dy)
        self.box_km = (nx * dx, ny * dy)

    def spectrum(self, values: np.ndarray) -> np.ndarray:
        """values (..., n_points) -> complex spectra (..., ny*nx)."""
        g = self.sampler(values) * self.win
        F = np.fft.fft2(g, axes=(-2, -1))
        return F.reshape(F.shape[:-2] + (self.ny * self.nx,))

    def energies(self, F: np.ndarray) -> dict:
        """F (..., ny*nx) -> {band: (...)} energies."""
        p = (F.real ** 2 + F.imag ** 2)
        e = (p @ self.mask_mat) * self.norm                    # (..., n_bands), one BLAS call
        return {b: e[..., k] for k, b in enumerate(self.band_names)}

    def variance(self, values: np.ndarray) -> dict:
        """Band variances of a field (mean removed on the regular grid before the window)."""
        g = self.sampler(values)
        g = (g - g.mean(axis=(-2, -1), keepdims=True)) * self.win
        F = np.fft.fft2(g, axes=(-2, -1)).reshape(g.shape[:-2] + (self.ny * self.nx,))
        return self.energies(F)


def splitter_for(tr: Trajectory, method="linear", cuts_km=CUTS_KM) -> BandSplitter:
    m = tr.meta
    radius = float(m.get("radius_km", 0) or 0)
    if radius > 0:
        clat = float(m.get("center_lat"))
        clon = float(m.get("center_lon")) % 360.0
        ext = box_extent(tr.lat, tr.lon, clat, clon, radius)
    else:
        ext = box_extent(tr.lat, tr.lon)
    return BandSplitter(GridSampler(tr.lat, tr.lon, ext, method=method), cuts_km=cuts_km)


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
