# PORT NOTE (2026-10-12, eval/evaluators/box_spectra): the regular-grid sampler and the
# isotropic spectrum of docs
# epics/fast-generative-downscaling/completed-tasks/20260930_T1_stage1_results/local_plots/plot_sampler_texture.py
# (classes Sampler and Spectrum, constants SPEC_RES_DEG, SPEC_EDGE_MARGIN_DEG, KM_PER_DEG, 40 bins),
# which standing rule 14 names as THE per-bin box tool. The numerical lines are unchanged; the
# docstrings are shortened and Sampler's default method is "linear", the script's command-line
# default and the rule-14 setting (nearest aliases the octahedral rows into a beat near 30 km). The reader of that script is not ported (it reads every weather
# state of every point, gigabytes on a global file): the runner reads the needed columns only.
"""Per-bin amplitude spectrum of a box: linear (barycentric) sampling of the O1280 points on a
regular 0.07-degree lat/lon grid, mean removed, 2-D Hann window, FFT, azimuthal average of |F|^2
in 40 log-spaced wavelength bins (the empty largest-scale bins dropped, which leaves 38 on the
Franklin/Idalia box). Amplitude = sqrt(power). The same operator is applied to the truth, the
interpolated input and the prediction, so their ratios are fair."""
from __future__ import annotations

import numpy as np

KM_PER_DEG = 111.195
SPEC_RES_DEG = 0.07          # regular grid step of the spectra
SPEC_EDGE_MARGIN_DEG = 1.0   # strip inside the box (cut-graph edge)
NBINS = 40


class Sampler:
    """Sampling of the unstructured points on a regular lat/lon grid; weights built once."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, extent, res: float, lat_scale: float = 1.2,
                 method: str = "linear"):
        from scipy.spatial import cKDTree
        lat_min, lat_max, lon_min, lon_max = extent
        m = (lat >= lat_min - 0.5) & (lat <= lat_max + 0.5) & (lon >= lon_min - 0.5) & (lon <= lon_max + 0.5)
        if not np.any(m):
            raise ValueError(f"no source points inside extent {extent}")
        self.src_idx = np.flatnonzero(m)
        self.gx = np.arange(lon_min, lon_max + 1e-9, res)
        self.gy = np.arange(lat_min, lat_max + 1e-9, res)
        GX, GY = np.meshgrid(self.gx, self.gy)
        self.shape = GY.shape
        self.method = method
        self.res = res
        self.lat0 = 0.5 * (lat_min + lat_max)
        pts = np.column_stack([GX.ravel(), GY.ravel()])
        if method == "nearest":
            tree = cKDTree(np.column_stack([lon[m], lat[m] * lat_scale]))
            _, idx = tree.query(np.column_stack([pts[:, 0], pts[:, 1] * lat_scale]), workers=-1)
            self.idx = self.src_idx[idx].reshape(self.shape)
        elif method == "linear":
            # barycentric (Delaunay) interpolation, weights computed once and reused for every field
            from scipy.spatial import Delaunay
            tri = Delaunay(np.column_stack([lon[m], lat[m]]))
            simplex = tri.find_simplex(pts)
            if np.any(simplex < 0):
                raise ValueError(f"{int((simplex < 0).sum())} grid points outside the triangulation")
            T = tri.transform[simplex]                     # (n, 3, 2)
            b = np.einsum("nij,nj->ni", T[:, :2, :], pts - T[:, 2, :])
            self.w = np.column_stack([b, 1.0 - b.sum(axis=1)])      # (n, 3)
            self.verts = self.src_idx[tri.simplices[simplex]]      # (n, 3) global point indices
        else:
            raise ValueError(method)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if self.method == "nearest":
            return values[self.idx]
        return np.sum(values[self.verts] * self.w, axis=1).reshape(self.shape)

    def dx_dy_km(self) -> tuple[float, float]:
        return (self.res * KM_PER_DEG * np.cos(np.deg2rad(self.lat0)), self.res * KM_PER_DEG)


class Spectrum:
    """Isotropic 2-D power spectrum on a regular grid: mean removed, 2-D Hann window, FFT,
    azimuthal average of |F|^2 in log-spaced wavelength bins. Amplitude = sqrt(power)."""

    def __init__(self, ny: int, nx: int, dx_km: float, dy_km: float, nbins: int = NBINS):
        self.ny, self.nx = ny, nx
        wy = np.hanning(ny)[:, None]
        wx = np.hanning(nx)[None, :]
        self.win = wy * wx
        self.win_power = float(np.mean(self.win ** 2))
        kx = np.fft.fftfreq(nx, d=dx_km)   # cycles per km
        ky = np.fft.fftfreq(ny, d=dy_km)
        KX, KY = np.meshgrid(kx, ky)
        k = np.hypot(KX, KY)
        kmax = min(0.5 / dx_km, 0.5 / dy_km)                 # Nyquist (cycles/km)
        kmin = 1.0 / min(nx * dx_km, ny * dy_km)              # one wave across the shorter side
        edges = np.geomspace(kmin, kmax, nbins + 1)
        self.wavelength_km = 1.0 / np.sqrt(edges[:-1] * edges[1:])   # bin centres (geometric)
        self.band_lo_km = 1.0 / edges[1:]
        self.band_hi_km = 1.0 / edges[:-1]
        self.bin = np.digitize(k.ravel(), edges) - 1
        valid = (self.bin >= 0) & (self.bin < nbins) & (k.ravel() > 0)
        self.bin = np.where(valid, self.bin, -1)
        self.counts = np.bincount(self.bin[self.bin >= 0], minlength=nbins).astype(np.float64)
        self.nbins = nbins
        # bins without any FFT cell (the largest scales, finer than the FFT resolution) are dropped
        self.keep = self.counts > 0
        self.wavelength_km = self.wavelength_km[self.keep]
        self.band_lo_km = self.band_lo_km[self.keep]
        self.band_hi_km = self.band_hi_km[self.keep]

    def fft(self, grid: np.ndarray) -> np.ndarray:
        a = (grid - np.mean(grid)) * self.win
        return np.fft.fft2(a) / np.sqrt(self.win_power * self.nx * self.ny)

    def power(self, F: np.ndarray) -> np.ndarray:
        """Azimuthally averaged power per bin (mean |F|^2 over the cells of the bin)."""
        p = (np.abs(F) ** 2).ravel()
        sel = self.bin >= 0
        s = np.bincount(self.bin[sel], weights=p[sel], minlength=self.nbins)
        return (s / np.maximum(self.counts, 1.0))[self.keep]

    def cross(self, F1: np.ndarray, F2: np.ndarray) -> np.ndarray:
        p = np.real(F1 * np.conj(F2)).ravel()
        sel = self.bin >= 0
        s = np.bincount(self.bin[sel], weights=p[sel], minlength=self.nbins)
        return (s / np.maximum(self.counts, 1.0))[self.keep]
