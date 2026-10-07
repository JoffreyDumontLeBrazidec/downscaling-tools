#!/usr/bin/env python3
"""Local maps and amplitude spectra of the 30-step samplers against the truth (CPU only).

Arms: b0 (churn 2.5, S_noise 1.05), c0_30 (churn off), n100_30 (S_noise 1.0), read from the
regional (Franklin/Idalia box) prediction files `predictions_YYYYMMDD_stepNNN.nc` of the RW50k
sampler campaign. Every field goes through the same operator (nearest-neighbour sampling of the
O1280 points on a regular lat/lon grid), so the truth, the interpolated input and the three arms
receive identical treatment and their ratios are fair.

Outputs (under --out):
  maps/      one PNG per (date, lead, region, variable): 4 rows x 4 columns
             row 1  field:            truth | b0 | c0_30 | n100_30
             row 2  fine part (*):    truth | b0 | c0_30 | n100_30
             row 3  minus truth:      input interpolated | b0-truth | c0-truth | n100-truth
             row 4  minus interp:     truth-interp | b0-interp | c0-interp | n100-interp
             (*) Gaussian high-pass on the regular grid, sigma = half of --fine-cut-deg.
  members/   member 1-3 of each arm next to the truth (field and fine part), wind speed.
  spectra/   one PNG per variable: AMPLITUDE spectra (not power) of truth, interp, arms over the
             full scale range; ratio to the truth; amplitude of (arm - truth) over the truth's;
             plus spectra_<var>.csv with the curves and summary.md with band ratios and the
             overall distance to the target.

Usage (example):
  python plot_sampler_texture.py \
     --arm b0=/home/ecm5702/scratch/eval/o320_o1280_hresRW50k_b0_tc_20260929 \
     --arm c0_30=/home/ecm5702/scratch/eval/o320_o1280_hresRW50k_c0_tc_20260929 \
     --arm n100_30=/home/ecm5702/scratch/eval/o320_o1280_hresRW50k_n100_tc_20260929 \
     --out /home/ecm5702/work-rebuild-20261003/fewstep-sampler/local_plots \
     --map-cases 20230828:24,20230826:24,20230826:120 --spectra-leads 24,120

No GPU, no job submission; numpy + scipy + matplotlib + netCDF4 only (cartopy optional, used
for coastlines when importable).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger("plot_sampler_texture")

FILE_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
KM_PER_DEG = 111.195

# Event boxes of the tc evaluator / tc_intensity.py (lat_min, lat_max, lon_min, lon_max).
EVENT_BOXES = {
    "idalia": (10.0, 40.0, -100.0, -80.0),
    "franklin": (15.0, 38.0, -78.0, -58.0),
}
# Regions away from both storms (fixed boxes, clipped to the file's bounds): ocean east of the
# Antilles, and land/coast over Texas and the western Gulf. The first version of this script used
# 10-20N, 58-46W, which lies OUTSIDE the file (its box is lat 10..40, lon 100W..58W): those maps
# show extrapolated edge points, not weather, and are void.
QUIET_BOXES = {
    "quiet_ocean": (12.0, 22.0, -72.0, -60.0),
    "quiet_land": (26.0, 36.0, -98.0, -88.0),
}
MAP_HALF_LAT = 5.0          # zoom window: +-5 deg latitude around the storm centre
MAP_RES_DEG = 0.05          # regular grid step of the maps (O1280 ~ 0.07 deg)
SPEC_RES_DEG = 0.07         # regular grid step of the spectra
SPEC_EDGE_MARGIN_DEG = 1.0  # strip inside the file's lat/lon bounds (cut-graph edge)

VAR_SPECS = {
    "ws": {"states": ("10u", "10v"), "title": "10 m wind speed", "unit": "m/s", "cmap": "viridis"},
    "10u": {"states": ("10u",), "title": "10 m zonal wind (10u)", "unit": "m/s", "cmap": "RdBu_r"},
    "10v": {"states": ("10v",), "title": "10 m meridional wind (10v)", "unit": "m/s", "cmap": "RdBu_r"},
    "2t": {"states": ("2t",), "title": "2 m temperature", "unit": "K", "cmap": "coolwarm"},
    "msl": {"states": ("msl",), "title": "mean sea level pressure", "unit": "hPa", "cmap": "cividis"},
}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def find_files(root: Path) -> dict[tuple[str, int], Path]:
    """{(date, step): path} of every prediction file under root (symlinks followed, deduplicated)."""
    found: dict[tuple[str, int], Path] = {}
    for p in sorted(root.rglob("predictions_*_step*.nc")):
        m = FILE_RE.search(p.name)
        if not m:
            continue
        key = (m.group(1), int(m.group(2)))
        real = p.resolve()
        if not real.is_file():
            LOG.warning("%s: broken link or missing file, skipped", p)
            continue
        if key in found:
            if found[key] != real:
                # prefer a file that sits in a folder called `predictions` (the campaign's own output)
                if real.parent.name == "predictions" and found[key].parent.name != "predictions":
                    LOG.warning("%s: two different files for %s; keeping %s over %s", root, key, real, found[key])
                    found[key] = real
                else:
                    LOG.warning("%s: two different files for %s: keeping %s over %s", root, key, found[key], real)
            continue
        found[key] = real
    return found


def _decode_states(var) -> list[str]:
    raw = var[:]
    out = []
    for v in np.asarray(raw).reshape(-1):
        if isinstance(v, bytes):
            out.append(v.decode())
        elif isinstance(v, np.ndarray):
            out.append(b"".join(x for x in v.tolist() if x).decode() if v.dtype.kind == "S" else str(v))
        else:
            out.append(str(v))
    return out


def _read_member(var, member_index: int) -> np.ndarray:
    """One member of a (sample, ensemble_member, grid_point, weather_state) variable as
    (points, states), whatever the on-disk dimension order (same rule as the texture evaluator)."""
    dims = var.dimensions
    index = []
    for d in dims:
        if d == "sample":
            index.append(0)
        elif d == "ensemble_member":
            index.append(member_index)
        else:
            index.append(slice(None))
    arr = np.asarray(var[tuple(index)], dtype=np.float64)
    remaining = [d for d in dims if d not in ("sample", "ensemble_member")]
    if arr.ndim == 2 and remaining and remaining[0] == "weather_state":
        arr = arr.T
    return arr


class PredFile:
    """Lazy reader of one prediction file."""

    def __init__(self, path: Path):
        import netCDF4
        self.path = path
        self.ds = netCDF4.Dataset(str(path))
        self.ds.set_auto_mask(False)
        self.lat = np.asarray(self.ds.variables["lat_hres"][:], dtype=np.float64).reshape(-1)
        lon = np.asarray(self.ds.variables["lon_hres"][:], dtype=np.float64).reshape(-1)
        self.lon = ((lon + 180.0) % 360.0) - 180.0
        if "weather_state" in self.ds.variables:
            self.states = _decode_states(self.ds.variables["weather_state"])
        else:
            n = len(self.ds.dimensions["weather_state"])
            self.states = [str(i) for i in range(n)]
        if "ensemble_member" in self.ds.variables:
            self.members = [int(v) for v in np.asarray(self.ds.variables["ensemble_member"][:]).reshape(-1)]
        else:
            self.members = list(range(1, len(self.ds.dimensions["ensemble_member"]) + 1))

    def close(self):
        self.ds.close()

    def member_index(self, label: int) -> int:
        return self.members.index(label)

    def truth(self, member_label: int) -> np.ndarray:
        v = self.ds.variables["y"]
        mi = self.member_index(member_label) if "ensemble_member" in v.dimensions else 0
        return _read_member(v, mi)

    def interp(self, member_label: int) -> np.ndarray:
        v = self.ds.variables["x_interp"]
        mi = self.member_index(member_label) if "ensemble_member" in v.dimensions else 0
        return _read_member(v, mi)

    def pred(self, member_label: int) -> np.ndarray:
        return _read_member(self.ds.variables["y_pred"], self.member_index(member_label))

    def field(self, arr: np.ndarray, var: str) -> np.ndarray:
        spec = VAR_SPECS[var]
        cols = [arr[:, self.states.index(s)] for s in spec["states"]]
        if var == "ws":
            return np.hypot(cols[0], cols[1])
        out = cols[0]
        if var == "msl":
            out = out / 100.0 if np.nanmedian(out) > 2000.0 else out
        return out


# ---------------------------------------------------------------------------
# Regular-grid sampling
# ---------------------------------------------------------------------------

class Sampler:
    """Nearest-neighbour sampling of the unstructured points on a regular lat/lon grid.

    Built once per point set (all files of the lane share the box grid) and reused."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, extent, res: float, lat_scale: float = 1.2,
                 method: str = "nearest"):
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


def highpass(grid: np.ndarray, res_deg: float, cut_deg: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    sigma = 0.5 * cut_deg / res_deg
    return grid - gaussian_filter(grid, sigma=sigma, mode="nearest")


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

class Spectrum:
    """Isotropic 2-D power spectrum on a regular grid: mean removed, 2-D Hann window, FFT,
    azimuthal average of |F|^2 in log-spaced wavelength bins. Amplitude = sqrt(power)."""

    def __init__(self, ny: int, nx: int, dx_km: float, dy_km: float, nbins: int = 40):
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


SUMMARY_BANDS = [(15, 25), (25, 40), (40, 60), (60, 100), (100, 200), (200, 400), (400, 1000)]


def band_ratio(wl: np.ndarray, p_model: np.ndarray, p_truth: np.ndarray, lo: float, hi: float) -> float:
    sel = (wl >= lo) & (wl < hi)
    if not np.any(sel):
        return float("nan")
    return float(np.sqrt(np.sum(p_model[sel]) / np.sum(p_truth[sel])))


def distance_to_target(wl: np.ndarray, p_model: np.ndarray, p_truth: np.ndarray, lo: float, hi: float) -> float:
    """RMS over the log-spaced bins in [lo, hi) km of (amplitude ratio - 1), in percent."""
    sel = (wl >= lo) & (wl < hi)
    r = np.sqrt(p_model[sel] / p_truth[sel])
    return float(100.0 * np.sqrt(np.mean((r - 1.0) ** 2)))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _try_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def _axes_grid(nrows, ncols, extent, figsize):
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    if ccrs is not None:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": proj},
                                 squeeze=False)
        for ax in axes.ravel():
            ax.set_extent([extent[2], extent[3], extent[0], extent[1]], crs=proj)
            ax.coastlines(resolution="10m", linewidth=0.6, color="k")
            gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)
        transform = {"transform": proj}
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for ax in axes.ravel():
            ax.set_xlim(extent[2], extent[3])
            ax.set_ylim(extent[0], extent[1])
            ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (extent[0] + extent[1]))))
        transform = {}
    return fig, axes, transform


def _draw(ax, gx, gy, grid, cmap, vmin, vmax, transform):
    return ax.pcolormesh(gx, gy, grid, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", **transform)


def _sym_limit(*grids, pct=99.0) -> float:
    vals = np.concatenate([np.abs(g).ravel() for g in grids])
    v = float(np.percentile(vals, pct))
    return v if v > 0 else 1.0


def storm_centre(pf: PredFile, box, file_bounds) -> tuple[float, float, float] | None:
    """(lat, lon, msl_min hPa) of the truth's msl minimum inside box ∩ file bounds, or None."""
    lat_min = max(box[0], file_bounds[0])
    lat_max = min(box[1], file_bounds[1])
    lon_min = max(box[2], file_bounds[2])
    lon_max = min(box[3], file_bounds[3])
    m = (pf.lat >= lat_min) & (pf.lat <= lat_max) & (pf.lon >= lon_min) & (pf.lon <= lon_max)
    if not np.any(m):
        return None
    msl = pf.field(pf.truth(pf.members[0]), "msl")
    i = np.flatnonzero(m)[np.argmin(msl[m])]
    return float(pf.lat[i]), float(pf.lon[i]), float(msl[i])


def region_extent(centre_lat, centre_lon, file_bounds, half_lat=MAP_HALF_LAT):
    half_lon = half_lat / np.cos(np.deg2rad(centre_lat))
    lat_min = max(file_bounds[0], centre_lat - half_lat)
    lat_max = min(file_bounds[1], centre_lat + half_lat)
    lon_min = max(file_bounds[2], centre_lon - half_lon)
    lon_max = min(file_bounds[3], centre_lon + half_lon)
    return (lat_min, lat_max, lon_min, lon_max)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def plot_map_figure(out: Path, arms: dict[str, PredFile], truth_pf: PredFile, extent, region_name,
                    region_note, var, member, date, step, fine_cut_deg):
    import matplotlib.pyplot as plt
    spec = VAR_SPECS[var]
    smp = Sampler(truth_pf.lat, truth_pf.lon, extent, MAP_RES_DEG)
    truth = smp(truth_pf.field(truth_pf.truth(member), var))
    interp = smp(truth_pf.field(truth_pf.interp(member), var))
    preds = {name: smp(pf.field(pf.pred(member), var)) for name, pf in arms.items()}
    names = list(arms)
    ncols = 1 + len(names)

    fine = {"truth": highpass(truth, MAP_RES_DEG, fine_cut_deg)}
    for n in names:
        fine[n] = highpass(preds[n], MAP_RES_DEG, fine_cut_deg)
    fine_lim = _sym_limit(fine["truth"], *[fine[n] for n in names], pct=99.5)
    res_lim = _sym_limit(*[preds[n] - truth for n in names], truth - interp, pct=99.0)
    vmin, vmax = float(np.percentile(truth, 0.5)), float(np.percentile(truth, 99.5))
    if var in ("10u", "10v"):
        v = max(abs(vmin), abs(vmax))
        vmin, vmax = -v, v

    fig, axes, tr = _axes_grid(4, ncols, extent, figsize=(4.2 * ncols, 14.5))
    # row 1: fields
    ims = [_draw(axes[0, 0], smp.gx, smp.gy, truth, spec["cmap"], vmin, vmax, tr)]
    axes[0, 0].set_title(f"truth (o1280 analysis)", fontsize=10)
    for j, n in enumerate(names, 1):
        _draw(axes[0, j], smp.gx, smp.gy, preds[n], spec["cmap"], vmin, vmax, tr)
        axes[0, j].set_title(f"{n}, member {member}", fontsize=10)
    # row 2: fine part
    ims.append(_draw(axes[1, 0], smp.gx, smp.gy, fine["truth"], "RdBu_r", -fine_lim, fine_lim, tr))
    axes[1, 0].set_title(f"truth, fine part (< {fine_cut_deg:.1f} deg high-pass)", fontsize=10)
    for j, n in enumerate(names, 1):
        _draw(axes[1, j], smp.gx, smp.gy, fine[n], "RdBu_r", -fine_lim, fine_lim, tr)
        axes[1, j].set_title(f"{n}, fine part", fontsize=10)
    # row 3: minus truth (col 0 = interp field)
    _draw(axes[2, 0], smp.gx, smp.gy, interp, spec["cmap"], vmin, vmax, tr)
    axes[2, 0].set_title("input (o320) interpolated to o1280", fontsize=10)
    for j, n in enumerate(names, 1):
        ims.append(_draw(axes[2, j], smp.gx, smp.gy, preds[n] - truth, "RdBu_r", -res_lim, res_lim, tr))
        axes[2, j].set_title(f"{n} minus truth", fontsize=10)
    # row 4: minus interp
    _draw(axes[3, 0], smp.gx, smp.gy, truth - interp, "RdBu_r", -res_lim, res_lim, tr)
    axes[3, 0].set_title("truth minus interp (what the model must add)", fontsize=10)
    for j, n in enumerate(names, 1):
        _draw(axes[3, j], smp.gx, smp.gy, preds[n] - interp, "RdBu_r", -res_lim, res_lim, tr)
        axes[3, j].set_title(f"{n} minus interp", fontsize=10)

    cb_labels = [f"{spec['title']} [{spec['unit']}]", f"fine part [{spec['unit']}]",
                 f"difference [{spec['unit']}]"]
    for row, (im, lab) in zip((0, 1, 2), zip(ims, cb_labels)):
        cb = fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.02, pad=0.01)
        cb.set_label(lab, fontsize=9)
    fig.colorbar(ims[2], ax=axes[3, :].tolist(), fraction=0.02, pad=0.01).set_label(cb_labels[2], fontsize=9)
    fig.suptitle(f"{spec['title']}: {region_name} ({region_note}); base {date} lead {step} h, "
                 f"member {member}; box lat {extent[0]:.1f}..{extent[1]:.1f}, lon {extent[2]:.1f}..{extent[3]:.1f}; "
                 f"maps on a {MAP_RES_DEG:.2f} deg nearest-neighbour grid", fontsize=11)
    path = out / f"map_{date}_step{step:03d}_{region_name}_{var}_m{member}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_member_figure(out: Path, arms: dict[str, PredFile], truth_pf: PredFile, extent, region_name,
                       var, members, date, step, fine_cut_deg, fine: bool):
    import matplotlib.pyplot as plt
    spec = VAR_SPECS[var]
    smp = Sampler(truth_pf.lat, truth_pf.lon, extent, MAP_RES_DEG)
    truth = smp(truth_pf.field(truth_pf.truth(members[0]), var))
    names = list(arms)
    grids = {n: [smp(pf.field(pf.pred(m), var)) for m in members] for n, pf in arms.items()}
    if fine:
        truth = highpass(truth, MAP_RES_DEG, fine_cut_deg)
        grids = {n: [highpass(g, MAP_RES_DEG, fine_cut_deg) for g in gs] for n, gs in grids.items()}
        lim = _sym_limit(truth, *[g for gs in grids.values() for g in gs], pct=99.5)
        cmap, vmin, vmax = "RdBu_r", -lim, lim
    else:
        cmap = spec["cmap"]
        vmin, vmax = float(np.percentile(truth, 0.5)), float(np.percentile(truth, 99.5))
    ncols = 1 + len(members)
    fig, axes, tr = _axes_grid(len(names), ncols, extent, figsize=(4.2 * ncols, 3.8 * len(names)))
    for i, n in enumerate(names):
        im = _draw(axes[i, 0], smp.gx, smp.gy, truth, cmap, vmin, vmax, tr)
        axes[i, 0].set_title("truth" + (" (fine part)" if fine else ""), fontsize=10)
        for j, m in enumerate(members, 1):
            _draw(axes[i, j], smp.gx, smp.gy, grids[n][j - 1], cmap, vmin, vmax, tr)
            axes[i, j].set_title(f"{n}, member {m}" + (" (fine part)" if fine else ""), fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01).set_label(
        f"{spec['title']} [{spec['unit']}]", fontsize=9)
    what = f"fine part (< {fine_cut_deg:.1f} deg)" if fine else "field"
    fig.suptitle(f"{spec['title']} {what}: members {members} of each arm next to the truth; "
                 f"{region_name}, base {date} lead {step} h", fontsize=11)
    path = out / f"members_{date}_step{step:03d}_{region_name}_{var}_{'fine' if fine else 'field'}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def compute_spectra(arm_files: dict[str, dict[tuple[str, int], Path]], keys, members, variables,
                    max_members: int | None, method: str = "nearest"):
    """Mean power per wavelength bin over files x members: truth, interp, each arm, and the
    (arm - truth) and (interp - truth) differences. Returns (spectrum, {var: {name: power}}, n_draws)."""
    names = list(arm_files)
    acc: dict[str, dict[str, np.ndarray]] = {}
    smp = None
    spec = None
    n_draws = 0
    n_fields = 0
    for key in keys:
        pfs = {n: PredFile(arm_files[n][key]) for n in names}
        ref = pfs[names[0]]
        for n in names[1:]:
            if pfs[n].lat.shape != ref.lat.shape or np.abs(pfs[n].lat - ref.lat).max() > 1e-3:
                raise RuntimeError(f"{key}: point sets differ between {names[0]} and {n}")
        if smp is None:
            fb = (float(ref.lat.min()), float(ref.lat.max()), float(ref.lon.min()), float(ref.lon.max()))
            extent = (fb[0] + SPEC_EDGE_MARGIN_DEG, fb[1] - SPEC_EDGE_MARGIN_DEG,
                      fb[2] + SPEC_EDGE_MARGIN_DEG, fb[3] - SPEC_EDGE_MARGIN_DEG)
            smp = Sampler(ref.lat, ref.lon, extent, SPEC_RES_DEG, method=method)
            dx, dy = smp.dx_dy_km()
            spec = Spectrum(len(smp.gy), len(smp.gx), dx, dy)
            LOG.info("spectra: %s sampling, grid %d x %d at %.2f deg (dx %.2f km, dy %.2f km at lat %.1f), extent %s",
                     method, len(smp.gy), len(smp.gx), SPEC_RES_DEG, dx, dy, smp.lat0, extent)
        mems = [m for m in ref.members if (not members or m in members)]
        if max_members:
            mems = mems[:max_members]
        for m in mems:
            truths = {}
            interps = {}
            for v in variables:
                truths[v] = spec.fft(smp(ref.field(ref.truth(m), v)))
                interps[v] = spec.fft(smp(ref.field(ref.interp(m), v)))
            for n, pf in pfs.items():
                pred = pf.pred(m)
                for v in variables:
                    F = spec.fft(smp(pf.field(pred, v)))
                    a = acc.setdefault(v, {})
                    a[n] = a.get(n, 0.0) + spec.power(F)
                    a[f"{n}-truth"] = a.get(f"{n}-truth", 0.0) + spec.power(F - truths[v])
                    a[f"{n}xtruth"] = a.get(f"{n}xtruth", 0.0) + spec.cross(F, truths[v])
            for v in variables:
                a = acc.setdefault(v, {})
                a["truth"] = a.get("truth", 0.0) + spec.power(truths[v])
                a["interp"] = a.get("interp", 0.0) + spec.power(interps[v])
                a["interp-truth"] = a.get("interp-truth", 0.0) + spec.power(interps[v] - truths[v])
            n_draws += 1
        for pf in pfs.values():
            pf.close()
        LOG.info("spectra: %s done (%d members)", key, len(mems))
    for v in acc:
        for k in acc[v]:
            acc[v][k] = acc[v][k] / n_draws
    return spec, acc, n_draws


def plot_spectra(out: Path, spec: Spectrum, acc: dict, names: list[str], n_draws: int, keys, var):
    import matplotlib.pyplot as plt
    wl = spec.wavelength_km
    a = acc[var]
    vs = VAR_SPECS[var]
    colors = {"truth": "k", "interp": "0.55", "b0": "tab:red", "c0_30": "tab:blue", "n100_30": "tab:green"}
    def col(n):
        return colors.get(n, None)

    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)
    ax = axes[0]
    ax.loglog(wl, np.sqrt(a["truth"]), color="k", lw=2.2, label="truth (o1280 analysis)")
    ax.loglog(wl, np.sqrt(a["interp"]), color="0.55", lw=1.6, ls="--", label="input (o320) interpolated")
    for n in names:
        ax.loglog(wl, np.sqrt(a[n]), color=col(n), lw=1.6, label=n)
    ax.set_ylabel(f"AMPLITUDE spectrum [{vs['unit']}]\nsqrt(azimuthal mean power)")
    ax.set_title(f"{vs['title']}: amplitude spectra on the Atlantic box, mean over {n_draws} draws "
                 f"({len(keys)} files x members)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.axhline(1.0, color="k", lw=1)
    ax.axhspan(0.95, 1.05, color="0.9")
    ax.semilogx(wl, np.sqrt(a["interp"] / a["truth"]), color="0.55", lw=1.6, ls="--", label="interp / truth")
    for n in names:
        ax.semilogx(wl, np.sqrt(a[n] / a["truth"]), color=col(n), lw=1.8, label=f"{n} / truth")
    ax.set_ylim(0.0, 2.0)
    ax.set_ylabel("AMPLITUDE ratio to the truth\n(1 = truth; grey band +-5 %)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.axhline(1.0, color="k", lw=1)
    ax.axhline(np.sqrt(2.0), color="k", lw=0.8, ls=":")
    ax.semilogx(wl, np.sqrt(a["interp-truth"] / a["truth"]), color="0.55", lw=1.6, ls="--",
                label="(interp - truth) / truth")
    for n in names:
        ax.semilogx(wl, np.sqrt(a[f"{n}-truth"] / a["truth"]), color=col(n), lw=1.8, label=f"({n} - truth) / truth")
    ax.set_ylim(0.0, 2.0)
    ax.set_ylabel("AMPLITUDE of (arm - truth) / truth's\n(1 = error as large as the signal;\n"
                  "dotted sqrt 2 = right amount, random phase)")
    ax.set_xlabel("wavelength [km] (log axis; left = large scales, right = grid scale)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    for ax in axes:
        ax.invert_xaxis()
    for x in (400, 100, 40, 20):
        for ax in axes:
            ax.axvline(x, color="0.7", lw=0.6, ls=":")
    path = out / f"spectra_{var}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    with open(out / f"spectra_{var}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["wavelength_km", "band_lo_km", "band_hi_km", "amp_truth", "amp_interp"] + \
               [f"amp_{n}" for n in names] + [f"ratio_{n}" for n in names] + \
               [f"err_{n}_over_truth" for n in names] + [f"coherence_{n}" for n in names] + \
               ["ratio_interp", "err_interp_over_truth"]
        w.writerow(cols)
        for i in range(len(wl)):
            row = [f"{wl[i]:.2f}", f"{spec.band_lo_km[i]:.2f}", f"{spec.band_hi_km[i]:.2f}",
                   f"{np.sqrt(a['truth'][i]):.6g}", f"{np.sqrt(a['interp'][i]):.6g}"]
            row += [f"{np.sqrt(a[n][i]):.6g}" for n in names]
            row += [f"{np.sqrt(a[n][i] / a['truth'][i]):.4f}" for n in names]
            row += [f"{np.sqrt(a[f'{n}-truth'][i] / a['truth'][i]):.4f}" for n in names]
            row += [f"{a[f'{n}xtruth'][i] / np.sqrt(a[n][i] * a['truth'][i]):.4f}" for n in names]
            row += [f"{np.sqrt(a['interp'][i] / a['truth'][i]):.4f}",
                    f"{np.sqrt(a['interp-truth'][i] / a['truth'][i]):.4f}"]
            w.writerow(row)
    return path


def write_summary(out: Path, spec: Spectrum, acc: dict, names: list[str], n_draws: int, keys, variables,
                  method: str = "nearest"):
    wl = spec.wavelength_km
    lines = ["# Amplitude spectra summary (AMPLITUDE, not power; same operator on every field)", "",
             f"Draws: {n_draws} ({len(keys)} files: {', '.join(f'{d}/{s}' for d, s in keys)} x members).",
             f"Grid: {spec.ny} x {spec.nx} at {SPEC_RES_DEG} deg, {method} sampling; 2-D Hann window; "
             f"{wl.size} log-spaced wavelength bins {wl.min():.1f}..{wl.max():.0f} km.",
             "Band ratio = sqrt(sum of power in the band, arm / reference). Distance = RMS over the log bins in "
             "the range of (amplitude ratio - 1), in percent; 40-400 km is the campaign's band, 20-400 km is "
             "given for completeness (the o1280 analysis is truncated at about 31 km, so the ratios below "
             "31 km are ratios of near-zero amplitudes). The curves (PNG/CSV) are the record; the band numbers "
             "are a reading aid.", ""]
    bands = SUMMARY_BANDS
    refs = [("truth", "truth")] + ([("c0_30", "c0_30")] if "c0_30" in names else [])
    for v in variables:
        a = acc[v]
        lines.append(f"## {VAR_SPECS[v]['title']} ({v})")
        for ref_key, ref_name in refs:
            lines.append("")
            lines.append(f"Amplitude ratio to {ref_name}, per band [km], then distance [%]:")
            lines.append("")
            hdr = "| arm | " + " | ".join(f"{lo}-{hi}" for lo, hi in bands) + " | d 40-400 | d 20-400 |"
            lines.append(hdr)
            lines.append("|" + "---|" * (3 + len(bands)))
            rows = (["interp"] if ref_key == "truth" else []) + [n for n in names if n != ref_key]
            for n in rows:
                cells = [f"{band_ratio(wl, a[n], a[ref_key], lo, hi):.3f}" for lo, hi in bands]
                cells.append(f"{distance_to_target(wl, a[n], a[ref_key], 40, 400):.1f}")
                cells.append(f"{distance_to_target(wl, a[n], a[ref_key], 20, 400):.1f}")
                lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Coherence with the truth per band (cross-power / sqrt(power x power); 1 = in phase, 0 = random):")
        lines.append("")
        lines.append("| arm | " + " | ".join(f"{lo}-{hi}" for lo, hi in bands) + " |")
        lines.append("|" + "---|" * (1 + len(bands)))
        for n in names:
            cells = []
            for lo, hi in bands:
                sel = (wl >= lo) & (wl < hi)
                c = np.sum(a[f"{n}xtruth"][sel]) / np.sqrt(np.sum(a[n][sel]) * np.sum(a["truth"][sel]))
                cells.append(f"{c:.3f}")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_cases(text: str) -> list[tuple[str, int]]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        d, s = tok.split(":")
        out.append((d, int(s)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="name=root, repeatable (order = plot order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-cases", default="20230828:24,20230826:24,20230826:120",
                    help="date:lead pairs for the maps")
    ap.add_argument("--map-vars", default="ws,10u,2t,msl", help="variables of the first map case")
    ap.add_argument("--map-vars-other", default="ws", help="variables of the other map cases")
    ap.add_argument("--member", type=int, default=1, help="member label shown on the maps")
    ap.add_argument("--members-fig", default="1,2,3", help="member labels of the members/ figures")
    ap.add_argument("--fine-cut-deg", type=float, default=0.6)
    ap.add_argument("--spectra-leads", default="24,120")
    ap.add_argument("--spectra-dates", default="", help="comma list; empty = every date present in all arms")
    ap.add_argument("--spectra-vars", default="10u,10v,ws,2t,msl")
    ap.add_argument("--spectra-max-members", type=int, default=None)
    ap.add_argument("--spectra-method", default="linear", choices=["nearest", "linear"],
                    help="sampling of the O1280 points on the regular grid: nearest neighbour aliases the "
                         "grid-scale content of the octahedral rows into a beat near 30 km; linear "
                         "(barycentric) does not")
    ap.add_argument("--regions", default="", help="comma list of map regions to draw (default all)")
    ap.add_argument("--skip-maps", action="store_true")
    ap.add_argument("--skip-spectra", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import matplotlib
    matplotlib.use("Agg")

    arms: dict[str, Path] = {}
    for tok in args.arm:
        name, root = tok.split("=", 1)
        arms[name] = Path(root)
    out = Path(args.out)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    (out / "members").mkdir(parents=True, exist_ok=True)
    (out / "spectra").mkdir(parents=True, exist_ok=True)

    arm_files = {n: find_files(r) for n, r in arms.items()}
    manifest = {"arms": {n: str(r) for n, r in arms.items()}, "files": {}}
    for n, files in arm_files.items():
        LOG.info("%s: %d prediction files under %s", n, len(files), arms[n])
        manifest["files"][n] = {f"{d}_step{s:03d}": str(p) for (d, s), p in sorted(files.items())}
    first = next(iter(arm_files))
    leads_wanted = {int(x) for x in args.spectra_leads.split(",")}
    ref_keys = {k for k in arm_files[first] if k[1] in leads_wanted}
    for n in list(arm_files):
        missing = sorted(ref_keys - set(arm_files[n]))
        if missing and n != first:
            LOG.warning("%s lacks %d of the %s (date, lead) files (%s); arm DROPPED", n, len(missing), first, missing)
            del arm_files[n]
            del arms[n]
    common = set.intersection(*[set(f) for f in arm_files.values()])
    if not common:
        LOG.error("no (date, lead) common to every arm")
        return 2

    t0 = time.time()
    if not args.skip_maps:
        cases = parse_cases(args.map_cases)
        for ci, (date, step) in enumerate(cases):
            key = (date, step)
            if key not in common:
                LOG.warning("map case %s/%d missing in one of the arms; skipped", date, step)
                continue
            pfs = {n: PredFile(arm_files[n][key]) for n in arms}
            ref = next(iter(pfs.values()))
            fb = (float(ref.lat.min()), float(ref.lat.max()), float(ref.lon.min()), float(ref.lon.max()))
            regions = {}
            for rname, box in EVENT_BOXES.items():
                c = storm_centre(ref, box, fb)
                if c is None:
                    LOG.warning("%s: event box %s outside the file; skipped", key, rname)
                    continue
                regions[rname] = (region_extent(c[0], c[1], fb),
                                  f"truth msl min {c[2]:.0f} hPa at {c[0]:.1f}N {abs(c[1]):.1f}W")
            for qname, qbox in QUIET_BOXES.items():
                lat_min, lat_max = max(qbox[0], fb[0]), min(qbox[1], fb[1])
                lon_min, lon_max = max(qbox[2], fb[2]), min(qbox[3], fb[3])
                if lat_max - lat_min < 0.5 * (qbox[1] - qbox[0]) or lon_max - lon_min < 0.5 * (qbox[3] - qbox[2]):
                    LOG.warning("%s: quiet box %s mostly outside the file bounds %s; skipped", key, qname, fb)
                    continue
                regions[qname] = ((lat_min, lat_max, lon_min, lon_max), "fixed box away from both storms")
            if args.regions:
                wanted = [r.strip() for r in args.regions.split(",")]
                regions = {k: v for k, v in regions.items() if k in wanted}
            variables = (args.map_vars if ci == 0 else args.map_vars_other).split(",")
            for rname, (extent, note) in regions.items():
                for v in variables:
                    p = plot_map_figure(out / "maps", pfs, ref, extent, rname, note, v.strip(),
                                        args.member, date, step, args.fine_cut_deg)
                    LOG.info("wrote %s", p)
            if ci == 0:
                mems = [int(x) for x in args.members_fig.split(",")]
                for rname in regions:
                    if rname not in regions:
                        continue
                    for fine in (False, True):
                        p = plot_member_figure(out / "members", pfs, ref, regions[rname][0], rname, "ws",
                                               mems, date, step, args.fine_cut_deg, fine)
                        LOG.info("wrote %s", p)
            for pf in pfs.values():
                pf.close()
        LOG.info("maps done in %.0f s", time.time() - t0)

    if not args.skip_spectra:
        leads = [int(x) for x in args.spectra_leads.split(",")]
        dates = [d for d in args.spectra_dates.split(",") if d]
        keys = sorted(k for k in common if k[1] in leads and (not dates or k[0] in dates))
        variables = [v.strip() for v in args.spectra_vars.split(",")]
        LOG.info("spectra over %d files: %s", len(keys), keys)
        spec, acc, n_draws = compute_spectra(arm_files, keys, None, variables, args.spectra_max_members,
                                             method=args.spectra_method)
        for v in variables:
            p = plot_spectra(out / "spectra", spec, acc, list(arms), n_draws, keys, v)
            LOG.info("wrote %s", p)
        write_summary(out / "spectra", spec, acc, list(arms), n_draws, keys, variables, args.spectra_method)
        manifest["spectra"] = {"files": [f"{d}_step{s:03d}" for d, s in keys], "draws": n_draws,
                               "grid": [spec.ny, spec.nx], "res_deg": SPEC_RES_DEG, "method": args.spectra_method}
        LOG.info("spectra done in %.0f s", time.time() - t0)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("all done in %.0f s; outputs under %s", time.time() - t0, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
