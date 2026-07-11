"""storm_maps backend — regional storm maps + full radial power spectra.

Self-contained (numpy/scipy/xarray/matplotlib). For a downscaling eval run it renders,
on TOP of the usual regional plots:

  1. <out>/storm_maps.png       : 10 m wind + MSL fields, TRUTH vs MODEL vs INPUT, zoomed on
                                  the deepest-eye storm instance (same colour scale per row).
  2. <out>/full_spectra.png     : full radial power spectrum at ALL wavenumbers for 10u/10v/msl,
                                  model vs truth vs input, with the 20-100 km fine band shaded.
  3. <out>/storm_maps_spectra.json : fine-band (20-100 km) power ratio to truth + log-log slope.

Method (identical to the T24 regional box-FFT audit, tc_o320_o1280):
  native O1280 box points -> nearest-neighbour onto a regular GRID_DEG grid (2 deg rim removed)
  -> per field: linear-plane detrend + 2D Hann window -> np.fft.rfft2 power -> isotropic radial
  binning into log-spaced wavenumber bins -> PER-MEMBER spectra averaged (never the ensemble mean).
  Truth = each member's target `y`; input = `x_interp`. Windowed box-FFT powers are the model/truth
  ratio under byte-identical processing only (NOT comparable to global healpix C_l boards).

CLI:  python -m eval._backends.storm_maps.render <predictions_dir> --out <dir> \
        [--event-box lat0,lat1,lon0,lon1] [--event-name idalia] [--step 072]
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np

LOG = logging.getLogger(__name__)
FIELDS = ("10u", "10v", "msl")
GRID_DEG = 0.075
RIM_DEG = 2.0
MEAN_LAT = 20.0


def _xyz(lat, lon):
    a, b = np.deg2rad(lat), np.deg2rad(lon)
    return np.c_[np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)]


class BoxSpectra:
    """Windowed 2D-FFT radial power spectra over a lat/lon box (interior, 2 deg rim removed)."""

    def __init__(self, lat, lon, box):
        from scipy.spatial import cKDTree
        lat0, lat1, lon0, lon1 = box
        self.glat = np.arange(lat0 + RIM_DEG, lat1 - RIM_DEG + 1e-9, GRID_DEG)
        self.glon = np.arange(lon0 + RIM_DEG, lon1 - RIM_DEG + 1e-9, GRID_DEG)
        self.ny, self.nx = len(self.glat), len(self.glon)
        gg_lat, gg_lon = np.meshgrid(self.glat, self.glon, indexing="ij")
        d, idx = cKDTree(_xyz(lat, lon)).query(_xyz(gg_lat.ravel(), gg_lon.ravel()), k=1)
        self.idx = idx
        self.max_nn_km = float(2.0 * 6371.0 * np.arcsin(d.max() / 2.0))
        self.win = np.hanning(self.ny)[:, None] * np.hanning(self.nx)[None, :]
        self.DY = GRID_DEG * 111.195
        self.DX = self.DY * np.cos(np.deg2rad(MEAN_LAT))
        ky = np.fft.fftfreq(self.ny, d=self.DY)
        kx = np.fft.rfftfreq(self.nx, d=self.DX)
        kk = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
        kmin, kmax = 1.0 / 2000.0, kk.max()
        self.kbins = np.logspace(np.log10(kmin), np.log10(kmax), 40)
        self.kmid = np.sqrt(self.kbins[1:] * self.kbins[:-1])
        self.wl = 1.0 / self.kmid
        self.kbin_idx = np.digitize(kk.ravel(), self.kbins)
        self._detrend_A = None

    def power(self, vals_pts):
        f = vals_pts[self.idx].reshape(self.ny, self.nx).astype(np.float64)
        yy, xx = np.mgrid[0:self.ny, 0:self.nx]
        A = np.c_[xx.ravel(), yy.ravel(), np.ones(self.ny * self.nx)]
        c, *_ = np.linalg.lstsq(A, f.ravel(), rcond=None)
        f = f - (A @ c).reshape(self.ny, self.nx)
        F = np.fft.rfft2(f * self.win)
        P = (F.real ** 2 + F.imag ** 2)
        P[:, 1:-1] *= 2.0
        return P

    def spectrum_1d(self, P):
        s = np.bincount(self.kbin_idx, weights=P.ravel(), minlength=len(self.kbins) + 1)
        n = np.bincount(self.kbin_idx, minlength=len(self.kbins) + 1)
        avg = np.zeros(len(self.kmid))
        for i in range(len(self.kmid)):
            avg[i] = s[i + 1] / n[i + 1] if n[i + 1] else np.nan
        return avg

    def slope(self, spec, lo_km=20.0, hi_km=100.0):
        m = (self.kmid >= 1.0 / hi_km) & (self.kmid <= 1.0 / lo_km) & (spec > 0)
        if m.sum() < 3:
            return float("nan")
        return float(np.polyfit(np.log10(self.kmid[m]), np.log10(spec[m]), 1)[0])

    def fine_ratio(self, spec_m, spec_t):
        fine = (self.wl >= 20) & (self.wl <= 100)
        return float(np.nansum(spec_m[fine]) / np.nansum(spec_t[fine]))


def _open(nc):
    import xarray as xr
    return xr.open_dataset(nc, decode_timedelta=False)


def render(predictions_dir, out_dir, event_box=(5, 35, -100, -40), event_name="storm",
           step="072", storm_box=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    predictions_dir = Path(predictions_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ncs = sorted(predictions_dir.glob(f"predictions_*_step{step}.nc"))
    if not ncs:
        ncs = sorted(predictions_dir.glob("predictions_*.nc"))
    if not ncs:
        raise FileNotFoundError(f"no predictions in {predictions_dir}")
    storm_box = storm_box or event_box

    d0 = _open(ncs[0])
    lat = d0.lat_hres.values.astype(np.float64); lon = d0.lon_hres.values.astype(np.float64)
    ws = list(d0.weather_state.values); si = {f: ws.index(f) for f in FIELDS}
    nmem = int(d0.sizes["ensemble_member"]); d0.close()
    box_mask = ((lat >= event_box[0] - RIM_DEG) & (lat <= event_box[1] + RIM_DEG) &
                (lon >= event_box[2] - RIM_DEG) & (lon <= event_box[3] + RIM_DEG))
    bidx = np.where(box_mask)[0]
    bs = BoxSpectra(lat[bidx], lon[bidx], event_box)
    LOG.info("storm_maps grid %dx%d maxNN %.1fkm boxpts %d", bs.ny, bs.nx, bs.max_nn_km, bidx.size)

    # --- spectra: per-member box-FFT over all instances, averaged (N = ndates*nmem) ---
    acc = {f: {c: [] for c in ("model", "truth", "input")} for f in FIELDS}
    deepest = None  # (msl, nc, member)
    for nc in ncs:
        ds = _open(nc)
        raw = {"model": ds.y_pred.isel(sample=0).values,
               "truth": ds.y.isel(sample=0).values,
               "input": ds.x_interp.isel(sample=0).values}
        ds.close()
        for f in FIELDS:
            for c in ("model", "truth", "input"):
                v = raw[c][:, bidx, si[f]]
                for mem in range(nmem):
                    acc[f][c].append(bs.spectrum_1d(bs.power(v[mem])))
        # storm search on truth msl within storm_box
        sm = ((lat[bidx] >= storm_box[0]) & (lat[bidx] <= storm_box[1]) &
              (lon[bidx] >= storm_box[2]) & (lon[bidx] <= storm_box[3]))
        tmsl = raw["truth"][:, bidx, si["msl"]]
        mm = np.where(sm[None, :], tmsl, 1e12).min(axis=1)
        mem = int(np.argmin(mm))
        if deepest is None or mm[mem] < deepest[0]:
            deepest = (float(mm[mem]), nc, mem)

    spec = {f: {c: np.nanmean(np.array(acc[f][c]), axis=0) for c in acc[f]} for f in FIELDS}
    jout = {"fine_band_20_100km_ratio_to_truth": {}, "slope_fine_20_100km": {}, "storm_box_min_msl_hpa": round(deepest[0] / 100.0, 1)}
    for f in FIELDS:
        jout["fine_band_20_100km_ratio_to_truth"][f] = round(bs.fine_ratio(spec[f]["model"], spec[f]["truth"]), 3)
        jout["slope_fine_20_100km"][f] = {c: round(bs.slope(spec[f][c]), 3) for c in ("model", "truth", "input")}
    (out_dir / "storm_maps_spectra.json").write_text(json.dumps(jout, indent=2))

    col = {"model": "#1a73e8", "truth": "#2e7d32", "input": "#9aa0a6"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=130)
    for ax, f in zip(axes, FIELDS):
        for c, lab in (("truth", "truth (target y)"), ("model", "model"), ("input", "input (interp)")):
            ax.loglog(bs.wl, spec[f][c], color=col[c], lw=2.4 if c == "truth" else 1.8,
                      ls=":" if c == "input" else "-", label=lab)
        ax.axvspan(20, 100, color="#f1c40f", alpha=.13)
        ax.invert_xaxis(); ax.grid(which="both", alpha=.18)
        ax.set_title(f"{f}  (fine ratio {jout['fine_band_20_100km_ratio_to_truth'][f]:.2f}, "
                     f"slope {jout['slope_fine_20_100km'][f]['model']:.2f} vs truth {jout['slope_fine_20_100km'][f]['truth']:.2f})",
                     fontsize=9.5)
        ax.set_xlabel("wavelength (km) — large ← → fine")
        if f == "10u":
            ax.set_ylabel("radial power (per-member avg)"); ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Full radial power spectra — {predictions_dir.parent.name} (box-FFT, step {step}, "
                 f"grid {bs.ny}x{bs.nx}, maxNN {bs.max_nn_km:.1f}km)", fontsize=10.5, y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "full_spectra.png", bbox_inches="tight"); plt.close(fig)

    # --- maps: deepest storm instance, wind + msl, truth vs model vs input ---
    from scipy.spatial import cKDTree
    _msl, nc, mem = deepest
    ds = _open(nc)
    yy = {"truth": ds.y.isel(sample=0).values, "model": ds.y_pred.isel(sample=0).values,
          "input": ds.x_interp.isel(sample=0).values}
    ds.close()
    tmsl = yy["truth"][mem, bidx, si["msl"]]
    sm = ((lat[bidx] >= storm_box[0]) & (lat[bidx] <= storm_box[1]) &
          (lon[bidx] >= storm_box[2]) & (lon[bidx] <= storm_box[3]))
    eloc = np.where(sm, tmsl, 1e12).argmin(); elat, elon = lat[bidx][eloc], lon[bidx][eloc]
    G, HALF = 0.06, 9.0
    la = np.arange(max(elat - HALF, event_box[0]), min(elat + HALF, event_box[1]) + 1e-9, G)
    lo = np.arange(max(elon - HALF, event_box[2]), min(elon + HALF, event_box[3]) + 1e-9, G)
    GLO, GLA = np.meshgrid(lo, la)
    _, midx = cKDTree(_xyz(lat[bidx], lon[bidx])).query(_xyz(GLA.ravel(), GLO.ravel()), k=1)

    def wind(c):
        return np.sqrt(yy[c][mem, bidx, si["10u"]] ** 2 + yy[c][mem, bidx, si["10v"]] ** 2)[midx].reshape(GLA.shape)

    def mslf(c):
        return (yy[c][mem, bidx, si["msl"]][midx].reshape(GLA.shape)) / 100.0

    wmax = np.nanpercentile(wind("truth"), 99.7)
    mmin, mmax = np.nanmin(mslf("truth")), np.nanpercentile(mslf("truth"), 98)
    ext = [lo.min(), lo.max(), la.min(), la.max()]
    order = [("truth", "TRUTH (target)"), ("model", "MODEL"), ("input", "input (interp)")]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.2), dpi=130)
    for j, (c, lab) in enumerate(order):
        im0 = ax[0, j].imshow(wind(c), origin="lower", extent=ext, cmap="turbo", vmin=0, vmax=wmax, aspect="auto")
        im1 = ax[1, j].imshow(mslf(c), origin="lower", extent=ext, cmap="viridis", vmin=mmin, vmax=mmax, aspect="auto")
        ax[0, j].set_title(lab, fontsize=11)
        for a in (ax[0, j], ax[1, j]):
            a.plot(elon, elat, "k+", ms=9, mew=1.4); a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("10 m wind (m/s)", fontsize=11); ax[1, 0].set_ylabel("MSL (hPa)", fontsize=11)
    fig.colorbar(im0, ax=ax[0, :].tolist(), fraction=0.013, pad=0.01, label="m/s")
    fig.colorbar(im1, ax=ax[1, :].tolist(), fraction=0.013, pad=0.01, label="hPa")
    fig.suptitle(f"Storm maps — {event_name}, {nc.name}, member {mem + 1} (truth-deepest, "
                 f"{deepest[0] / 100.0:.1f} hPa) · same colour scale per row · '+' = truth eye", fontsize=10.5, y=0.99)
    fig.savefig(out_dir / "storm_maps.png", bbox_inches="tight"); plt.close(fig)
    LOG.info("storm_maps wrote %s", out_dir)
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regional storm maps + full spectra from an eval run")
    ap.add_argument("predictions_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--event-box", default="5,35,-100,-40", help="lat0,lat1,lon0,lon1")
    ap.add_argument("--storm-box", default="10,35,-100,-80", help="lat0,lat1,lon0,lon1 for the storm search")
    ap.add_argument("--event-name", default="storm")
    ap.add_argument("--step", default="072")
    a = ap.parse_args(argv)
    box = tuple(float(x) for x in a.event_box.split(","))
    sbox = tuple(float(x) for x in a.storm_box.split(","))
    out = a.out or str(Path(a.predictions_dir).parent / "evaluators" / "storm_maps")
    logging.basicConfig(level=logging.INFO)
    print(render(a.predictions_dir, out, event_box=box, event_name=a.event_name, step=a.step, storm_box=sbox))


if __name__ == "__main__":
    main()
