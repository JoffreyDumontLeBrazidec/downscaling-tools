"""storm_maps core — regional storm maps + full radial power spectra.

Self-contained (numpy/scipy/xarray/matplotlib). For a downscaling eval run it renders,
on TOP of the usual regional plots:

  1. <out>/storm_maps.png (+ .pdf) : 10 m wind speed + MSL pressure maps (coastlines, borders,
                                  labelled grid lines), TRUTH vs MODEL vs INPUT, zoomed on the
                                  deepest-eye storm instance (one shared colour scale per row).
  2. <out>/full_spectra.png (+ .pdf) : full radial power spectrum at ALL wavenumbers for 10u/10v/msl,
                                  model vs truth vs input, with the 40-150 km fine band shaded (off the 16.7 km Nyquist).
  3. <out>/storm_maps_spectra.json : fine-band (40-150 km, off-Nyquist) power ratio to truth + log-log slope.

Method (identical to the T24 regional box-FFT audit, tc_o320_o1280):
  native O1280 box points -> nearest-neighbour onto a regular GRID_DEG grid (2 deg rim removed)
  -> per field: linear-plane detrend + 2D Hann window -> np.fft.rfft2 power -> isotropic radial
  binning into log-spaced wavenumber bins -> PER-MEMBER spectra averaged (never the ensemble mean).
  Truth = each member's target `y`; input = `x_interp`. Windowed box-FFT powers are the model/truth
  ratio under byte-identical processing only (NOT comparable to global healpix C_l boards).

Figures are drawn in the house style of ``eval.plotting`` (role colours, variable-table colour
maps and units) by ``plot_full_spectra`` and ``plot_storm_maps``, which only draw.

CLI:  python -m eval.evaluators.storm_maps.core.render <predictions_dir> --out <dir> \
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
    """Windowed 2D-FFT radial power spectra over a lat/lon box (interior, 2 deg rim removed).

    GUARD (2026-07-13): the reported fine band is 40-150 km, deliberately OFF the 16.7 km
    Nyquist (2*grid). The former 20-100 km band integrated a grid-scale noise floor that the
    UNIFIED multi-GPU edge-sharded runtime inflates ~3.5-4.2x (a runtime artifact, NOT model
    over-power). Score ALL arms in the SAME pristine (fp32, single-tile/local-graph) runtime;
    never mix unified. See tc-o320-o1280 20260711_lane_soundness_audit.md CORRECTION."""

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

    def slope(self, spec, lo_km=40.0, hi_km=150.0):
        m = (self.kmid >= 1.0 / hi_km) & (self.kmid <= 1.0 / lo_km) & (spec > 0)
        if m.sum() < 3:
            return float("nan")
        return float(np.polyfit(np.log10(self.kmid[m]), np.log10(spec[m]), 1)[0])

    def fine_ratio(self, spec_m, spec_t):
        # 40-150 km "clean" mesoscale band, deliberately OFF the 16.7 km Nyquist (2*grid).
        # The former 20-100 km band reached to 1.2x Nyquist and integrated a grid-scale noise
        # floor; the unified multi-GPU runtime inflates that floor to a spurious 3.5-4.2x
        # (tc-o320-o1280 20260711_lane_soundness_audit.md CORRECTION). Score arms PRISTINE-only.
        fine = (self.wl >= 40) & (self.wl <= 150)
        return float(np.nansum(spec_m[fine]) / np.nansum(spec_t[fine]))


def _ogrid(npoints: int) -> str | None:
    """Name of the octahedral reduced Gaussian grid with ``npoints`` points (4N^2 + 36N), if any."""
    n = int(round((-36.0 + np.sqrt(1296.0 + 16.0 * float(npoints))) / 8.0))
    return f"O{n}" if n > 0 and 4 * n * n + 36 * n == int(npoints) else None


def _column_titles(grids: dict | None) -> dict:
    """Panel titles for truth / model / input, naming their grids when they are known."""
    g = grids or {}
    hres, lres = g.get("hres"), g.get("lres")
    on = f" ({hres})" if hres else ""
    if lres and hres:
        inp = f"Input ({lres} interpolated to {hres})"
    else:
        inp = "Input (interpolated to the target grid)"
    return {"truth": f"Truth{on}", "model": f"Model{on}", "input": inp}


def plot_full_spectra(out_dir, wavelengths_km, spec: dict, jout: dict, *, run_name: str = "",
                      step: str = "", grid_shape=None, max_nn_km: float | None = None,
                      n_curves: int | None = None, grids: dict | None = None):
    """``full_spectra.png`` (+ ``.pdf``): radial box-FFT power spectra per field, roles styled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.plotting import AXIS, eval_style, role_style, save_figure, variable_spec

    titles = _column_titles(grids)
    labels = {"truth": titles["truth"].replace("Truth", "truth"),
              "model": titles["model"].replace("Model", "model"),
              "input": titles["input"].replace("Input", "input")}
    nlab = f" (n = {n_curves} member fields)" if n_curves else ""
    with eval_style():
        fig, axes = plt.subplots(1, len(FIELDS), figsize=(16.5, 5.6), layout="constrained")
        for ax, f in zip(np.atleast_1d(axes), FIELDS):
            ax.axvspan(40, 150, color="0.88", zorder=0, lw=0,
                       label="40-150 km band of the fine-band ratio")
            for c in ("truth", "model", "input"):
                ax.loglog(wavelengths_km, spec[f][c], label=labels[c] + nlab, **role_style(c))
            ax.invert_xaxis()
            ax.grid(which="both", alpha=.3)
            ratio = jout["fine_band_40_150km_ratio_to_truth"][f]
            sl = jout["slope_fine_40_150km"][f]
            ax.set_title(f"{variable_spec(f).name}\nfine-band power model / truth {ratio:.2f}; "
                         f"slope model {sl['model']:.2f}, truth {sl['truth']:.2f}", fontsize=10)
            ax.set_xlabel(AXIS["wavelength"] + ", large scales left, fine scales right")
            if f == FIELDS[0]:
                ax.set_ylabel("Spectral power, mean over members (arbitrary units)")
                ax.legend(fontsize=8, loc="lower left")
        where = f"{run_name}, " if run_name else ""
        shape = f"regular {grid_shape[0]} x {grid_shape[1]} grid of {GRID_DEG} degrees" if grid_shape else ""
        nn = f", largest nearest-neighbour distance {max_nn_km:.1f} km" if max_nn_km is not None else ""
        fig.suptitle(f"Full radial power spectra in the storm box, {where}{'lead time ' + str(int(step)) + ' h' if str(step).isdigit() else 'step ' + str(step)}\n"
                     f"box FFT on a {shape}{nn}; per-member spectra averaged", fontsize=11)
        save_figure(fig, Path(out_dir) / "full_spectra.png", close=True)


def plot_storm_maps(out_dir, lon_grid, lat_grid, fields: dict, *, eye_lat: float, eye_lon: float,
                    event_name: str = "storm", file_name: str = "", member: int = 0,
                    deepest_hpa: float | None = None, wind_max: float, msl_min: float,
                    msl_max: float, grids: dict | None = None):
    """``storm_maps.png`` (+ ``.pdf``): wind speed and MSL pressure, truth / model / input.

    ``fields`` maps ``truth``/``model``/``input`` to ``(wind_ms, msl_hpa)`` 2-D arrays on the
    regular ``lat_grid`` x ``lon_grid`` (1-D, degrees). Each row shares one colour scale; the
    limits are the ones the caller computed from the truth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from eval.plotting import (add_row_colorbar, axis_label, eval_style, extend_for, map_grid,
                               save_figure, variable_spec)

    order = ("truth", "model", "input")
    titles = _column_titles(grids)
    extent = (float(np.min(lon_grid)), float(np.max(lon_grid)),
              float(np.min(lat_grid)), float(np.max(lat_grid)))
    rows = (("10ff", 0, Normalize(vmin=0.0, vmax=wind_max)),
            ("msl", 1, Normalize(vmin=msl_min, vmax=msl_max)))
    pc = ccrs.PlateCarree()
    with eval_style():
        fig, ax = map_grid(2, 3, extent, panel_size=(4.3, 4.0), resolution="50m")
        ax[0, 0].get_gridspec().update(hspace=0.34)   # room for the second row's two-line titles
        for var, k, norm in rows:
            spec = variable_spec(var)
            mesh = None
            for j, c in enumerate(order):
                a = ax[k, j]
                mesh = a.pcolormesh(lon_grid, lat_grid, fields[c][k], cmap=spec.field_cmap(),
                                    norm=norm, shading="nearest", transform=pc, rasterized=True)
                a.plot(eye_lon, eye_lat, "+", color="k", ms=11, mew=1.8, transform=pc, zorder=7)
                a.set_title(f"{titles[c]}\n{spec.name}", fontsize=10)
            add_row_colorbar(fig, mesh, ax[k, :], axis_label(var),
                             extend=extend_for(norm, *[fields[c][k] for c in order]))
        deep = f", truth minimum {deepest_hpa:.1f} hPa" if deepest_hpa is not None else ""
        fig.suptitle(f"Storm maps, {event_name}: {file_name}, member {member + 1}{deep}\n"
                     "member and time where the truth storm is deepest; one colour scale per row; "
                     "+ marks the truth pressure minimum", fontsize=11)
        save_figure(fig, Path(out_dir) / "storm_maps.png", close=True)


def _open(nc):
    import xarray as xr
    return xr.open_dataset(nc, decode_timedelta=False)


def render(predictions_dir, out_dir, event_box=(5, 35, -100, -40), event_name="storm",
           step="072", storm_box=None):
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
    nmem = int(d0.sizes["ensemble_member"])
    grids = {"hres": d0.attrs.get("grid") or _ogrid(d0.sizes.get("grid_point_hres", lat.size)),
             "lres": _ogrid(d0.sizes["grid_point_lres"]) if "grid_point_lres" in d0.sizes else None}
    d0.close()
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
    jout = {"fine_band_40_150km_ratio_to_truth": {}, "slope_fine_40_150km": {}, "storm_box_min_msl_hpa": round(deepest[0] / 100.0, 1)}
    for f in FIELDS:
        jout["fine_band_40_150km_ratio_to_truth"][f] = round(bs.fine_ratio(spec[f]["model"], spec[f]["truth"]), 3)
        jout["slope_fine_40_150km"][f] = {c: round(bs.slope(spec[f][c]), 3) for c in ("model", "truth", "input")}
    (out_dir / "storm_maps_spectra.json").write_text(json.dumps(jout, indent=2))

    plot_full_spectra(out_dir, bs.wl, spec, jout, run_name=predictions_dir.parent.name, step=step,
                      grid_shape=(bs.ny, bs.nx), max_nn_km=bs.max_nn_km,
                      n_curves=len(acc[FIELDS[0]]["model"]), grids=grids)

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
    plot_storm_maps(out_dir, lo, la, {c: (wind(c), mslf(c)) for c in ("truth", "model", "input")},
                    eye_lat=float(elat), eye_lon=float(elon), event_name=event_name,
                    file_name=nc.name, member=mem, deepest_hpa=deepest[0] / 100.0,
                    wind_max=float(wmax), msl_min=float(mmin), msl_max=float(mmax), grids=grids)
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
