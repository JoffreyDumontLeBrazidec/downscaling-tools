#!/usr/bin/env python3
"""Plot intense precipitation events from downscaling prediction files.

Selection is delegated to eval._backends.region_plotting.precip_events
(find_precip_events), so the pages always match the evaluator's events.json.

Each event page shows, zoomed tightly around the event centre:
  truth | interp input | prediction | prediction - truth

Truth and the interp-input baseline fall back to the lane's GRIB sources when
the predictions do not embed them (tp truth was historically missing from the
o1280->o2560 bundles, and x_interp tp is identically zero there because tp is
an output-only channel). All colourbars are mm per 6h window.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

from eval._backends.precip.sources import (
    LresInterpBaseline,
    PrecipTruthSource,
    is_degenerate_channel,
)
from .precip_events import Event, find_precip_events

PRECIP_VARS = ("tp", "cp")
DEFAULT_DLAT = 2.0
DEFAULT_DLON = 2.5
DEFAULT_N_TOP = 3

try:
    from anemoi.training.diagnostics.maps import Coastlines

    COASTLINES = Coastlines()
except Exception:  # pragma: no cover - coastlines are a presentation garnish
    COASTLINES = None


def _add_coastlines(ax) -> None:
    if COASTLINES is None:
        return
    coast_segs_deg = [np.degrees(s) for s in COASTLINES.lines.get_segments()]
    ax.add_collection(LineCollection(coast_segs_deg, linewidths=0.7,
                                     colors="black", zorder=10))


def _zoom_mask(lat: np.ndarray, lon: np.ndarray, clat: float, clon: float,
               dlat: float, dlon: float) -> np.ndarray:
    return (
        (lat >= clat - dlat) & (lat <= clat + dlat)
        & (lon >= clon - dlon) & (lon <= clon + dlon)
    )


def _format_map(ax, *, clat: float, clon: float, dlat: float, dlon: float) -> None:
    ax.set_xlim(clon - dlon, clon + dlon)
    ax.set_ylim(clat - dlat, clat + dlat)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("auto", adjustable=None)
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.4)
    _add_coastlines(ax)


def _robust_limits(*arrays) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays
                           if a is not None and a.size])
    vals = vals[vals >= 0]
    if vals.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.nanpercentile(vals, 99.7)), 1.0)


def _error_limit(error: np.ndarray) -> float:
    vals = np.abs(error[np.isfinite(error)])
    if vals.size == 0:
        return 1.0
    return max(float(np.nanpercentile(vals, 99.0)), 1.0)


def _member_id(ds: xr.Dataset, mi: int) -> int:
    raw = str(ds.attrs.get("member_ids", ""))
    if raw:
        try:
            return [int(x) for x in raw.split(",")][mi]
        except (ValueError, IndexError):
            pass
    return mi + 1


class _EventData:
    """Per-event field slices (mm), resolved from NC + fallback GRIB sources."""

    def __init__(self, truth_grib_tpl: str, baseline_grib_tpl: str,
                 interp_index_cache: str, var: str, member_index: int):
        self.truth_grib_tpl = truth_grib_tpl
        self.baseline_grib_tpl = baseline_grib_tpl
        self.interp_index_cache = interp_index_cache
        self.var = var
        self.mi = member_index
        self._truth_src: PrecipTruthSource | None = None
        self._baseline_src: LresInterpBaseline | None = None

    def load(self, event: Event):
        ds = xr.open_dataset(event.nc_path)
        try:
            ws = [str(s) for s in ds["weather_state"].values]
            vi = ws.index(self.var)
            lat = ds["lat_hres"].values
            lon = ds["lon_hres"].values
            pred = ds["y_pred"][0, self.mi].values[:, vi] * 1000.0

            truth = ds["y"][0, self.mi].values[:, vi]
            if np.isfinite(truth).mean() < 0.99 and self.truth_grib_tpl:
                if self._truth_src is None:
                    self._truth_src = PrecipTruthSource(self.truth_grib_tpl,
                                                        var=self.var)
                truth = self._truth_src.load(event.date, event.step)
                self._truth_src.verify_grid(lat, lon)
            truth = truth * 1000.0 if np.isfinite(truth).mean() > 0.5 else None

            base = None
            if "x_interp" in ds.variables:
                cand = ds["x_interp"][0, self.mi].values[:, vi]
                if not is_degenerate_channel(cand):
                    base = cand * 1000.0
            if base is None and self.baseline_grib_tpl:
                if self._baseline_src is None:
                    self._baseline_src = LresInterpBaseline(
                        self.baseline_grib_tpl, self.interp_index_cache or None,
                        var=self.var)
                    self._baseline_src.ensure_index(lat, lon,
                                                    probe_date=event.date)
                base = self._baseline_src.load(
                    event.date, event.step, _member_id(ds, self.mi)) * 1000.0
        finally:
            ds.close()
        return lat, lon, truth, base, pred


def _make_event_figure(event: Event, data: _EventData, run_label: str,
                       dlat: float, dlon: float) -> plt.Figure:
    lat_hr, lon_hr, truth, base, pred = data.load(event)
    clat, clon = event.lat, event.lon
    hr_mask = _zoom_mask(lat_hr, lon_hr, clat, clon, dlat, dlon)

    def crop(a):
        return a[hr_mask] if a is not None else None

    truth_z, base_z, pred_z = crop(truth), crop(base), crop(pred)
    lat_z, lon_z = lat_hr[hr_mask], lon_hr[hr_mask]

    finite = np.isfinite(lat_z) & np.isfinite(lon_z) & np.isfinite(pred_z)
    for a in (truth_z, base_z):
        if a is not None:
            finite &= np.isfinite(a)
    lat_z, lon_z, pred_z = lat_z[finite], lon_z[finite], pred_z[finite]
    truth_z = truth_z[finite] if truth_z is not None else None
    base_z = base_z[finite] if base_z is not None else None

    vmin, vmax = _robust_limits(truth_z, base_z, pred_z)
    unit = f"{data.var} (mm / 6h)"
    panels = []
    if truth_z is not None:
        panels.append((truth_z, f"Truth: {data.var}", "viridis", vmin, vmax, unit))
    if base_z is not None:
        panels.append((base_z, "Interp input (o1280)", "viridis", vmin, vmax, unit))
    panels.append((pred_z, f"Prediction: {data.var}", "viridis", vmin, vmax, unit))
    if truth_z is not None:
        error_z = pred_z - truth_z
        err_vmax = _error_limit(error_z)
        panels.append((error_z, "Prediction - truth", "RdBu_r",
                       -err_vmax, err_vmax, "Error (mm / 6h)"))
        peak_summary = (f"peak truth={float(np.nanmax(truth_z)):.1f} mm | "
                        f"peak pred={float(np.nanmax(pred_z)):.1f} mm")
    else:
        peak_summary = f"peak pred={float(np.nanmax(pred_z)):.1f} mm"

    triangulation = mtri.Triangulation(lon_z, lat_z)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 6.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (arr, title, cmap, lo, hi, cbar_label) in zip(axes, panels):
        sc = ax.tripcolor(triangulation, arr, cmap=cmap, vmin=lo, vmax=hi,
                          shading="gouraud", rasterized=True)
        ax.plot(clon, clat, marker="+", color="white", markersize=12,
                markeredgewidth=2.1)
        ax.plot(clon, clat, marker="+", color="black", markersize=9,
                markeredgewidth=1.2)
        plt.colorbar(sc, ax=ax, label=cbar_label, pad=0.025, shrink=0.82)
        _format_map(ax, clat=clat, clon=clon, dlat=dlat, dlon=dlon)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        f"{run_label} | {event.label} | event ({clat:.2f}°N, {clon:.2f}°E)"
        f" | zoom ±{dlat:g}° x ±{dlon:g}° | {peak_summary}",
        fontsize=10,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot intense precipitation events.")
    parser.add_argument("--predictions-dir", required=True,
                        help="Directory of predictions_*.nc files.")
    parser.add_argument("--out", required=True, help="Output PDF path.")
    parser.add_argument("--var", default="tp")
    parser.add_argument("--n-top", type=int, default=DEFAULT_N_TOP)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--dlat", type=float, default=DEFAULT_DLAT)
    parser.add_argument("--dlon", type=float, default=DEFAULT_DLON)
    parser.add_argument("--rank-by", choices=("pred", "truth"), default="pred")
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--truth-grib-tpl", default="")
    parser.add_argument("--baseline-grib-tpl", default="")
    parser.add_argument("--interp-index-cache", default="")
    args = parser.parse_args()

    src_path = Path(args.predictions_dir)
    run_label = args.run_label or src_path.parent.name

    events = find_precip_events(
        src_path, n_events=args.n_top, dlat=args.dlat, dlon=args.dlon,
        rank_by=args.rank_by, var=args.var, member=args.member_index,
        truth_grib_tpl=args.truth_grib_tpl,
    )
    print(f"Top {len(events)} events by max {args.rank_by} {args.var} (m):")
    for e in events:
        print(f"  {e.label}: {e.peak_value:.6f} m = {e.peak_value * 1000:.2f} mm"
              f" at ({e.lat:.2f}, {e.lon:.2f})")

    data = _EventData(args.truth_grib_tpl, args.baseline_grib_tpl,
                      args.interp_index_cache, args.var, args.member_index)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for event in events:
            fig = _make_event_figure(event, data, run_label, args.dlat, args.dlon)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
