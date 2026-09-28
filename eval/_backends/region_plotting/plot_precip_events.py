#!/usr/bin/env python3
"""Plot intense precipitation events from downscaling prediction files.

Selection is delegated to eval._backends.region_plotting.precip_events
(find_precip_events), so the pages always match the evaluator's events.json.

Each event page shows, zoomed tightly around the event centre, as Cartopy maps:
  truth | interpolated input | model | model minus truth
The pages go to one PDF plus a PNG per page in ``<name>_pages/``.

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
import numpy as np
import xarray as xr

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

def _zoom_mask(lat: np.ndarray, lon: np.ndarray, clat: float, clon: float,
               dlat: float, dlon: float) -> np.ndarray:
    return (
        (lat >= clat - dlat) & (lat <= clat + dlat)
        & (lon >= clon - dlon) & (lon <= clon + dlon)
    )


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
        self.input_grid: str | None = None   # for the panel titles, e.g. "O1280"
        self.target_grid: str | None = None  # e.g. "O2560"

    def load(self, event: Event):
        ds = xr.open_dataset(event.nc_path)
        try:
            ws = [str(s) for s in ds["weather_state"].values]
            vi = ws.index(self.var)
            from eval.plotting.maps_helpers import octahedral_grid_name

            if "grid_point_lres" in ds.sizes:
                self.input_grid = octahedral_grid_name(int(ds.sizes["grid_point_lres"]))
            self.target_grid = (str(ds.attrs.get("grid", "")).strip()
                                or octahedral_grid_name(int(ds.sizes["grid_point_hres"])))
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
                if truth.size != lat.size:
                    # The first load of a regional run returns the full truth grid,
                    # because the support index is only built by verify_grid.
                    truth = self._truth_src.load(event.date, event.step)
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
    """Truth | interpolated input | model | model minus truth, zoomed on the event.

    Cartopy maps (projection from ``select_projection``, coastlines, borders, labelled grid
    lines). The three fields share one colour scale; the error panel is zero-centred
    (``BrBG``: wetter than truth is blue-green, drier is brown). Values in mm per 6 h.
    """
    from eval.plotting import add_geography, eval_style, extend_for, variable_spec
    from eval.plotting.maps import symmetric_norm
    from eval.plotting.maps_helpers import (
        colorbar_beside,
        draw_unstructured,
        region_projection,
        set_grid_ticks,
        set_inner_extent,
    )
    from matplotlib.colors import Normalize

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

    spec = variable_spec(data.var)
    vmin, vmax = _robust_limits(truth_z, base_z, pred_z)
    field_norm = Normalize(vmin=vmin, vmax=vmax)
    field_label = f"{spec.name}, 6 h accumulation ({spec.unit})"
    tgt = f" ({data.target_grid})" if data.target_grid else ""
    panels = []  # (values, title, group)
    if truth_z is not None:
        panels.append((truth_z, f"Truth{tgt}", "field"))
    if base_z is not None:
        into = f" to {data.target_grid}" if data.target_grid else ""
        src = f" ({data.input_grid})" if data.input_grid else ""
        panels.append((base_z, f"Input{src} interpolated{into}", "field"))
    panels.append((pred_z, f"Model{tgt}", "field"))
    error_z = None
    if truth_z is not None:
        error_z = pred_z - truth_z
        err_norm, _ = symmetric_norm(error_z, limit=_error_limit(error_z))
        panels.append((error_z, "Model minus truth", "error"))
        peak_summary = (f"peak truth {float(np.nanmax(truth_z)):.1f} mm, "
                        f"peak model {float(np.nanmax(pred_z)):.1f} mm")
    else:
        peak_summary = f"peak model {float(np.nanmax(pred_z)):.1f} mm (no truth)"

    extent = (clon - dlon, clon + dlon, clat - dlat, clat + dlat)
    n = len(panels)
    n_field = sum(1 for p in panels if p[2] == "field")
    with eval_style():
        fig = plt.figure(figsize=(4.3 * n + 1.6, 4.9))
        ratios = [1.0] * n_field + [0.16] + ([1.0, 0.16] if error_z is not None else [])
        gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.08,
                              left=0.05, right=0.97, bottom=0.08, top=0.80)
        proj = region_projection(*extent)
        field_axes, error_axes, field_mesh, error_mesh = [], [], None, None
        slot = 0
        for k, (arr, title, group) in enumerate(panels):
            if group == "error":
                slot = n_field + 1
            ax = fig.add_subplot(gs[0, slot], projection=proj)
            slot += 1
            set_inner_extent(ax, extent)
            if group == "field":
                field_mesh = draw_unstructured(ax, lon_z, lat_z, arr, extent,
                                               cmap=spec.field_cmap(), norm=field_norm)
                field_axes.append(ax)
            else:
                error_mesh = draw_unstructured(ax, lon_z, lat_z, arr, extent,
                                               cmap=spec.error_cmap(), norm=err_norm)
                error_axes.append(ax)
            gl = add_geography(ax, label_size=7, resolution="10m")
            set_grid_ticks(gl, extent)
            if gl is not None:
                gl.left_labels = k == 0
            ax.plot(clon, clat, marker="+", color="white", markersize=12, markeredgewidth=2.4,
                    transform=_plate_carree(), zorder=8)
            ax.plot(clon, clat, marker="+", color="black", markersize=9, markeredgewidth=1.2,
                    transform=_plate_carree(), zorder=9)
            ax.set_title(title, fontsize=10)
        fields = [p[0] for p in panels if p[2] == "field"]
        if field_mesh is not None:
            colorbar_beside(fig, field_axes, field_mesh, field_label,
                            extend=extend_for(field_norm, *fields), width=0.012, pad=0.008)
        if error_mesh is not None:
            colorbar_beside(fig, error_axes, error_mesh, f"Model minus truth ({spec.unit})",
                            extend=extend_for(err_norm, error_z), width=0.012, pad=0.008)
        head = f"{run_label}: " if run_label else ""
        fig.suptitle(
            f"{head}heavy-precipitation event {event.label.split('_')[0].replace('event', '')} "
            f"({event.date}, lead time {event.step} h)\n"
            f"centre {abs(clat):.2f}°{'N' if clat >= 0 else 'S'} "
            f"{abs(clon):.2f}°{'E' if clon >= 0 else 'W'}, window ±{dlat:g}° latitude × ±{dlon:g}° longitude; "
            f"{peak_summary}",
            fontsize=11,
        )
    return fig


def _plate_carree():
    import cartopy.crs as ccrs

    return ccrs.PlateCarree()


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
    from eval.plotting import FigureBook

    with FigureBook(out_path, png=True) as book:
        for event in events:
            fig = _make_event_figure(event, data, run_label, args.dlat, args.dlon)
            book.add(fig, name=event.label)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
