#!/usr/bin/env python3
"""Plot intense precipitation events from o48->o96 prediction files.

Handles two input layouts:
  --predictions-nc FILE   single predictions.nc (from-dataloader output, multiple samples)
  --predictions-dir DIR   directory of predictions_*.nc files (from-bundle output)

Outputs a PDF with one page per top event:
  truth | prediction | prediction - truth, zoomed tightly around the event centre.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

from .plotting.datetime_utils import safe_datetime_str

PRECIP_VARS = ("tp", "cp")
DEFAULT_DLAT = 2.0
DEFAULT_DLON = 2.5
DEFAULT_N_TOP = 3

try:
    from anemoi.training.diagnostics.maps import Coastlines

    COASTLINES = Coastlines()
except Exception:  # pragma: no cover - coastlines are a presentation garnish
    COASTLINES = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _iter_samples(src: Path):
    """Yield (label, ds_sample) for each (sample, member=0) in src."""
    if src.is_file():
        ds = xr.open_dataset(src, engine="netcdf4")
        n = int(ds.sizes.get("sample", 1))
        for i in range(n):
            si = ds.isel(sample=i) if "sample" in ds.dims else ds
            si = si.isel(ensemble_member=0) if "ensemble_member" in si.dims else si
            date_val = si["date"].values if "date" in si else i
            yield safe_datetime_str(date_val) or str(i), si
        ds.close()
    elif src.is_dir():
        files = sorted(src.glob("predictions_*.nc"))
        if not files:
            raise FileNotFoundError(f"No predictions_*.nc in {src}")
        for f in files:
            ds = xr.open_dataset(f, engine="netcdf4")
            si = ds.isel(sample=0) if "sample" in ds.dims else ds
            si = si.isel(ensemble_member=0) if "ensemble_member" in si.dims else si
            date_val = si["date"].values if "date" in si else f.stem
            yield safe_datetime_str(date_val) or f.stem, si
            ds.close()
    else:
        raise FileNotFoundError(f"Not a file or directory: {src}")


def _find_precip_var(ws_values) -> str | None:
    for v in PRECIP_VARS:
        if v in ws_values:
            return v
    return None


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------

def _collect_events(src: Path, var: str, n_top: int, *, rank_by: str = "pred") -> list[tuple[float, str, xr.Dataset, int]]:
    """Return top n_top (max_val, label, ds_sample, max_hr_idx) sorted by -max_val."""
    field = "y" if rank_by == "truth" else "y_pred"
    events: list[tuple[float, str, xr.Dataset, int]] = []
    for label, ds in _iter_samples(src):
        ws = list(ds["weather_state"].values)
        if var not in ws:
            continue
        values = ds[field].sel(weather_state=var).values.ravel()
        max_val = float(np.nanmax(values))
        max_idx = int(np.nanargmax(values))
        events.append((max_val, label, ds, max_idx))
    events.sort(key=lambda e: -e[0])
    return events[:n_top]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _zoom_mask(lat: np.ndarray, lon: np.ndarray, clat: float, clon: float,
               dlat: float, dlon: float) -> np.ndarray:
    return (
        (lat >= clat - dlat) & (lat <= clat + dlat)
        & (lon >= clon - dlon) & (lon <= clon + dlon)
    )


def _add_coastlines(ax) -> None:
    if COASTLINES is None:
        return
    coast_segs_deg = [np.degrees(s) for s in COASTLINES.lines.get_segments()]
    ax.add_collection(LineCollection(coast_segs_deg, linewidths=0.7, colors="black", zorder=10))


def _format_map(ax, *, clat: float, clon: float, dlat: float, dlon: float) -> None:
    ax.set_xlim(clon - dlon, clon + dlon)
    ax.set_ylim(clat - dlat, clat + dlat)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("auto", adjustable=None)
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.4)
    _add_coastlines(ax)


def _robust_limits(*arrays: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    vals = vals[vals >= 0]
    if vals.size == 0:
        return 0.0, 1.0
    return 0.0, max(float(np.nanpercentile(vals, 99.7)), 1.0)


def _error_limit(error: np.ndarray) -> float:
    vals = np.abs(error[np.isfinite(error)])
    if vals.size == 0:
        return 1.0
    return max(float(np.nanpercentile(vals, 99.0)), 1.0)


def _make_event_figure(
    event_label: str,
    ds: xr.Dataset,
    var: str,
    max_hr_idx: int,
    run_label: str,
    dlat: float,
    dlon: float,
) -> plt.Figure:
    lat_hr = ds["lat_hres"].values
    lon_hr = ds["lon_hres"].values
    clat = float(lat_hr[max_hr_idx])
    clon = float(lon_hr[max_hr_idx])

    hr_mask = _zoom_mask(lat_hr, lon_hr, clat, clon, dlat, dlon)

    y_pred_full = ds["y_pred"].sel(weather_state=var).values.ravel() * 1000.0
    y_pred_z = y_pred_full[hr_mask]

    has_truth = "y" in ds
    n_panels = 3 if has_truth else 1

    if has_truth:
        y_full = ds["y"].sel(weather_state=var).values.ravel() * 1000.0
        y_z = y_full[hr_mask]
        vmin, vmax = _robust_limits(y_z, y_pred_z)
        error_z = y_pred_z - y_z
        err_vmax = _error_limit(error_z)
        peak_summary = f"peak truth={float(np.nanmax(y_z)):.1f} mm | peak pred={float(np.nanmax(y_pred_z)):.1f} mm"
    else:
        y_z = None
        error_z = None
        err_vmax = 1.0
        vmin, vmax = _robust_limits(y_pred_z)
        peak_summary = f"peak pred={float(np.nanmax(y_pred_z)):.1f} mm"

    lat_z = lat_hr[hr_mask]
    lon_z = lon_hr[hr_mask]
    finite = np.isfinite(lat_z) & np.isfinite(lon_z) & np.isfinite(y_pred_z)
    if y_z is not None:
        finite &= np.isfinite(y_z)
    lat_z = lat_z[finite]
    lon_z = lon_z[finite]
    y_pred_z = y_pred_z[finite]
    if y_z is not None:
        y_z = y_z[finite]
        error_z = error_z[finite]
    triangulation = mtri.Triangulation(lon_z, lat_z)

    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 6.6), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    if y_z is not None:
        panels = [
            (y_z, f"Truth: {var}", "viridis", vmin, vmax, f"{var} (mm / 6h)"),
            (y_pred_z, f"Prediction: {var}", "viridis", vmin, vmax, f"{var} (mm / 6h)"),
            (error_z, "Prediction - truth", "RdBu_r", -err_vmax, err_vmax, "Error (mm / 6h)"),
        ]
    else:
        panels = [(y_pred_z, f"Prediction: {var}", "viridis", vmin, vmax, f"{var} (mm / 6h)")]

    for ax, (data, title, cmap, lo, hi, colorbar_label) in zip(axes, panels):
        sc = ax.tripcolor(
            triangulation, data, cmap=cmap, vmin=lo, vmax=hi,
            shading="gouraud", rasterized=True,
        )
        ax.plot(clon, clat, marker="+", color="white", markersize=12, markeredgewidth=2.1)
        ax.plot(clon, clat, marker="+", color="black", markersize=9, markeredgewidth=1.2)
        plt.colorbar(sc, ax=ax, label=colorbar_label, pad=0.025, shrink=0.82)
        _format_map(ax, clat=clat, clon=clon, dlat=dlat, dlon=dlon)
        ax.set_title(title, fontsize=9)

    fig.suptitle(
        f"{run_label} | {event_label} | event ({clat:.2f}°N, {clon:.2f}°E)"
        f" | zoom ±{dlat:g}° x ±{dlon:g}° | {peak_summary}",
        fontsize=10,
    )
    return fig


def _make_overview_figure(
    events: list[tuple[float, str, xr.Dataset, int]],
    var: str,
    run_label: str,
    rank_by: str,
) -> plt.Figure:
    """Global scatter showing event locations coloured by max tp."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for max_val, label, ds, max_hr_idx in events:
        lat_hr = ds["lat_hres"].values
        lon_hr = ds["lon_hres"].values
        ax.plot(
            float(lon_hr[max_hr_idx]),
            float(lat_hr[max_hr_idx]),
            "o",
            markersize=8,
            label=f"{label[:10]} ({max_val * 1000:.1f} mm)",
        )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(f"{run_label} — top-{len(events)} {var} event locations (max {rank_by})")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot intense precipitation events.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--predictions-nc", default="", help="Single predictions.nc file.")
    src.add_argument("--predictions-dir", default="", help="Directory of predictions_*.nc files.")
    parser.add_argument("--out", required=True, help="Output PDF path.")
    parser.add_argument("--var", default="", help="Precipitation variable (default: auto-detect tp/cp).")
    parser.add_argument("--n-top", type=int, default=DEFAULT_N_TOP, help="Number of top events to plot.")
    parser.add_argument("--run-label", default="", help="Label shown in plot titles.")
    parser.add_argument("--dlat", type=float, default=DEFAULT_DLAT)
    parser.add_argument("--dlon", type=float, default=DEFAULT_DLON)
    parser.add_argument("--rank-by", choices=("pred", "truth"), default="pred")
    args = parser.parse_args()

    src_path = Path(args.predictions_nc or args.predictions_dir)
    run_label = args.run_label or src_path.parent.name

    # Detect var from first sample if not given
    var = args.var
    if not var:
        for _, ds in _iter_samples(src_path):
            ws = list(ds["weather_state"].values)
            var = _find_precip_var(ws)
            if var:
                break
        if not var:
            raise SystemExit(f"No precipitation variable ({PRECIP_VARS}) found in predictions.")
    print(f"Precipitation variable: {var}")

    events = _collect_events(src_path, var, args.n_top, rank_by=args.rank_by)
    if not events:
        raise SystemExit(f"No samples with variable '{var}' found.")
    print(f"Top {len(events)} events by max {args.rank_by} {var} (m):")
    for mv, lbl, _, _ in events:
        print(f"  {lbl}: {mv:.6f} m = {mv * 1000:.2f} mm")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        for max_val, label, ds, max_hr_idx in events:
            fig = _make_event_figure(label, ds, var, max_hr_idx, run_label, args.dlat, args.dlon)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
