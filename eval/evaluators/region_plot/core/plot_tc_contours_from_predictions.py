from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# The config module names the table KNOWN_REGION_BOXES; importing the old name
# PREDICTION_REGION_BOXES from it failed, so this module could not be imported.
from .plotting.config import KNOWN_REGION_BOXES as PREDICTION_REGION_BOXES
from .plotting.config import RENDER_DPI  # noqa: F401  (kept for importers)
from .plotting.coordinate_utils import get_region_ds
from eval.shared.manifest import write_manifest
from .plotting.metadata import sample_meta_title
from .plotting.preprocessing import ensure_x_interp_for_plotting

DEFAULT_REGIONS = ["idalia", "franklin"]


def _absolute_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _state_values(ds_region: xr.Dataset, variable_name: str, weather_state: str) -> np.ndarray:
    return np.asarray(ds_region[variable_name].sel(weather_state=weather_state).values, dtype=float)


def _wind10m(ds_region: xr.Dataset, variable_name: str) -> np.ndarray:
    u = _state_values(ds_region, variable_name, "10u")
    v = _state_values(ds_region, variable_name, "10v")
    return np.sqrt(u**2 + v**2)


def _msl_hpa(ds_region: xr.Dataset, variable_name: str) -> np.ndarray:
    values = _state_values(ds_region, variable_name, "msl")
    if float(np.nanmedian(values)) > 2000.0:
        values = values * 0.01
    return values


def _levels(fields: list[np.ndarray], *, n_levels: int = 21) -> np.ndarray:
    """About ``n_levels`` round contour levels spanning the 2nd to 98th percentile."""
    from matplotlib.ticker import MaxNLocator

    flat = np.concatenate([field.reshape(-1) for field in fields])
    vmin = float(np.nanpercentile(flat, 2))
    vmax = float(np.nanpercentile(flat, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(flat))
        vmax = float(np.nanmax(flat))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    levels = MaxNLocator(nbins=n_levels - 1).tick_values(vmin, vmax)
    levels = levels[(levels >= vmin - 1e-9) & (levels <= vmax + 1e-9)]
    return levels if levels.size >= 3 else np.linspace(vmin, vmax, n_levels)


def _draw_panel(ax, lon: np.ndarray, lat: np.ndarray, field: np.ndarray, *, levels: np.ndarray, title: str,
                cmap=None, left_labels: bool = True, bottom_labels: bool = True):
    """Filled contours plus contour lines of one field on a Cartopy map panel.

    The points are projected first and triangulated in map coordinates: letting Cartopy
    reproject filled tricontour paths can merge or drop polygons.
    """
    import cartopy.crs as ccrs
    from eval.plotting import add_geography

    xyz = ax.projection.transform_points(ccrs.PlateCarree(), np.asarray(lon, float), np.asarray(lat, float))
    x, y = xyz[:, 0], xyz[:, 1]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(field)
    contourf = ax.tricontourf(x[ok], y[ok], field[ok], levels=levels, cmap=cmap or "viridis", extend="both")
    contourf.set_rasterized(True)
    ax.tricontour(x[ok], y[ok], field[ok], levels=levels, colors="black", linewidths=0.4, alpha=0.6)
    gl = add_geography(ax, label_size=7)
    if gl is not None:
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
    ax.set_title(title, fontsize=10)
    return contourf


def _write_manifest(*, out_root: Path, predictions_path: Path, region_names: list[str], sample_index: int, ensemble_member_index: int, generated: list[str]) -> Path:
    payload = {
        "suite_kind": "storm",
        "plot_style": "tc_contour",
        "predictions_file": str(predictions_path),
        "out_dir": str(out_root),
        "regions": list(region_names),
        "sample_index": int(sample_index),
        "ensemble_member_index": int(ensemble_member_index),
        "generated_files": list(generated),
        "panel_contract": {
            "rows": ["msl_hpa", "wind10m_ms"],
            "columns": ["x_interp", "y", "y_pred"],
        },
    }
    return write_manifest(out_root=out_root, payload=payload)


def render_tc_contour_suite_from_predictions_file(
    *,
    predictions_nc: str | Path,
    out_dir: str | Path,
    region_names: list[str] | None = None,
    sample_index: int = 0,
    ensemble_member_index: int = 0,
    also_png: bool = True,
) -> list[str]:
    """Contour maps of mean sea level pressure and 10 m wind speed for storm regions.

    Writes ``all_regions_plots.pdf`` (one page per region) and ``<region>.pdf`` plus
    ``<region>.png`` per region. ``also_png`` is kept for callers; the PNG is always written.
    """
    predictions_path = _absolute_path(predictions_nc)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    out_root = _absolute_path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    combined_pdf = out_root / "all_regions_plots.pdf"
    generated: list[str] = [str(combined_pdf)]

    names = region_names or DEFAULT_REGIONS
    unknown = [name for name in names if name not in PREDICTION_REGION_BOXES]
    if unknown:
        raise ValueError(f"Unknown region names: {unknown}. Known regions: {sorted(PREDICTION_REGION_BOXES)}")

    with xr.open_dataset(predictions_path) as ds:
        if "sample" in ds.dims:
            ds = ds.isel(sample=sample_index)
        if "ensemble_member" in ds.dims:
            ds = ds.isel(ensemble_member=ensemble_member_index)
        ds = ensure_x_interp_for_plotting(ds, predictions_path=predictions_path)

        for required in ("x_interp", "y", "y_pred"):
            if required not in ds.variables:
                raise ValueError(f"Missing required variable for TC contour suite: {required}")
        for weather_state in ("msl", "10u", "10v"):
            if weather_state not in ds["weather_state"].values:
                raise ValueError(f"Missing required weather_state={weather_state} in {predictions_path}")

        lon = np.asarray(ds["lon_hres"].values, dtype=float)
        lat = np.asarray(ds["lat_hres"].values, dtype=float)

        from eval.plotting import FigureBook, eval_style, save_figure, variable_spec
        from eval.plotting.maps_helpers import (
            octahedral_grid_name,
            region_panel_title,
            region_projection,
            set_inner_extent,
        )

        target_grid = str(ds.attrs.get("grid", "")).strip() or octahedral_grid_name(int(ds.sizes["grid_point_hres"]))
        input_grid = octahedral_grid_name(int(ds.sizes["grid_point_lres"])) if "grid_point_lres" in ds.sizes else None
        titles = {key: region_panel_title(key, input_grid=input_grid, target_grid=target_grid)
                  for key in ("x_interp", "y", "y_pred")}
        msl_spec, wind_spec = variable_spec("msl"), variable_spec("10ff")

        with FigureBook(combined_pdf) as book:
            for region_name in names:
                ds_region = get_region_ds(ds, PREDICTION_REGION_BOXES[region_name])
                region_lon = np.asarray(ds_region["lon_hres"].values, dtype=float)
                region_lat = np.asarray(ds_region["lat_hres"].values, dtype=float)

                msl_fields = [
                    _msl_hpa(ds_region, "x_interp"),
                    _msl_hpa(ds_region, "y"),
                    _msl_hpa(ds_region, "y_pred"),
                ]
                wind_fields = [
                    _wind10m(ds_region, "x_interp"),
                    _wind10m(ds_region, "y"),
                    _wind10m(ds_region, "y_pred"),
                ]
                msl_levels = _levels(msl_fields)
                wind_levels = _levels(wind_fields)

                lat_min, lat_max, lon_min, lon_max = (float(v) for v in PREDICTION_REGION_BOXES[region_name])
                extent = (lon_min, lon_max, lat_min, lat_max)
                with eval_style():
                    fig, axs = plt.subplots(2, 3, figsize=(14, 9), squeeze=False,
                                            subplot_kw={"projection": region_projection(*extent)})
                    top_mappables = []
                    bottom_mappables = []
                    for col, key in enumerate(("x_interp", "y", "y_pred")):
                        for row in range(2):
                            set_inner_extent(axs[row, col], extent)
                        top_mappables.append(
                            _draw_panel(
                                axs[0, col], region_lon, region_lat, msl_fields[col],
                                levels=msl_levels, title=f"{titles[key]}\n{msl_spec.name}",
                                cmap=msl_spec.field_cmap(), left_labels=col == 0, bottom_labels=False,
                            )
                        )
                        bottom_mappables.append(
                            _draw_panel(
                                axs[1, col], region_lon, region_lat, wind_fields[col],
                                levels=wind_levels, title=f"{titles[key]}\n{wind_spec.name}",
                                cmap=wind_spec.field_cmap(), left_labels=col == 0,
                            )
                        )
                    cbar_top = fig.colorbar(top_mappables[0], ax=axs[0, :], orientation="vertical",
                                            fraction=0.025, pad=0.02, aspect=25)
                    cbar_top.set_ticks(msl_levels[::max(1, len(msl_levels) // 8)])
                    cbar_top.set_label(msl_spec.label)
                    cbar_bottom = fig.colorbar(bottom_mappables[0], ax=axs[1, :], orientation="vertical",
                                               fraction=0.025, pad=0.02, aspect=25)
                    cbar_bottom.set_ticks(wind_levels[::max(1, len(wind_levels) // 8)])
                    cbar_bottom.set_label(wind_spec.label)
                    fig.suptitle(sample_meta_title(ds_region, region_name, sample_index)
                                 + " (tropical-cyclone contour maps)")
                    # Per-region PNG + PDF, and a page of the combined PDF.
                    for path in save_figure(fig, out_root / region_name):
                        generated.append(str(path))
                    book.add(fig, name=region_name)

    manifest_path = _write_manifest(
        out_root=out_root,
        predictions_path=predictions_path,
        region_names=names,
        sample_index=sample_index,
        ensemble_member_index=ensemble_member_index,
        generated=generated,
    )
    generated.append(str(manifest_path))
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render TC contour suites from predictions_*.nc files.")
    parser.add_argument("--predictions-nc", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ensemble-member-index", type=int, default=0)
    parser.add_argument("--also-png", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = render_tc_contour_suite_from_predictions_file(
        predictions_nc=args.predictions_nc,
        out_dir=args.out_dir,
        region_names=[value.strip() for value in args.regions.split(",") if value.strip()] or None,
        sample_index=args.sample_index,
        ensemble_member_index=args.ensemble_member_index,
        also_png=args.also_png,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
