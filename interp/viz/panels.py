"""Canonical matplotlib panel templates shared by every interp renderer.

Three shapes cover almost everything the tools produce:
  - heatmap   : (rows x sigmas) matrices — permutation, CKA, patching recovery
  - loglog    : metric-vs-sigma curves — ablation, activation norms
  - ranked_barh : top-K driver variables — full-sampling permutation, IG
plus the cartopy-optional geographic scatter for attribution maps.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

try:
    import cartopy.crs as ccrs
    from cartopy.geodesic import Geodesic
    HAS_CARTOPY = True
except Exception:  # pragma: no cover
    HAS_CARTOPY = False

try:
    # Importing registers Fabio Crameri's perceptually-uniform colormaps with
    # matplotlib under the 'cmc.*' names (cmc.batlow, cmc.lipari, ...).
    import cmcrameri.cm  # noqa: F401
    SEQ_CMAP = "cmc.batlow"   # sequential (permutation importance, etc.)
    SEQ_CMAP2 = "cmc.lipari"  # second sequential (per-block activation norms)
    PRECIP_CMAP = "cmc.devon_r"  # white->blue, for observed precip context maps
except Exception:  # pragma: no cover — fall back to matplotlib built-ins
    SEQ_CMAP = "viridis"
    SEQ_CMAP2 = "magma"
    PRECIP_CMAP = "Blues"


def fig_grid(n_panels: int, cols: int, panel_w: float = 6.0, panel_h: float = 4.5):
    """Figure with a rows x cols grid; returns (fig, flat axes list).

    Panels beyond n_panels are hidden.
    """
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(panel_w * cols, panel_h * rows),
                             squeeze=False)
    flat = [axes[i // cols][i % cols] for i in range(rows * cols)]
    for ax in flat[n_panels:]:
        ax.axis("off")
    return fig, flat[:n_panels]


def heatmap(ax, M, xlabels, ylabels, *, title="", cmap="viridis", norm=None,
            vmin=None, vmax=None, cbar_label="", hlines=(), fig=None,
            ytick_fontsize=7):
    """One (rows x cols) matrix panel with tick labels and a colorbar."""
    im = ax.imshow(np.asarray(M, dtype=float), aspect="auto", cmap=cmap,
                   norm=norm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels([str(x) for x in xlabels])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels([str(y) for y in ylabels], fontsize=ytick_fontsize)
    for y in hlines:
        ax.axhline(y, color="black", linewidth=0.6)
    ax.set_title(title, fontsize=9)
    if fig is not None:
        fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    return im


def loglog(ax, x, series: dict, *, xlabel="sigma", ylabel="", title="",
           logy=True, styles=None, ylim=None):
    """Metric-vs-sigma curves. `series` maps label -> y values; `styles` maps
    label -> dict of plot kwargs (marker, color, linestyle)."""
    any_pos = False
    for label, ys in series.items():
        kw = dict(marker="o")
        if styles and label in styles:
            kw.update(styles[label])
        ax.plot(x, ys, label=label, **kw)
        arr = np.asarray(ys, dtype=float)
        any_pos = any_pos or bool(np.any(np.isfinite(arr) & (arr > 0)))
    ax.set_xscale("log")
    if logy and any_pos:  # log y only when there is positive data to scale
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)


def ranked_barh(ax, names, vals, *, colors=None, stds=None, edgecolors=None,
                xlabel="", title="", floor=None):
    """Horizontal top-K bar chart (largest on top)."""
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, xerr=stds, color=colors, alpha=0.85,
            edgecolor=edgecolors, linewidth=1.0 if edgecolors else 0.0)
    if floor is not None:
        ax.axvline(floor, color="grey", linestyle=":", label=f"floor = {floor:.3g}")
        ax.legend(loc="lower right", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    # Declutter the x-axis: few ticks + a single shared 10^k offset rather than
    # crowded full-precision labels on tiny-magnitude attribution axes.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(axis="x", labelsize=7)


def _to180(x):
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def geo_panel(fig, subplotspec, lat, lon, vals, *, title, diverging,
              center=None, probe_radius_km=None, obs=None, cmap=None,
              cbar_label=None, vmin=None, vmax=None, peak=None, norm=None):
    """One geographic panel drawn into a GridSpec cell.

    `subplotspec` is a matplotlib SubplotSpec (e.g. ``gs[r, c]``); the GeoAxes
    is created there with the cartopy projection. Renders an INTERPOLATED
    surface (Delaunay tripcolor) rather than sparse scatter dots — essential on
    coarse grids like O96 (~1 deg spacing). Overlays: black contours of the
    OBSERVED target field (`obs`) so the weather structure (TC, rainband) is
    visible under the field; a solid ring at the probe's own (OUTPUT-disk)
    radius; a dashed 500 km reference ring only when the probe is smaller.

    Default colouring: ``diverging`` -> RdBu_r signed attribution; else a magma
    |attribution| map. Pass ``cmap``/``cbar_label`` (and optionally
    ``vmin``/``vmax``) to draw a plain field instead (used by the context maps).
    """
    lon = _to180(np.asarray(lon, dtype=float))
    lat = np.asarray(lat, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if HAS_CARTOPY:
        ax = fig.add_subplot(subplotspec, projection=ccrs.PlateCarree())
        tr = {"transform": ccrs.PlateCarree()}
    else:
        ax = fig.add_subplot(subplotspec)
        tr = {}
    if cmap is not None and norm is not None:
        # Explicit norm (e.g. sqrt PowerNorm) carries its own vmin/vmax — use it
        # verbatim so the colourbar spans exactly the intended range.
        kw = dict(cmap=cmap, norm=norm)
        cbl = cbar_label or ""
    elif cmap is not None:
        lo = vmin if vmin is not None else float(np.nanpercentile(vals, 1))
        hi = vmax if vmax is not None else float(np.nanpercentile(vals, 99))
        if hi <= lo:
            hi = lo + 1e-12
        kw = dict(cmap=cmap, vmin=lo, vmax=hi)
        cbl = cbar_label or ""
    elif diverging:
        vmx = float(np.percentile(np.abs(vals), 99)) or 1e-12
        kw = dict(cmap="RdBu_r", vmin=-vmx, vmax=vmx)
        cbl = "signed attribution"
    else:
        vmx = float(np.percentile(vals, 99)) or 1e-12
        kw = dict(cmap="magma", vmin=0.0, vmax=vmx)
        cbl = "|attribution|"
    # SMOOTH C1 cubic interpolation of the irregular O96 field onto a DENSE
    # regular raster, then pcolormesh. The O96 grid is only ~500 cells in a zoom
    # window; nearest-fill showed discrete Voronoi cells, flat tripcolor showed
    # facets, gouraud/coarse smoothing looked blurry. A geometry-based cubic
    # interpolant over the Delaunay mesh gives a continuous high-resolution field
    # (cells dissolved) — the finest the O96 data can render as. NaN outside the
    # data hull is masked. (Underlying information is still O96-native.)
    import matplotlib.tri as mtri
    if len(vals) >= 4:
        tri = mtri.Triangulation(lon, lat)
        tspan = lon[tri.triangles].max(axis=1) - lon[tri.triangles].min(axis=1)
        tri.set_mask(tspan > 8.0)  # drop periodic-seam / gap slivers
        n = 360
        gx = np.linspace(lon.min(), lon.max(), n)
        gy = np.linspace(lat.min(), lat.max(), n)
        GX, GY = np.meshgrid(gx, gy)
        try:
            Z = mtri.CubicTriInterpolator(tri, vals, kind="geom")(GX, GY)
        except Exception:
            Z = mtri.LinearTriInterpolator(tri, vals)(GX, GY)
        sc = ax.pcolormesh(GX, GY, Z, shading="auto", rasterized=True, **kw, **tr)
        if obs is not None:
            try:
                cs = ax.tricontour(tri, np.asarray(obs, dtype=float), levels=7,
                                   colors="k", linewidths=0.6, **tr)
                ax.clabel(cs, fontsize=5, fmt="%g")
            except Exception:
                pass
    else:
        sc = ax.scatter(lon, lat, c=vals, s=40, **kw, **tr)
    if HAS_CARTOPY:
        try:
            ax.coastlines(resolution="110m", linewidth=0.5)
        except Exception:
            pass
        pad = 1.0
        ax.set_extent([lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad],
                      crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6}
    else:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    if center is not None:
        clon, clat = _to180(center[1]), center[0]
        ax.plot([clon], [clat], marker="*", color="cyan", ms=15, mec="k", **tr)
        if HAS_CARTOPY:
            geod = Geodesic()
            ring_specs = []
            if probe_radius_km:
                ring_specs.append((probe_radius_km, "-",
                                   f"output disk {probe_radius_km:g} km"))
                if probe_radius_km < 400.0:
                    ring_specs.append((500.0, "--", "500 km"))
            else:
                ring_specs.append((500.0, "--", "500 km"))
            for rk, ls, lbl in ring_specs:
                circ = np.asarray(geod.circle(lon=float(clon), lat=float(clat),
                                              radius=rk * 1000.0, n_samples=120))
                ax.plot(circ[:, 0], circ[:, 1], color="k", linestyle=ls, lw=0.9,
                        transform=ccrs.PlateCarree())
                top = circ[np.argmax(circ[:, 1])]
                ax.text(top[0], top[1], lbl, fontsize=6, ha="center",
                        va="bottom", transform=ccrs.PlateCarree(),
                        bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                  ec="none", alpha=0.7))
    if peak is not None:
        # Mark where the field's extreme actually sits (vs the probe centre).
        ax.plot([_to180(peak[1])], [peak[0]], marker="x", color="lime", ms=12,
                mew=2.5, label="field peak", **tr)
    ax.set_title(title, fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label=cbl)
    return ax
