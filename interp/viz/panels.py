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

try:
    import cartopy.crs as ccrs
    from cartopy.geodesic import Geodesic
    HAS_CARTOPY = True
except Exception:  # pragma: no cover
    HAS_CARTOPY = False


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
    for label, ys in series.items():
        kw = dict(marker="o")
        if styles and label in styles:
            kw.update(styles[label])
        ax.plot(x, ys, label=label, **kw)
    ax.set_xscale("log")
    if logy:
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


def _to180(x):
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def geo_panel(fig, nrows, ncols, pos, lat, lon, vals, *, title, diverging,
              center=None, rings=(200, 500)):
    """One geographic scatter panel; coastlines best-effort (offline-safe)."""
    lon = _to180(lon)
    if HAS_CARTOPY:
        ax = fig.add_subplot(nrows, ncols, pos, projection=ccrs.PlateCarree())
        tr = {"transform": ccrs.PlateCarree()}
    else:
        ax = fig.add_subplot(nrows, ncols, pos)
        tr = {}
    if diverging:
        vmax = float(np.percentile(np.abs(vals), 99)) or 1e-12
        kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        cbl = "signed attribution"
    else:
        vmax = float(np.percentile(vals, 99)) or 1e-12
        kw = dict(cmap="magma", vmin=0.0, vmax=vmax)
        cbl = "|attribution|"
    sc = ax.scatter(lon, lat, c=vals, s=5, **kw, **tr)
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
            for rk in rings:
                circ = np.asarray(geod.circle(lon=float(clon), lat=float(clat),
                                              radius=rk * 1000.0, n_samples=120))
                ax.plot(circ[:, 0], circ[:, 1], "k--", lw=0.7,
                        transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label=cbl)
    return ax
