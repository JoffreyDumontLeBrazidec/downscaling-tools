"""One row of three maps: input member k (9 km) | iekm (4.4 km) | model member k (4.4 km).

Plain matplotlib + cartopy. Each field is interpolated smoothly (Clough-Tocher, C1) from its own
native points onto one fine regular grid, so no grid cells and no triangle facets are drawn.
Shared colour range = 2nd-98th percentile of iekm inside the box. dpi 250.

usage: python simple_maps.py CROP.npz VAR OUT.jpg --box LONMIN LONMAX LATMIN LATMAX --when "1 Oct 2025 00 UTC +24 h"
       [--model G300] [--place "Adriatic"]
VAR: wind | 2t | msl
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import CloughTocher2DInterpolator

SPEC = {
    "wind": dict(label="10 m wind speed (m s$^{-1}$)", name="10 m wind speed", cmap="viridis"),
    "2t": dict(label="2 m temperature (°C)", name="2 m temperature", cmap="RdYlBu_r"),
    "msl": dict(label="Mean sea level pressure (hPa)", name="Mean sea level pressure", cmap="viridis"),
}


def field(d, pre, var):
    if var == "wind":
        return np.hypot(d[pre + "10u"], d[pre + "10v"]).astype(float)
    v = d[pre + var].astype(float)
    if var == "2t":
        return v - 273.15
    if var == "msl":
        return v / 100.0 if np.nanmedian(v) > 2000 else v
    return v


def regrid(lon, lat, val, glon, glat, lat0):
    k = np.isfinite(val)
    c = np.cos(np.deg2rad(lat0))
    f = CloughTocher2DInterpolator(np.c_[lon[k] * c, lat[k]], val[k])
    return f(glon * c, glat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("var", choices=SPEC); ap.add_argument("out")
    ap.add_argument("--box", nargs=4, type=float, required=True)
    ap.add_argument("--when", required=True)
    ap.add_argument("--model", default="G300")
    ap.add_argument("--place", default="")
    ap.add_argument("--step-deg", type=float, default=0.02)
    a = ap.parse_args()
    d = np.load(a.npz)
    lo0, lo1, la0, la1 = a.box
    lat0 = 0.5 * (la0 + la1)
    m = int(d["member"])
    gx = np.arange(lo0, lo1 + 1e-9, a.step_deg)
    gy = np.arange(la0, la1 + 1e-9, a.step_deg)
    glon, glat = np.meshgrid(gx, gy)

    panels = [
        (f"Input member {m} (9 km)", d["lon_lres"], d["lat_lres"], field(d, "x_", a.var)),
        ("iekm (4.4 km)", d["lon_hres"], d["lat_hres"], field(d, "y_", a.var)),
        (f"{a.model} member {m} (4.4 km)", d["lon_hres"], d["lat_hres"], field(d, "p_", a.var)),
    ]
    grids = [regrid(lon.astype(float), lat.astype(float), v, glon, glat, lat0) for _, lon, lat, v in panels]
    vmin, vmax = np.nanpercentile(grids[1], [2, 98])
    if a.var == "wind":
        vmin = 0.0

    s = SPEC[a.var]
    aspect = (lo1 - lo0) * np.cos(np.deg2rad(lat0)) / (la1 - la0)
    w = 5.2
    fig, axs = plt.subplots(1, 3, figsize=(3 * w, w / aspect + 1.2), layout="constrained",
                            subplot_kw=dict(projection=ccrs.PlateCarree()))
    ext = [lo0, lo1, la0, la1]
    for ax, (title, *_), g in zip(axs, panels, grids):
        ax.set_extent(ext, crs=ccrs.PlateCarree())
        im = ax.imshow(g, origin="lower", extent=ext, transform=ccrs.PlateCarree(),
                       cmap=s["cmap"], vmin=vmin, vmax=vmax, interpolation="bilinear")
        if a.var == "msl":
            lv = np.arange(np.floor(np.nanmin(grids[1]) / 4) * 4, 1060, 4)
            ax.contour(gx, gy, g, levels=lv, colors="k", linewidths=0.4, alpha=0.6)
        ax.coastlines("10m", linewidth=0.6, color="k")
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, edgecolor="0.25")
        ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
        ax.set_title(title, fontsize=13)
    head = f"{s['name']}, {a.when}" + (f", {a.place}" if a.place else "")
    fig.text(0.5, 1.01, head, ha="center", va="bottom", fontsize=15)
    cb = fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.6, aspect=50,
                      extend="max" if a.var == "wind" else "both")
    cb.set_label(s["label"])
    fig.savefig(a.out, dpi=250, bbox_inches="tight", pil_kwargs={"quality": 92})
    print(a.out, f"range {vmin:.1f}-{vmax:.1f}")


if __name__ == "__main__":
    main()
