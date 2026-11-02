#!/usr/bin/env python3
"""o2560 box read: zoom maps of tp (mm per 6 h) and 10 m wind speed (m/s): truth, input and members 1-3 of each row.
Cases: (1) the Alps box 43-49 N 4-16 E at the date with the largest truth tp over that box at the lead; (2) a storm: the date
whose truth msl minimum north of 35 N inside the box (strip excluded) is the deepest, on a 14 x 20 degree window centred on
that minimum; the read says 'no storm' when that minimum is above 990 hPa. Fields are sampled to a regular 0.035-degree grid
by nearest neighbour (o2560 native spacing). Writes PNGs and cases.json. Usage: python zoom_maps_o2560.py <lead> <out dir> <label>=<pred dir> [...]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kit"))
from o2560_common import edge_strip  # noqa: E402

lead = int(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
ROWS = {k: Path(v) for k, v in (a.split("=", 1) for a in sys.argv[3:])}
labs = list(ROWS); first = ROWS[labs[0]]
ALPS = (43.0, 49.0, 4.0, 16.0)


def load(path, var_names, members=(0, 1, 2)):
    with netCDF4.Dataset(str(path)) as d:
        d.set_auto_mask(False)
        st = [str(s) for s in d["weather_state"][:]]
        lat = np.asarray(d["lat_hres"][:]); lon = ((np.asarray(d["lon_hres"][:]) + 180) % 360) - 180
        g = {}
        for v in var_names:
            j = st.index(v)
            g[v] = (np.asarray(d["y"][0, :, j]), np.asarray(d["x_interp"][0, :, j]), np.asarray(d["y_pred"][0, list(members), :, j]))
    return lat, lon, g


common = set.intersection(*[{p.name[12:20] for p in d.glob(f"predictions_*_step{lead:03d}.nc")} for d in ROWS.values()])
if not common:
    sys.exit(f"FATAL: no date common to all rows at +{lead} h")
files = sorted(first / f"predictions_{q}_step{lead:03d}.nc" for q in common)   # cases are chosen among dates every row holds
best_alps = best_storm = None
for f in files:
    lat, lon, g = load(f, ("tp", "msl"), members=(0,))
    m = (lat >= ALPS[0]) & (lat <= ALPS[1]) & (lon >= ALPS[2]) & (lon <= ALPS[3])
    a = float(g["tp"][0][m].max()) * 1000
    if best_alps is None or a > best_alps[1]:
        best_alps = (f.name[12:20], a)
    s = (lat > 35) & ~edge_strip(lat, lon)
    k = np.flatnonzero(s)[np.argmin(g["msl"][0][s])]
    if best_storm is None or g["msl"][0][k] < best_storm[1]:
        best_storm = (f.name[12:20], float(g["msl"][0][k]), float(lat[k]), float(lon[k]))
cases = {"alps": dict(date=best_alps[0], truth_tp_max_mm=best_alps[1], box=list(ALPS)),
         "storm": dict(date=best_storm[0], truth_msl_min_hpa=best_storm[1] / 100, centre=[best_storm[2], best_storm[3]],
                       is_storm=best_storm[1] / 100 < 990.0,
                       box=[best_storm[2] - 7, best_storm[2] + 7, best_storm[3] - 10, best_storm[3] + 10])}
print("CASES", json.dumps(cases), flush=True)


def panel_grid(lat, lon, box, res=0.035):
    gy = np.arange(box[0], box[1], res); gx = np.arange(box[2], box[3], res)
    GX, GY = np.meshgrid(gx, gy)
    m = (lat >= box[0] - 0.2) & (lat <= box[1] + 0.2) & (lon >= box[2] - 0.2) & (lon <= box[3] + 0.2)
    idx = np.flatnonzero(m)
    _, nn = cKDTree(np.column_stack([lon[idx] * np.cos(np.deg2rad(lat[idx])), lat[idx]])).query(
        np.column_stack([GX.ravel() * np.cos(np.deg2rad(GY.ravel())), GY.ravel()]))
    return gx, gy, idx[nn].reshape(GX.shape)


for case, c in cases.items():
    date, box = c["date"], c["box"]
    panels = []
    for i, lab in enumerate(labs):
        lat, lon, g = load(ROWS[lab] / f"predictions_{date}_step{lead:03d}.nc", ("tp", "10u", "10v"))
        if i == 0:
            gx, gy, nn = panel_grid(lat, lon, box)
            panels += [("truth", g["tp"][0] * 1000, np.hypot(g["10u"][0], g["10v"][0])),
                       ("input (o1280)", g["tp"][1] * 1000, np.hypot(g["10u"][1], g["10v"][1]))]
        for k in range(3):
            panels.append((f"{lab} m{k + 1}", g["tp"][2][k] * 1000, np.hypot(g["10u"][2][k], g["10v"][2][k])))
    for vi, (vname, cmap, vmax) in enumerate((("tp mm/6h", "Blues", None), ("10 m wind m/s", "viridis", None))):
        vals = [p[1 + vi][nn] for p in panels]
        vmax = float(np.percentile(vals[0], 99.9)) * 1.2 if vmax is None else vmax
        nc = 5; nr = int(np.ceil(len(panels) / nc))
        fig, ax = plt.subplots(nr, nc, figsize=(4 * nc, 3.2 * nr), squeeze=False)
        for a, (p, v) in zip(ax.ravel(), zip(panels, vals)):
            im = a.pcolormesh(gx, gy, v, cmap=cmap, vmin=0, vmax=max(vmax, 1e-6), shading="auto")
            a.set_title(f"{p[0]} max {v.max():.1f}", fontsize=9); a.set_aspect(1 / np.cos(np.deg2rad(np.mean(box[:2]))))
        for a in ax.ravel()[len(panels):]:
            a.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.6, label=vname)
        fig.suptitle(f"{case} {date} +{lead} h, {vname}; o2560 box read 2 Nov 2026 (EMA-10000, rule-31 seeds)")
        p = out / f"zoom_{case}_{date}_L{lead:03d}_{'tp' if vi == 0 else 'ws'}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); print("WROTE", p, flush=True)
(out / "cases.json").write_text(json.dumps(cases, indent=1))
print("ZOOM_MAPS_DONE", lead)
