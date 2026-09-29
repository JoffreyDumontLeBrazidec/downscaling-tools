"""Saved trajectory panels, invoked only by the opt-in eval.cli zoom_maps mode."""
from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import warnings

import numpy as np


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot_member_wind_maps import _parse_kv

    if args.run or args.grib or args.members or args.variable != "wind10m":
        raise SystemExit("Saved-NPZ mode is a separate wind-only comparison; do not mix source modes.")
    if not args.rotation_instrument:
        raise SystemExit("--trajectory-npz requires --rotation-instrument.")
    instrument = Path(args.rotation_instrument).resolve()
    spec = importlib.util.spec_from_file_location("zoom_maps_tc_rotation", instrument)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    class BoxGeometry(T.Geometry):
        def window(self, storm):
            if "box" not in self._win:
                idx = np.arange(len(self.lat))
                d, nb = self.tree.query(self.xyz[idx], k=48)
                dkm = d * T.R_EARTH
                w = np.exp(-0.5 * (dkm / T.SIG_MSL) ** 2)
                w[dkm > 3 * T.SIG_MSL] = 0.0
                w /= w.sum(axis=1, keepdims=True)
                self._win["box"] = (idx, nb, w)
            return self._win["box"]

    paths = _parse_kv(args.trajectory_npz, "trajectory-npz")
    titles = _parse_kv(args.title, "title")
    if len(paths) != 3:
        raise SystemExit("Supply three model NPZ files; the first supplies the fourth target panel.")
    panels, sources, reference = [], [], None
    for key, path in paths.items():
        with np.load(path, allow_pickle=False) as z:
            seeds = np.asarray(z["seeds"])
            selected = np.flatnonzero(seeds == args.seed)
            if len(selected) != 1:
                raise ValueError(f"Seed {args.seed} missing or duplicated: {path}")
            geo = {k: np.asarray(z[k]).copy() for k in ("lat", "lon", "center_lat", "center_lon")}
            truth = {v: np.asarray(z[f"truth_{v}"], dtype=float) for v in ("msl", "10u", "10v", "2t")}
            if reference is None:
                reference = (geo, truth)
            else:
                for k in geo:
                    if not np.array_equal(geo[k], reference[0][k]):
                        raise ValueError(f"Geometry mismatch for {key}: {k}")
                for v in truth:
                    if not np.array_equal(truth[v], reference[1][v]):
                        raise ValueError(f"Target mismatch for {key}: {v}")
            fld = {v: np.asarray(z[f"free_{v}"][selected[0]], dtype=float) for v in truth}
            panels.append((titles.get(key, key), fld))
            sources.append({"key": key, "path": str(Path(path).resolve()), "sha256": _hash(path),
                            "saved_seeds": seeds.tolist(), "selected_index": int(selected[0]),
                            "free_sigma": float(z["free_sigma"])})
    panels.append(("Truth (independent ENFO realization)", reference[1]))
    geo = reference[0]
    geom = BoxGeometry(geo["lat"], (geo["lon"] + 180) % 360 - 180)
    prepared = []
    for title, fld in panels:
        if not all(np.isfinite(a).all() for a in fld.values()):
            raise ValueError(f"Non-finite saved fields: {title}")
        if not 80000 < np.median(fld["msl"]) < 110000:
            raise ValueError("Saved pressure must be in Pa.")
        fld["tcw"] = np.zeros_like(fld["msl"])
        latg, long, _ = T.smoothed_min_near(geom, fld["msl"], float(geo["center_lat"]),
                                           float(geo["center_lon"]), 400.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rec, _, _, _ = T.analyse_case(geom, fld, "franklin", first_guess=(latg, long))
        if not all(np.isfinite(rec[k]) for k in ("lat_p", "lon_p")):
            raise ValueError(f"Invalid pressure centre: {title}")
        L = T.LocalGrid(geom, latg, long)
        up, vp = L.wind_to_plane(fld["10u"][L.idx], fld["10v"][L.idx])
        gr = L.interp(np.c_[up, vp])
        U, V = gr[..., 0], gr[..., 1]
        WS = np.hypot(U, V)
        s1, s2 = T.BANDS["b40_150"]
        band = T.nangauss(WS, s1) - T.nangauss(WS, s2)
        xp, yp = T.ae_forward(rec["lat_p"], rec["lon_p"], latg, long)
        prepared.append(dict(title=title, U=U, V=V, ws=WS, band=band,
                             x=T.GX-xp, y=T.GX-yp, record=rec))
    target = prepared[-1]["record"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"command": shlex.join(["python", "-m", "eval.cli", *sys.argv[1:]]),
                "working_directory": str(Path.cwd()), "sources": sources, "seed": args.seed,
                "instrument": str(instrument), "instrument_sha256": _hash(instrument),
                "grid_spacing_km": T.DX, "band_gaussian_sigmas_km": [s1, s2],
                "half_width_km": args.storm_half_width_km,
                "recenter": "Each panel uses its own refined pressure centre; offsets refer to target.",
                "target": "Same-index independent ENFO realization; not paired pixelwise truth.",
                "sample": "One free sample per model, not restart fields or an ensemble mean.",
                "panels": [], "outputs": []}
    for p in prepared:
        r = p["record"]
        p["offset"] = float(T.gc_dist(target["lat_p"], target["lon_p"], r["lat_p"], r["lon_p"]))
        manifest["panels"].append({"title": p["title"], "centre_lat": r["lat_p"], "centre_lon": r["lon_p"],
                                   "offset_from_target_km": p["offset"], "spiral_angle_deg": r["orient_ws_sgn_med"]})
    init = datetime.strptime(str(args.date) + str(args.time).zfill(4), "%Y%m%d%H%M")
    from eval.plotting import eval_style, save_figure, variable_spec

    ws_spec = variable_spec("10ff")
    for field in ("ws", "band"):
        with eval_style():
            fig, axes = plt.subplots(2, 2, figsize=(12, 11))
            fig.subplots_adjust(left=.075, right=.87, bottom=.11, top=.865, hspace=.29, wspace=.23)
            vmin, vmax = ((0 if args.vmin is None else args.vmin, 60 if args.vmax is None else args.vmax)
                          if field == "ws" else (-args.band_vmax, args.band_vmax))
            for ax, p in zip(axes.flat, prepared):
                mesh = ax.pcolormesh(p["x"], p["y"], p[field], shading="auto", rasterized=True,
                                     cmap=ws_spec.field_cmap() if field == "ws" else "RdBu_r", vmin=vmin, vmax=vmax)
                ax.set_facecolor("#e6e6e6")
                ax.plot(0, 0, marker="+", color="black", ms=8, mew=1)
                if field == "ws":
                    sl = slice(None, None, 12)
                    X, Y = np.meshgrid(p["x"][sl], p["y"][sl])
                    q = ax.quiver(X, Y, p["U"][sl, sl], p["V"][sl, sl], color="white",
                                  scale=650, width=.003, headwidth=3.5)
                r = p["record"]
                ax.set_title(f'{p["title"]}\nCentre: {r["lat_p"]:.2f}°N, {abs(r["lon_p"]):.2f}°W; offset {p["offset"]:.0f} km',
                             fontsize=10, pad=8)
                ax.set(xlim=(-args.storm_half_width_km, args.storm_half_width_km),
                       ylim=(-args.storm_half_width_km, args.storm_half_width_km), aspect="equal",
                       xlabel="East of own centre (km)", ylabel="North of own centre (km)")
                ax.tick_params(labelsize=9)
                ax.grid(False)
            cax = fig.add_axes([.90, .21, .019, .54])
            cb = fig.colorbar(mesh, cax=cax, extend="max" if field == "ws" else "both")
            cb.set_label(ws_spec.label if field == "ws" else f"40–150 km wind-speed component ({ws_spec.unit})")
            if field == "ws":
                axes.flat[-1].quiverkey(q, .50, .072, 30, f"30 {ws_spec.unit}", coordinates="figure", labelpos="E")
            view = "Full 10 m wind speed and direction" if field == "ws" else "40–150 km wind-speed features"
            fig.suptitle(f'{args.region_tag.capitalize()}: {view}', fontsize=15, y=.965)
            fig.text(.5, .925, f'Initialization {init:%d %b %Y %H:%M} UTC · lead +{args.step} h · member {args.member:02d} · free seed {args.seed}',
                     ha="center", fontsize=11)
            note = "Colours show full speed; arrows show wind direction and strength." if field == "ws" else "Colours show a difference-of-Gaussians band-pass, not model-minus-input residuals."
            fig.text(.5, .044, note, ha="center", fontsize=10)
            fig.text(.5, .024, "Each storm is recentered. Linear interpolation to a 5 km local grid; grey denotes unavailable data.",
                     ha="center", fontsize=9)
            path = out / f'{args.region_tag}_{field}_seed{args.seed}.png'
            save_figure(fig, path, close=True, tight=False)
        manifest["outputs"].append({"path": str(path), "vmin": vmin, "vmax": vmax, "field": field})
        print(path, flush=True)
    manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0
