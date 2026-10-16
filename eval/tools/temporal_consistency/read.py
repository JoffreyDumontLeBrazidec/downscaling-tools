"""The temporal-consistency read (see the package docstring).

    python -m eval.tools.temporal_consistency.read \\
        --run independent=<pred_dir_a> --run shared=<pred_dir_b> \\
        --output-dir <dir> [--box e2] [--variables 10u,10v,ws,2t,msl,tp] \\
        [--members 1,2,3] [--dates 20230226,...] [--storm-box 10,40,-80,-40]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval.discovery.predictions import find_predictions
from eval.evaluators.box_spectra.core.spectrum import SPEC_EDGE_MARGIN_DEG, SPEC_RES_DEG, Sampler
from eval.evaluators.box_spectra.runner import COMPOSITES, _contiguous, _File, resolve_box

LOG = logging.getLogger(__name__)

KM_PER_DEG = 111.195
# the ladder of standing rule 14: truth ends 31.3 km, mesh resolves 112 km, input ends 125.5 km
BANDS = {"31-56km": (31.3, 56.0), "56-125km": (56.0, 125.5), "gt125km": (125.5, np.inf)}
FIELDS = {"pred": "y_pred", "truth": "y", "interp": "x_interp"}
DEFAULT_VARIABLES = ["10u", "10v", "ws", "2t", "msl", "tp"]
EDGE_FRAC = 0.1


class BandPass:
    """Isotropic wavelength band masks on the regular grid of a Sampler."""

    def __init__(self, ny: int, nx: int, dx_km: float, dy_km: float):
        kx = np.fft.fftfreq(nx, d=dx_km)
        ky = np.fft.fftfreq(ny, d=dy_km)
        KX, KY = np.meshgrid(kx, ky)
        k = np.hypot(KX, KY)
        with np.errstate(divide="ignore"):
            wl = np.where(k > 0, 1.0 / np.maximum(k, 1e-30), np.inf)
        self.masks = {name: (wl >= lo) & (wl < hi) for name, (lo, hi) in BANDS.items()}
        self.ny, self.nx = ny, nx

    def __call__(self, grid: np.ndarray) -> dict[str, np.ndarray]:
        F = np.fft.fft2(grid - np.mean(grid))
        return {name: np.real(np.fft.ifft2(F * m)) for name, m in self.masks.items()}


def _interior(shape: tuple[int, int], frac: float) -> tuple[slice, slice]:
    ny, nx = shape
    my, mx = int(round(ny * frac)), int(round(nx * frac))
    return slice(my, ny - my), slice(mx, nx - mx)


def _weights(lat_rows: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    w = np.cos(np.deg2rad(lat_rows))[:, None] * np.ones((1, shape[1]))
    return w


def _wmean(a: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(a * w) / np.sum(w))


def _wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    ma, mb = _wmean(a, w), _wmean(b, w)
    cov = _wmean((a - ma) * (b - mb), w)
    va, vb = _wmean((a - ma) ** 2, w), _wmean((b - mb) ** 2, w)
    return float(cov / np.sqrt(max(va * vb, 1e-300)))


def read_run(pred_dir: Path, *, variables, members, dates, steps, box_spec, res_deg, margin_deg,
             storm_box, storm_radius_deg) -> tuple[list[dict], list[dict], dict]:
    files = find_predictions(Path(pred_dir))
    by_date: dict[str, dict[int, Path]] = defaultdict(dict)
    for pf in files:
        if dates and pf.date not in dates:
            continue
        if steps and pf.step not in steps:
            continue
        by_date[pf.date][pf.step] = Path(pf.path)
    if not by_date:
        raise SystemExit(f"{pred_dir}: no predictions_*.nc for the requested dates and steps")

    smp = bp = None
    meta: dict = {}
    grid_lat = grid_lon = None
    # (variable, band, field) -> lists over (date, member, pair)
    diff_acc: dict[tuple, list[float]] = defaultdict(list)
    corr_acc: dict[tuple, list[float]] = defaultdict(list)
    amp_acc: dict[tuple, list[float]] = defaultdict(list)
    storm_rows: list[dict] = []
    for date in sorted(by_date):
        step_list = sorted(by_date[date])
        if len(step_list) < 2:
            LOG.warning("%s: only one step on disk, skipped", date)
            continue
        opened = {st: _File(by_date[date][st]) for st in step_list}
        try:
            f0 = opened[step_list[0]]
            if smp is None:
                box, how = resolve_box(box_spec, f0.lat, f0.lon)
                la0, la1, lo0, lo1 = box
                extent = (la0 + margin_deg, la1 - margin_deg, lo0 + margin_deg,
                          lo0 + ((lo1 - lo0) % 360.0 or 360.0) - margin_deg)
                lon_c = _contiguous(f0.lon, lo0)
                smp = Sampler(f0.lat, lon_c, extent, res_deg, method="linear")
                dx, dy = smp.dx_dy_km()
                bp = BandPass(len(smp.gy), len(smp.gx), dx, dy)
                grid_lat, grid_lon = f0.lat, f0.lon
                meta = {"box": box, "box_chosen_by": how, "extent_sampled": list(extent),
                        "grid": [len(smp.gy), len(smp.gx)], "res_deg": res_deg, "dx_km": dx, "dy_km": dy,
                        "edge_frac": EDGE_FRAC, "bands_km": {k: [v[0], None if np.isinf(v[1]) else v[1]]
                                                             for k, v in BANDS.items()}}
                inner = _interior(smp.shape, EDGE_FRAC)
                w = _weights(smp.gy, smp.shape)[inner]
            for f in opened.values():
                if f.lat.shape != grid_lat.shape or np.abs(f.lat - grid_lat).max() > 1e-6 \
                        or np.abs(f.lon - grid_lon).max() > 1e-6:
                    raise ValueError(f"{date}: point set differs between steps")
            idx = smp.src_idx
            mems = [m for m in f0.members if not members or m in members]

            # One netCDF read per (file, variable name, member): the (grid, state) block of that member,
            # then every weather state is picked from memory. Reading one state at a time through
            # ``_File.field`` decompresses the member's whole block once per state and per field, which
            # made the first E2 read take about two minutes per member and step (16 Oct 2026).
            blocks: dict[tuple[int, str, int], np.ndarray] = {}

            def block_of(f, st, name, m):
                key = (st, name, m)
                if key not in blocks:
                    v = f.ds[name]
                    sel = []
                    for d in v.dimensions:
                        if d == "sample":
                            sel.append(0)
                        elif d == "ensemble_member":
                            sel.append(f.members.index(m))
                        else:
                            sel.append(slice(None))
                    arr = np.asarray(v[tuple(sel)], dtype=np.float32)
                    if v.dimensions.index("weather_state") < v.dimensions.index("grid_point_hres"):
                        arr = arr.T
                    blocks[key] = arr          # (grid, state)
                return blocks[key]

            def grid_of(f, st, name, m, var):
                blk = block_of(f, st, name, m)
                if var in COMPOSITES:
                    u, w = (blk[idx, f.states.index(c)] for c in COMPOSITES[var])
                    vals = np.hypot(u, w)
                else:
                    vals = blk[idx, f.states.index(var)]
                full = np.zeros(f.lat.size)
                full[idx] = vals
                return smp(full)

            for m in mems:
                blocks.clear()
                for var in variables:
                    if not f0.has(var):
                        continue
                    bands_prev = None
                    for st in step_list:
                        f = opened[st]
                        bands_now = {fld: bp(grid_of(f, st, name, m, var)) for fld, name in FIELDS.items()}
                        for fld in FIELDS:
                            for band, g in bands_now[fld].items():
                                amp_acc[(var, band, fld)].append(float(np.sqrt(_wmean(g[inner] ** 2, w))))
                        if bands_prev is not None:
                            for fld in FIELDS:
                                for band in BANDS:
                                    a, b = bands_prev[fld][band][inner], bands_now[fld][band][inner]
                                    diff_acc[(var, band, fld)].append(_wmean(np.abs(b - a), w))
                                    corr_acc[(var, band, fld)].append(_wcorr(a, b, w))
                        bands_prev = bands_now
            # storm continuity on msl and ws, if a search box is given and the variables exist
            if storm_box and f0.has("msl"):
                storm_rows.extend(_storm_sequence(opened, step_list, date, mems, storm_box, storm_radius_deg))
        finally:
            for f in opened.values():
                f.close()

    rows = []
    for (var, band, fld) in sorted(diff_acc):
        rows.append({"variable": var, "band": band, "field": fld,
                     "mean_abs_diff": float(np.mean(diff_acc[(var, band, fld)])),
                     "lag1_corr": float(np.mean(corr_acc[(var, band, fld)])),
                     "band_rms": float(np.mean(amp_acc[(var, band, fld)])),
                     "n_pairs": len(diff_acc[(var, band, fld)])})
    return rows, storm_rows, meta


def _storm_sequence(opened, step_list, date, mems, storm_box, radius_deg) -> list[dict]:
    la0, la1, lo0, lo1 = storm_box
    f0 = opened[step_list[0]]
    lon_n = (f0.lon + 180.0) % 360.0 - 180.0
    in_box = (f0.lat >= la0) & (f0.lat <= la1) & (lon_n >= lo0) & (lon_n <= lo1)
    if not np.any(in_box):
        return []
    idx_box = np.flatnonzero(in_box)
    rows = []
    seq: dict[tuple[str, int], list[float]] = defaultdict(list)   # (field, member) -> values per step
    for st in step_list:
        f = opened[st]
        msl_true = f.column("y", f.members[0], "msl", idx_box)
        j = int(np.argmin(msl_true))
        clat, clon = float(f.lat[idx_box[j]]), float(lon_n[idx_box[j]])
        d = np.hypot(f.lat[idx_box] - clat, (lon_n[idx_box] - clon) * np.cos(np.deg2rad(clat)))
        near = idx_box[d <= radius_deg]
        for fld, name in FIELDS.items():
            for m in (mems if fld == "pred" else [f.members[0]]):
                msl = f.column(name, m, "msl", near)
                seq[("msl_min", fld, m)].append(float(np.min(msl)))
                if f.has("ws"):
                    seq[("ws_max", fld, m)].append(float(np.max(f.field(name, m, "ws", near))))
        rows.append({"date": date, "step": st, "centre_lat": clat, "centre_lon": clon,
                     "truth_msl_min": float(msl_true[j])})
    out = []
    for (quant, fld, m), vals in sorted(seq.items()):
        v = np.asarray(vals)
        changes = np.diff(v)
        out.append({"date": date, "quantity": quant, "field": fld, "member": m,
                    "values": json.dumps([round(x, 2) for x in v.tolist()]),
                    "mean_abs_change": float(np.mean(np.abs(changes))) if changes.size else float("nan"),
                    "std_change": float(np.std(changes)) if changes.size > 1 else float("nan")})
    for r in rows:
        r.update({"quantity": "centre", "field": "truth", "member": 0})
    return out + rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def write_read(runs: dict[str, tuple[list[dict], list[dict]]], meta: dict, path: Path) -> None:
    names = list(runs)
    lines = ["# Temporal consistency read (TD-227)", "",
             f"Box {meta.get('box')} ({meta.get('box_chosen_by')}), grid {meta.get('grid')} at {meta.get('res_deg')} deg, "
             f"interior margin {meta.get('edge_frac')} of each side; bands 31-56, 56-125 and >125 km.", "",
             "Quantities per variable and band, pooled over dates, members and consecutive lead pairs:",
             "mean |f(L+dt) - f(L)| (area-weighted) and lag-1 correlation of the band-passed field.",
             "`ratio` = prediction over truth; the target is 1.0 (match the truth's own lead-to-lead change),",
             "above 1 the prediction flickers more than the truth, below 1 it is smoother in time than the truth.", ""]
    for name in names:
        rows, _ = runs[name]
        by = {(r["variable"], r["band"], r["field"]): r for r in rows}
        variables = sorted({r["variable"] for r in rows})
        lines += [f"## Run `{name}`", "",
                  "| variable | band | mean abs diff pred | truth | input | ratio pred/truth | lag-1 corr pred | truth | input | band rms pred/truth | n pairs |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for var in variables:
            for band in BANDS:
                p, t, i = (by.get((var, band, f)) for f in ("pred", "truth", "interp"))
                if not (p and t):
                    continue
                lines.append(f"| {var} | {band} | {p['mean_abs_diff']:.4g} | {t['mean_abs_diff']:.4g} | "
                             f"{i['mean_abs_diff'] if i else float('nan'):.4g} | "
                             f"{p['mean_abs_diff'] / max(t['mean_abs_diff'], 1e-30):.3f} | "
                             f"{p['lag1_corr']:.3f} | {t['lag1_corr']:.3f} | {i['lag1_corr'] if i else float('nan'):.3f} | "
                             f"{p['band_rms'] / max(t['band_rms'], 1e-30):.3f} | {p['n_pairs']} |")
        lines.append("")
    if len(names) >= 2:
        a, b = names[0], names[1]
        ra = {(r["variable"], r["band"]): r for r in runs[a][0] if r["field"] == "pred"}
        rb = {(r["variable"], r["band"]): r for r in runs[b][0] if r["field"] == "pred"}
        lines += [f"## `{b}` over `{a}` (prediction only)", "",
                  "| variable | band | mean abs diff ratio | lag-1 corr a | lag-1 corr b | band rms ratio |",
                  "|---|---|---|---|---|---|"]
        for key in sorted(set(ra) & set(rb)):
            x, y = ra[key], rb[key]
            lines.append(f"| {key[0]} | {key[1]} | {y['mean_abs_diff'] / max(x['mean_abs_diff'], 1e-30):.3f} | "
                         f"{x['lag1_corr']:.3f} | {y['lag1_corr']:.3f} | "
                         f"{y['band_rms'] / max(x['band_rms'], 1e-30):.3f} |")
        lines.append("")
    for name in names:
        _, storms = runs[name]
        seqs = [r for r in storms if r.get("quantity") in ("msl_min", "ws_max")]
        if not seqs:
            continue
        lines += [f"## Storm continuity, run `{name}` (truth minimum tracked in the search box; std of the lead-to-lead change)", "",
                  "| date | quantity | field | member | values per step | mean abs change | std change |",
                  "|---|---|---|---|---|---|---|"]
        for r in seqs:
            lines.append(f"| {r['date']} | {r['quantity']} | {r['field']} | {r['member']} | {r['values']} | "
                         f"{r['mean_abs_change']:.3g} | {r['std_change']:.3g} |")
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PRED_DIR",
                    help="a predictions directory holding consecutive steps; repeat for a pair")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variables", default=",".join(DEFAULT_VARIABLES))
    ap.add_argument("--members", default="", help="comma-separated member labels (default all)")
    ap.add_argument("--dates", default="", help="comma-separated YYYYMMDD (default all on disk)")
    ap.add_argument("--steps", default="", help="comma-separated steps (default all on disk)")
    ap.add_argument("--box", default="", help="e2, alps, north_atlantic, tc_atlantic, or 4 numbers (default: file bounds / e2)")
    ap.add_argument("--res-deg", type=float, default=SPEC_RES_DEG)
    ap.add_argument("--margin-deg", type=float, default=SPEC_EDGE_MARGIN_DEG)
    ap.add_argument("--storm-box", default="10,45,-80,-30",
                    help="lat_min,lat_max,lon_min,lon_max where the truth's msl minimum is tracked ('' to skip)")
    ap.add_argument("--storm-radius-deg", type=float, default=4.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    variables = [v for v in args.variables.split(",") if v]
    members = [int(m) for m in args.members.split(",") if m]
    dates = [d for d in args.dates.split(",") if d]
    steps = [int(s) for s in args.steps.split(",") if s]
    storm_box = [float(v) for v in args.storm_box.split(",")] if args.storm_box else None
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs: dict[str, tuple[list[dict], list[dict]]] = {}
    meta: dict = {}
    for spec in args.run:
        name, _, pred_dir = spec.partition("=")
        if not pred_dir:
            raise SystemExit(f"--run expects NAME=PRED_DIR, got {spec!r}")
        rows, storms, meta = read_run(Path(pred_dir), variables=variables, members=members, dates=dates,
                                      steps=steps, box_spec=args.box, res_deg=args.res_deg,
                                      margin_deg=args.margin_deg, storm_box=storm_box,
                                      storm_radius_deg=args.storm_radius_deg)
        for r in rows:
            r["run"] = name
        for r in storms:
            r["run"] = name
        runs[name] = (rows, storms)
        LOG.info("%s: %d band rows, %d storm rows", name, len(rows), len(storms))
    write_csv([r for rows, _ in runs.values() for r in rows], out / "bands.csv")
    write_csv([r for _, storms in runs.values() for r in storms], out / "storms.csv")
    (out / "meta.json").write_text(json.dumps({"meta": meta, "runs": {n: str(s.partition('=')[2]) for n, s in
                                                                     zip(runs, args.run)}}, indent=2))
    write_read(runs, meta, out / "read.md")
    print(f"wrote {out / 'read.md'}")


if __name__ == "__main__":
    main()
