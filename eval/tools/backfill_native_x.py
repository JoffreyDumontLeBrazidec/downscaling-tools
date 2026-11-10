#!/usr/bin/env python3
"""Add the native input x to existing ENS-member prediction files (rule 41).

Rule 41 (owner, 9 Nov 2026): the input is scored on its own grid, never after interpolation
to the output grid. Prediction files written before the predictors saved the native input
carry only x_interp / x_interp_members (the input interpolated to the output grid). This tool
APPENDS, in netCDF4 append mode, the native input of each draw's own input member,

    x        (sample, ensemble_member, grid_point_lres, weather_state)  float32
    lat_lres (grid_point_lres)  float64,  lon_lres (grid_point_lres)  float64

read from the zarr member store the predictions were made from, and the file attribute
``native_x_backfill`` (store, row, date, lead, checks). It never changes an existing variable,
and it skips files that already carry x. Dry run by default; ``--apply`` writes.

Store layout (both ENS-member lanes): a zarr group with ``data[row, variable, member, point]``,
attrs ``variables`` and ``row_map`` (one entry per row with ``hdate``, ``lead``, ``fake``), and
``latitudes``/``longitudes``. The 4 km lane (o1280 -> o2560) keeps tp and cp in a second store
of the same layout (``member_tp_store``). By default the stores are the ones the file records
in its attributes ``member_store`` and ``member_tp_store``; ``--store``/``--tp-store`` override.
Draw k (0-based, ensemble_member k+1) is conditioned on store member index k (input member pf k+1).

Built-in checks, per file, before anything is written (a failed check aborts that file):
  * pairing, on msl only (never on rain): the native x of every member is mapped to the output
    points by nearest native point, and its deviation from the member mean must correlate best
    with the deviation of x_interp_members[k] (or, without x_interp_members, of y_pred[k]) for
    the same k;
  * units: per weather state, the mapped x must match x_interp_members (or x_interp) in mean
    and spread (tp and cp in metres, as the file stores them); tp and cp are clipped at 0 when
    the file's input members were (attribute precip_clipped_input_members), as the predictor did;
  * the row: the store row of (date, lead) must carry the file's fake date when both record one.

Usage: python -m eval.tools.backfill_native_x PATH [PATH ...] [--apply] [--workers N] [--report R.json]
PATH is a predictions_<date>_step<lead>.nc file or a directory of them.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np

PRED_RE = re.compile(r"predictions_(\d{8})_step(\d{3})\.nc$")
PRECIP = ("tp", "cp")
PAIRING_STATE = "msl"
CHECK_BANDS = 4              # contiguous bands of output points read for the checks
CHECK_BAND_POINTS = 100_000
TOOL = "eval/tools/backfill_native_x.py"


class CheckFailed(RuntimeError):
    """A built-in check refused the file; nothing was written."""


def _xyz(lat, lon) -> np.ndarray:
    la = np.radians(np.asarray(lat, dtype=np.float64).reshape(-1))
    lo = np.radians(np.asarray(lon, dtype=np.float64).reshape(-1))
    c = np.cos(la)
    return np.column_stack((c * np.cos(lo), c * np.sin(lo), np.sin(la)))


def _states(var) -> list[str]:
    raw = np.asarray(var[:]).reshape(-1)
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]


def _attr(ds, name: str, default=None):
    return ds.getncattr(name) if name in ds.ncattrs() else default


class _Store:
    """One zarr member store: rows by (hdate, lead), variables by name."""

    def __init__(self, path: str):
        import zarr

        self.path = str(path)
        self.g = zarr.open_group(self.path, mode="r")
        attrs = dict(self.g.attrs)
        self.variables = [str(v) for v in attrs["variables"]]
        self.row_map = list(attrs["row_map"])
        self.rows = {(str(r["hdate"]), int(r["lead"])): i for i, r in enumerate(self.row_map)}
        self.members = attrs.get("members")
        self.lat = np.asarray(self.g["latitudes"][:], dtype=np.float64)
        self.lon = np.asarray(self.g["longitudes"][:], dtype=np.float64)
        shape = self.g["data"].shape
        if len(shape) != 4 or shape[1] != len(self.variables) or shape[3] != self.lat.size:
            raise ValueError(f"{self.path}: data shape {shape} is not (row, {len(self.variables)} variables, "
                             f"member, {self.lat.size} points)")
        self.n_members = int(shape[2])

    def row(self, hdate: str, lead: int) -> int:
        if (hdate, lead) not in self.rows:
            raise CheckFailed(f"{self.path} has no row for {hdate} +{lead} h")
        return self.rows[(hdate, lead)]

    def field(self, row: int, var: str) -> np.ndarray:
        """(member, point) of one variable."""
        return np.asarray(self.g["data"][row, self.variables.index(var)])


def _bands(n: int) -> list[slice]:
    band = min(CHECK_BAND_POINTS, max(1, n // CHECK_BANDS))
    starts = np.unique(np.linspace(0, n - band, CHECK_BANDS).astype(int))
    return [slice(int(s), int(s) + band) for s in starts]


def _read_bands(var, bands, n_lead: int) -> np.ndarray:
    """(member, points of the bands, state) of a (sample, [member,] point, state) variable."""
    parts = []
    for b in bands:
        if n_lead == 2:
            parts.append(np.asarray(var[0, :, b, :], dtype=np.float64))
        else:
            parts.append(np.asarray(var[0, b, :], dtype=np.float64)[None])
    return np.concatenate(parts, axis=1)


def pairing_check(x_msl_mapped: np.ndarray, ref_msl: np.ndarray) -> dict:
    """Correlation of member deviations (member minus member mean), x (rows) against the reference (columns).

    Passes when every member k of the reference correlates best with x member k.
    """
    a = x_msl_mapped - x_msl_mapped.mean(axis=0, keepdims=True)
    b = ref_msl - ref_msl.mean(axis=0, keepdims=True)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    corr = (b @ a.T) / np.maximum(np.outer(nb, na), 1e-300)        # [ref k, x j]
    best = corr.argmax(axis=1)
    n = corr.shape[0]
    off = corr[~np.eye(n, dtype=bool)] if n > 1 else np.array([np.nan])
    return {
        "ok": bool(np.array_equal(best, np.arange(n))),
        "diag_min": float(np.diag(corr).min()),
        "diag_mean": float(np.diag(corr).mean()),
        "offdiag_max": float(np.nanmax(off)),
        "best_match": [int(v) for v in best],
    }


def units_check(x_mapped: np.ndarray, ref: np.ndarray, states: list[str]) -> dict:
    """Per state: mapped x and the reference agree in mean and spread (catches mm vs m, K vs degC)."""
    out = {}
    for j, s in enumerate(states):
        xv, rv = x_mapped[..., j].ravel(), ref[..., j].ravel()
        ok_ = np.isfinite(xv) & np.isfinite(rv)
        xv, rv = xv[ok_], rv[ok_]
        if xv.size == 0:
            out[s] = {"ok": False, "why": "no finite values"}
            continue
        mx, mr, sx, sr = float(xv.mean()), float(rv.mean()), float(xv.std()), float(rv.std())
        scale = max(sr, sx, 1e-12)
        ratio = (sx / sr) if sr > 1e-12 else (1.0 if sx <= 1e-12 else float("inf"))
        good = abs(mx - mr) <= 0.25 * scale + 1e-12 and 0.5 <= ratio <= 2.0
        out[s] = {"ok": bool(good), "mean_x": mx, "mean_ref": mr, "std_x": sx, "std_ref": sr}
    return out


def prepare(path: Path, store_override: str | None = None, tp_override: str | None = None, stores=None) -> dict:
    """Read and check one file; return the plan (arrays included) or raise CheckFailed."""
    import netCDF4

    stores = {} if stores is None else stores
    m = PRED_RE.search(path.name)
    if not m:
        raise CheckFailed(f"{path.name} is not predictions_<date>_step<lead>.nc")
    hdate, lead = m.group(1), int(m.group(2))
    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        if "x" in ds.variables:
            status = _attr(ds, "native_x_backfill_status", "")
            if status != "writing":
                return {"file": str(path), "action": "skip", "why": "already carries x"}
        states = _states(ds["weather_state"])
        n_m = len(ds.dimensions["ensemble_member"])
        ids = [int(v) for v in np.asarray(ds["ensemble_member"][:]).reshape(-1)] \
            if "ensemble_member" in ds.variables else list(range(1, n_m + 1))
        if ids != list(range(1, n_m + 1)):
            raise CheckFailed(f"ensemble_member is {ids}, not 1..{n_m}; draw k <-> store member k is not established")
        if _attr(ds, "lead_step_hours") is not None and int(_attr(ds, "lead_step_hours")) != lead:
            raise CheckFailed(f"lead_step_hours {_attr(ds, 'lead_step_hours')} differs from the file name's +{lead} h")
        store_path = store_override or _attr(ds, "member_store")
        tp_path = tp_override or _attr(ds, "member_tp_store")
        if not store_path or store_path == "control":
            raise CheckFailed("no member store (attribute member_store absent; pass --store)")
        fake = _attr(ds, "fake_date")
        clip_precip = _attr(ds, "precip_clipped_input_members") is not None
        lat_h = np.asarray(ds["lat_hres"][:], dtype=np.float64).reshape(-1)
        lon_h = np.asarray(ds["lon_hres"][:], dtype=np.float64).reshape(-1)
        bands = _bands(lat_h.size)
        has_members = "x_interp_members" in ds.variables
        ref_name = "x_interp_members" if has_members else "y_pred"
        ref_pair = _read_bands(ds[ref_name], bands, 2)                     # (member, pts, state)
        unit_name = "x_interp_members" if has_members else "x_interp"
        ref_units = ref_pair if has_members else _read_bands(ds["x_interp"], bands, 1)
        y_pred_probe = np.asarray(ds["y_pred"][0, 0, :1000, :])
    if PAIRING_STATE not in states:
        raise CheckFailed(f"the file has no {PAIRING_STATE}; member pairing cannot be checked (never on rain)")

    for p in [store_path] + ([tp_path] if tp_path else []):
        if p not in stores:
            stores[p] = _Store(p)
    main = stores[store_path]
    tps = stores[tp_path] if tp_path else None
    row = main.row(hdate, lead)
    entry = main.row_map[row]
    if fake and entry.get("fake"):
        if str(entry["fake"])[:19] != str(fake)[:19]:
            raise CheckFailed(f"store row {row} has fake date {entry['fake']}, the file {fake}")
    if main.n_members != n_m:
        raise CheckFailed(f"store has {main.n_members} members, the file {n_m} draws")
    tp_row = None
    if tps is not None:
        tp_row = tps.row(hdate, lead)
        if np.abs(tps.lat - main.lat).max() > 1e-6 or tps.n_members != n_m:
            raise CheckFailed("the tp store's points or members differ from the member store's")

    x = np.empty((n_m, main.lat.size, len(states)), dtype=np.float32)
    source, clipped = {}, {}
    for j, s in enumerate(states):
        if tps is not None and s in tps.variables:
            vals, source[s] = tps.field(tp_row, s), "tp_store"
        elif s in main.variables:
            vals, source[s] = main.field(row, s), "store"
        else:
            raise CheckFailed(f"weather state {s} is in neither store")
        vals = np.asarray(vals, dtype=np.float64)
        if s in PRECIP and clip_precip:
            clipped[s] = int((vals < 0).sum())
            vals = np.clip(vals, 0.0, None)
        x[:, :, j] = vals
    if not np.isfinite(x).all():
        raise CheckFailed("non-finite values in the native input")

    sub = np.concatenate([np.arange(b.start, b.stop) for b in bands])
    from scipy.spatial import cKDTree
    _, nn = cKDTree(_xyz(main.lat, main.lon)).query(_xyz(lat_h[sub], lon_h[sub]), k=1)
    x_mapped = x[:, nn, :].astype(np.float64)                             # nearest native point, no interpolation
    jm = states.index(PAIRING_STATE)
    pairing = pairing_check(x_mapped[:, :, jm], ref_pair[:, :, jm])
    pairing["reference"] = ref_name
    units = units_check(x_mapped if has_members else x_mapped.mean(axis=0, keepdims=True), ref_units, states)
    bad_units = [s for s, u in units.items() if not u["ok"]]
    plan = {
        "file": str(path), "action": "write", "hdate": hdate, "lead": lead,
        "store": store_path, "tp_store": tp_path or "", "row": int(row), "tp_row": tp_row,
        "store_row_entry": {k: entry.get(k) for k in ("hdate", "lead", "fake", "row", "store_row") if k in entry},
        "fake_date": str(fake) if fake else None, "store_members": main.members,
        "n_members": n_m, "n_points_lres": int(main.lat.size), "states": states, "state_source": source,
        "precip_clipped": clipped, "pairing": pairing, "units_reference": unit_name, "units": units,
        "x_megabytes": round(x.nbytes / 2**20, 1),
    }
    if not pairing["ok"]:
        raise CheckFailed(f"member pairing on msl failed: {json.dumps(pairing)}")
    if bad_units:
        raise CheckFailed(f"units differ from {unit_name} for {bad_units}: {json.dumps({s: units[s] for s in bad_units})}")
    plan["_x"], plan["_lat"], plan["_lon"], plan["_probe"] = x, main.lat, main.lon, y_pred_probe
    return plan


def apply(plan: dict) -> None:
    """Append x, lat_lres, lon_lres and the attributes; existing variables are never touched."""
    import netCDF4

    path = plan["file"]
    with netCDF4.Dataset(path, "a") as ds:
        before = {k: (v.dimensions, v.shape) for k, v in ds.variables.items()}
        ds.setncattr("native_x_backfill_status", "writing")
        n_p = plan["n_points_lres"]
        if "grid_point_lres" not in ds.dimensions:
            ds.createDimension("grid_point_lres", n_p)
        elif len(ds.dimensions["grid_point_lres"]) != n_p:
            raise CheckFailed("the file already has a grid_point_lres dimension of another size")
        for name, arr in (("lat_lres", plan["_lat"]), ("lon_lres", plan["_lon"])):
            v = ds[name] if name in ds.variables else ds.createVariable(name, "f8", ("grid_point_lres",))
            v[:] = arr
        v = ds["x"] if "x" in ds.variables else ds.createVariable(
            "x", "f4", ("sample", "ensemble_member", "grid_point_lres", "weather_state"))
        v.setncattr("description", "native input of each draw's own input member (store member k for draw k), "
                                   "on lat_lres/lon_lres; added by " + TOOL)
        v[0] = plan["_x"]
        meta = {k: plan[k] for k in ("store", "tp_store", "row", "tp_row", "hdate", "lead", "store_row_entry",
                                     "fake_date", "store_members", "state_source", "precip_clipped")}
        meta.update(tool=TOOL, utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    rule="rule 41: the input is scored on its own grid",
                    pairing="draw k (0-based) conditioned on store member index k (input member pf k+1)")
        ds.setncattr("native_x_backfill", json.dumps(meta))
        ds.setncattr("native_x_pairing_check", json.dumps(plan["pairing"]))
        ds.setncattr("native_x_backfill_status", "complete")
        for k, (dims, shape) in before.items():
            if (ds[k].dimensions, ds[k].shape) != (dims, shape):
                raise RuntimeError(f"{path}: variable {k} changed shape during the backfill")
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        if not np.array_equal(np.asarray(ds["y_pred"][0, 0, :1000, :]), plan["_probe"]):
            raise RuntimeError(f"{path}: y_pred differs after the backfill")
        if not np.array_equal(np.asarray(ds["x"][0, :, :50, :]), plan["_x"][:, :50, :]):
            raise RuntimeError(f"{path}: x read back differs from what was written")


def _expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.glob("predictions_*.nc") if PRED_RE.search(q.name))
        else:
            out.append(p)
    return out


def _public(plan: dict) -> dict:
    return {k: v for k, v in plan.items() if not k.startswith("_")}


def process(path: Path, *, do_apply: bool, store: str | None, tp_store: str | None, stores=None) -> dict:
    try:
        plan = prepare(path, store, tp_store, stores)
    except CheckFailed as exc:
        return {"file": str(path), "action": "abort", "why": str(exc)}
    if plan["action"] == "skip":
        return plan
    if do_apply:
        apply(plan)
        plan["action"] = "written"
    else:
        plan["action"] = "dry-run (would write)"
    return _public(plan)


def _worker(args):
    path, do_apply, store, tp_store = args
    try:
        return process(Path(path), do_apply=do_apply, store=store, tp_store=tp_store)
    except Exception as exc:  # noqa: BLE001 - report every failure per file, keep the others going
        return {"file": str(path), "action": "error", "why": f"{type(exc).__name__}: {exc}"}


def _line(r: dict) -> str:
    name = "/".join(Path(r["file"]).parts[-2:])
    if r["action"] in ("skip", "abort", "error"):
        return f"{r['action'].upper():8s} {name}: {r.get('why')}"
    p = r["pairing"]
    units = ",".join(f"{s}:{'ok' if u['ok'] else 'BAD'}" for s, u in r["units"].items())
    return (f"{r['action'].upper():8s} {name}: store row {r['row']} ({Path(r['store']).name}"
            f"{', tp ' + Path(r['tp_store']).name if r['tp_store'] else ''}), x ({r['n_members']}, "
            f"{r['n_points_lres']}, {len(r['states'])}) {r['x_megabytes']} MB; msl pairing vs {p['reference']}: "
            f"diag min {p['diag_min']:.3f} mean {p['diag_mean']:.3f}, off-diag max {p['offdiag_max']:.3f}, "
            f"ok={p['ok']}; units vs {r['units_reference']}: {units}; precip clipped {r['precip_clipped']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Append the native input x (and lat_lres/lon_lres) to ENS-member prediction files from "
                    "their zarr member store (rule 41). Dry run unless --apply.")
    ap.add_argument("paths", nargs="+", help="predictions_<date>_step<lead>.nc files or directories of them")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run, nothing is written)")
    ap.add_argument("--store", help="member store (default: the file's member_store attribute)")
    ap.add_argument("--tp-store", help="tp/cp member store (default: the file's member_tp_store attribute)")
    ap.add_argument("--workers", type=int, default=1, help="files processed in parallel (default 1)")
    ap.add_argument("--report", help="write the per-file results to this JSON file")
    args = ap.parse_args(argv)

    files = _expand(args.paths)
    jobs = [(str(f), args.apply, args.store, args.tp_store) for f in files]
    print(f"backfill_native_x: {len(files)} file(s), {'APPLY' if args.apply else 'DRY RUN'}", flush=True)
    results = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for r in pool.imap(_worker, jobs):
                results.append(r)
                print(_line(r), flush=True)
    else:
        for job in jobs:
            r = _worker(job)
            results.append(r)
            print(_line(r), flush=True)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    print(f"backfill_native_x: done {counts}", flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=1, default=str) + "\n")
    return 0 if not any(r["action"] in ("abort", "error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
