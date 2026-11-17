#!/usr/bin/env python3
"""Reference row files of one start date of the extremes set XSET-1 (16 Nov 2026): the 9 km ensemble row ENS9 and, inside
the same files, everything the other rows share (docs note s4.2). Atos, from the MARS GRIB files (never FDB).

Per lead L of 12..72 h it writes
  <out>/rows/ENS9/predictions_<date>_step<LLL>.nc   on the evaluation points of points_<date>.npz:
        y_pred  enfo pf 1-10 (the 9 km ensemble, scored as a row),   y  enfo cf (the single 9 km field),
        x       eefo pf 1-10 on the native O320 points (lat_lres, lon_lres; the input row, rule 41),
        imerg_tp12  IMERG Final 12 h rain ending at the valid time [mm] (NaN outside the store's period),
        why     the reason flags of xset_points.py, point_index / lres_index the positions in the global stores
  <out>/hist/ENS9_<date>_step<LLL>.npz, <out>/hist/INPUT_<date>_step<LLL>.npz   global tail histograms (xset_common)
Variables 10u 10v 2t msl tp; tp in metres per 12 h = acc(step) - acc(step - 12), floored at 0 (store recipe).
The model predictor on the GPU cluster copies y, x, imerg_tp12 and the coordinates from these files and adds its draws.

    python3 xset_rows_date.py --gout STORE --points DIR --grib DIR --stores DIR --out DIR --date 20240804 [--date ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from xset_build_in import read_grib
from xset_common import LEADS, SAVE_VARS, tail_histograms, wrap180, write_row_file

N_PF = 10
IMERG_ZARR = "/ec/ai/project/ai-ml/datasets/aifs-imerg-final-v07b-nasa-o1280-1998-2025-6h-v1-observations.zarr"


def field(fields, num, key, lead):
    if key == "tp":
        prev = fields[(num, key, lead - 12)] if lead > 12 else 0.0
        return np.clip(fields[(num, key, lead)] - prev, 0.0, None)
    return fields[(num, key, lead)]


class Imerg:
    def __init__(self, path, n_points):
        import zarr
        self.g = zarr.open(str(path), mode="r")
        names = list(self.g.attrs.get("variables", ["tp"]))
        self.var = names.index("tp")
        t = np.asarray(self.g["dates"][:]).astype("datetime64[s]").astype(dt.datetime)
        self.pos = {x: i for i, x in enumerate(t)}
        assert self.g["data"].shape[-1] == n_points, (self.g["data"].shape, n_points)

    def tp12(self, valid, idx):
        """mm over (valid - 12 h, valid] at the points idx, or None when a 6 h step is missing."""
        steps = [valid - dt.timedelta(hours=6), valid]
        if any(s not in self.pos for s in steps):
            return None
        return sum(np.asarray(self.g["data"][self.pos[s], self.var, 0, :], np.float32)[idx] for s in steps) * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gout", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--grib", required=True)
    ap.add_argument("--stores", required=True, help="folder holding in/<date>.zarr (xset_build_in.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imerg", default=IMERG_ZARR)
    ap.add_argument("--date", action="append", required=True)
    a = ap.parse_args()
    import zarr
    go = zarr.open_group(a.gout, mode="r")
    hlat, hlon = np.asarray(go["latitudes"][:]), np.asarray(go["longitudes"][:])
    out, grib = Path(a.out), Path(a.grib)
    (out / "rows" / "ENS9").mkdir(parents=True, exist_ok=True)
    (out / "hist").mkdir(parents=True, exist_ok=True)
    (out / "check").mkdir(parents=True, exist_ok=True)
    imerg = Imerg(a.imerg, hlat.size) if a.imerg and Path(a.imerg).exists() else None
    states = list(SAVE_VARS)
    for d in a.date:
        done = out / "check" / f"rows_{d}.done"
        if done.exists():
            print("have", d, flush=True)
            continue
        t0 = time.time()
        pts = np.load(Path(a.points) / f"points_{d}.npz")
        idx, lidx = pts["idx"].astype(np.int64), pts["lres_idx"].astype(np.int64)
        gi = zarr.open_group(str(Path(a.stores) / "in" / f"{d}.zarr"), mode="r")
        ivars = list(gi.attrs["variables"])
        llat, llon = np.asarray(gi["latitudes"][:], np.float64), np.asarray(gi["longitudes"][:], np.float64)
        fields = {}
        for f in ("tr_sfc_cf", "tr_sfc_pf"):
            p = grib / f"{f}_{d}.grib"
            assert p.exists() and Path(str(p) + ".ok").exists(), f"missing or incomplete {p}"
            fields.update(read_grib(p, set(states), hlat, hlon))
        rep = {}
        for li, lead in enumerate(LEADS):
            valid = dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=lead)
            pf = np.stack([np.stack([field(fields, k, v, lead) for v in states], axis=-1).astype(np.float32)
                           for k in range(1, N_PF + 1)])
            cf = np.stack([field(fields, 0, v, lead) for v in states], axis=-1).astype(np.float32)
            assert np.isfinite(pf).all() and np.isfinite(cf).all(), f"non-finite truth {d} +{lead}"
            h = tail_histograms(hlat, pf[:, :, states.index("tp")], pf[:, :, states.index("10u")], pf[:, :, states.index("10v")])
            np.savez(out / "hist" / f"ENS9_{d}_step{lead:03d}.npz", **h)
            xin = np.asarray(gi["data"][li])                                   # (70, members, O320 points)
            assert list(gi.attrs["leads"])[li] == lead
            h = tail_histograms(llat, xin[ivars.index("tp")], xin[ivars.index("10u")], xin[ivars.index("10v")])
            np.savez(out / "hist" / f"INPUT_{d}_step{lead:03d}.npz", **h)
            x = np.stack([xin[ivars.index(v)][:, lidx] for v in states], axis=-1).astype(np.float32)   # (members, native, states)
            im = imerg.tp12(valid, idx) if imerg is not None else None
            extra = {"why": pts["why"].astype(np.uint8),
                     "imerg_tp12": (im if im is not None else np.full(idx.size, np.nan)).astype(np.float32)}
            name = f"predictions_{d}_step{lead:03d}.nc"
            tmp = out / "rows" / "ENS9" / (name + ".part")
            write_row_file(tmp, lat=hlat[idx], lon=wrap180(hlon[idx]), point_index=idx, states=states, y=cf[idx],
                           y_pred=pf[:, idx, :], x=x, lat_lres=llat[lidx], lon_lres=wrap180(llon[lidx]), lres_index=lidx,
                           extra=extra,
                           attrs=dict(model="ENS9 (od enfo 0001 pf 1-10, 00 UTC)",
                                      competitor_row="ENS9: od enfo 0001 pf 1-10, the 9 km operational ensemble", truth="y = od enfo cf", date=d, lead=lead,
                                      input="x = od eefo 0001 pf 1-10 on native O320 points (rule 41)", xset="XSET-1",
                                      member_rule="y_pred member k = enfo pf k; x member k = eefo pf k; the two are NOT paired",
                                      imerg="IMERG Final V07B 12 h sum ending at valid time, mm" if im is not None else "none",
                                      points=str(pts["counts"]), tp_units="m per 12 h"))
            tmp.rename(out / "rows" / "ENS9" / name)
            tp_mm = pf[:, :, states.index("tp")] * 1000.0
            rep[lead] = dict(points=int(idx.size), native_points=int(lidx.size), tp_max_mm=float(tp_mm.max()),
                             ws_max=float(np.hypot(pf[:, :, 0], pf[:, :, 1]).max()), msl_min_hpa=float(pf[:, :, 3].min() / 100.0),
                             t2_mean=float(pf[:, :, 2].mean()), imerg_finite=None if im is None else float(np.isfinite(im).mean()),
                             imerg_max_mm=None if im is None else float(np.nanmax(im)))
            print(f"  {d} +{lead} h: {json.dumps(rep[lead])}", flush=True)
        (out / "check" / f"rows_{d}.json").write_text(json.dumps(rep, indent=1))
        done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        del fields
        print(f"ROWS {d} in {time.time() - t0:.0f}s", flush=True)
    print("ROWS_DONE", flush=True)


if __name__ == "__main__":
    main()
