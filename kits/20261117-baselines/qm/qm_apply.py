#!/usr/bin/env python3
"""QM baseline, phase 3 (Atos CPU): the QM row of an evaluation set (XSET-1 or S25), one start date at a time.

For each lead and input member k (operational eefo pf k of the set's input store, A/stores/in/<date>.zarr): 10u 10v 2t msl
tp interpolated to the whole O1280 grid (the phase-1 weights), mapped with the tables of the window of the start date's
month (2t, msl, tp directly; 10 m wind speed mapped and 10u, 10v scaled by mapped / input speed). Member k of the row =
the mapping of input member k (one realisation per input member, rule 36; QM adds no noise).
Writes, exactly like a model row of the XSET kit:
    A/rows/QM/predictions_<date>_step<LLL>.nc   y_pred on the evaluation points; y, native x, imerg_tp12, why, coordinates
                                                copied from the ENS9 template file of the same date and lead
    A/hist/QM_<date>_step<LLL>.npz              global tail histograms of the 10 mapped members (xset_common)

    python3 qm_apply.py --work A --qm QMDIR --gout STORE --date 20240804 [--date ...] [--label QM]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "20261116-extremes-set"))
from qm_lib import VARS, from_work, idw_apply, qm_map, to_work  # noqa: E402
from xset_common import LEADS, SAVE_VARS, tail_histograms, write_row_file  # noqa: E402

PHYS = ("10u", "10v", "2t", "msl", "tp")


def read_template(path):
    import netCDF4
    with netCDF4.Dataset(str(path)) as d:
        t = {k: np.asarray(d[k][:]) for k in ("lat_hres", "lon_hres", "point_index", "y", "x", "lat_lres", "lon_lres",
                                             "lres_index")}
        t["states"] = [str(s) for s in d["weather_state"][:]]
        t["extra"] = {k: np.asarray(d[k][:]) for k in ("why", "imerg_tp12") if k in d.variables}
    t["y"], t["x"] = t["y"][0], t["x"][0]
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="set work folder A (holds stores/in, rows/ENS9, hist)")
    ap.add_argument("--qm", required=True, help="QM folder (idw_o320_o1280_k4.npz, tables/)")
    ap.add_argument("--gout", required=True, help="global O1280 store (latitudes for the histograms)")
    ap.add_argument("--date", action="append", required=True)
    ap.add_argument("--label", default="QM")
    a = ap.parse_args()
    import zarr
    A, Q = Path(a.work), Path(a.qm)
    W = np.load(Q / "idw_o320_o1280_k4.npz")
    idx, w = W["idx"], W["w"]
    hlat = np.asarray(zarr.open_group(a.gout, mode="r")["latitudes"][:])
    assert hlat.size == idx.shape[0]
    rows, hist = A / "rows" / a.label, A / "hist"
    rows.mkdir(parents=True, exist_ok=True)
    hist.mkdir(parents=True, exist_ok=True)
    tables, tmonth = {}, None
    for d in a.date:
        t0 = time.time()
        if all((rows / f"predictions_{d}_step{L:03d}.nc").exists() for L in LEADS):
            print("have", d, flush=True)
            continue
        mo = int(d[4:6])
        if mo != tmonth:
            tables = {(v, s): np.load(Q / "tables" / f"w{mo:02d}_{v}_{s}.npy", mmap_mode="r") for v in VARS for s in ("in", "t")}
            tables = {k: np.asarray(v, np.float32) for k, v in tables.items()}   # ~7 GB float32 in memory
            tmonth = mo
        gi = zarr.open_group(str(A / "stores" / "in" / f"{d}.zarr"), mode="r")
        ivars = list(gi.attrs["variables"])
        assert list(gi.attrs["leads"]) == list(LEADS)
        rep = {}
        for li, lead in enumerate(LEADS):
            name = f"predictions_{d}_step{lead:03d}.nc"
            if (rows / name).exists():
                continue
            tpl = read_template(A / "rows" / "ENS9" / name)
            assert tpl["states"] == list(SAVE_VARS), tpl["states"]
            pidx = tpl["point_index"].astype(np.int64)
            xin = np.asarray(gi["data"][li], np.float32)                     # (70, members, O320)
            nm = xin.shape[1]
            full = {v: np.empty((nm, hlat.size), np.float32) for v in PHYS}
            for k in range(nm):
                pin = {v: idw_apply(xin[ivars.index(v), k], idx, w) for v in PHYS}
                wk = {v: to_work(v, pin) for v in VARS}
                mapped = {v: qm_map(v, wk[v], tables[(v, "in")],
                                    tables[(v, "t")]) for v in VARS}
                ws_in = wk["ws"]
                scale = np.where(ws_in > 0.01, from_work("ws", mapped["ws"]) / np.maximum(ws_in, 0.01), 1.0)
                full["10u"][k] = pin["10u"] * scale
                full["10v"][k] = pin["10v"] * scale
                full["2t"][k] = from_work("2t", mapped["2t"])
                full["msl"][k] = from_work("msl", mapped["msl"])
                full["tp"][k] = from_work("tp", mapped["tp"])
            assert all(np.isfinite(full[v]).all() for v in PHYS), f"non-finite QM {d} +{lead}"
            h = tail_histograms(hlat, full["tp"], full["10u"], full["10v"])
            np.savez(hist / f"{a.label}_{d}_step{lead:03d}.npz", **h)
            y_pred = np.stack([full[v][:, pidx] for v in SAVE_VARS], axis=-1)
            tmp = rows / (name + ".part")
            write_row_file(tmp, lat=tpl["lat_hres"], lon=tpl["lon_hres"], point_index=pidx, states=list(SAVE_VARS), y=tpl["y"],
                           y_pred=y_pred, x=tpl["x"], lat_lres=tpl["lat_lres"], lon_lres=tpl["lon_lres"],
                           lres_index=tpl["lres_index"], extra=tpl["extra"],
                           attrs=dict(model=f"{a.label}: per-point quantile mapping of the interpolated 36 km input onto the "
                                            f"9 km training climatology (window centred on month {mo:02d})",
                                      competitor_row=f"{a.label}: classical statistical downscaling baseline",
                                      truth="y = od enfo cf", date=d, lead=lead, member_rule="y_pred member k = QM(input member k)",
                                      qm_dir=str(Q), xset="see work folder", tp_units="m per 12 h"))
            tmp.rename(rows / name)
            rep[lead] = dict(tp_max_mm=float(full["tp"].max() * 1000), ws_max=float(np.hypot(full["10u"], full["10v"]).max()),
                             msl_min_hpa=float(full["msl"].min() / 100), t2_mean=float(full["2t"].mean()))
            print(f"  {a.label} {d} +{lead} h: {json.dumps(rep[lead])}", flush=True)
        (A / "check").mkdir(exist_ok=True)
        (A / "check" / f"{a.label}_{d}.json").write_text(json.dumps(rep, indent=1))
        print(f"QM {d} in {time.time() - t0:.0f}s", flush=True)
    print("QM_APPLY_DONE", flush=True)


if __name__ == "__main__":
    main()
