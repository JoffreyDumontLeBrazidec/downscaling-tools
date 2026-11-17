#!/usr/bin/env python3
"""QM baseline, phase 1 (Atos CPU): training pairs -> per-month sample matrices on the O1280 grid.

For calendar month M, N_BASE base dates of that month are drawn (seed SEED + M) among the base dates whose chunks exist in
BOTH global v4 stores (input eefh O320, target enfh O1280; the stores GL512 was trained on), at the six leads 12..72 h.
For each (base date, lead): input 10u 10v 2t msl tp interpolated to O1280 (qm_lib.idw_*), target read as is; the four
mapped variables in working units (qm_lib.to_work) are written as float16 rows of
    OUT/samples/m<MM>_<var>_<in|t>.f16      shape (N_BASE * 6, 6,599,680), row = sample, NaN rows = not written
plus OUT/samples/m<MM>_index.json (base dates, leads, row states). Re-running skips rows already marked done.

    python3 qm_extract.py --probe --gin STORE --gout STORE            # print both stores' layout, write nothing
    python3 qm_extract.py --gin STORE --gout STORE --out DIR --month 9 [--n-base 40]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qm_lib import LEADS, VARS, idw_apply, idw_weights, to_work  # noqa: E402

SEED = 20261117
PHYS = ("10u", "10v", "2t", "msl", "tp")


def layout(g):
    """-> dict(data, base_dates (datetime64[D] array), steps (list of hours), variables, n_points) of a v4 trajectories store."""
    data = g["data"]
    attrs = dict(g.attrs)
    vars_ = list(attrs["variables"])
    if data.ndim != 5:
        raise SystemExit(f"expected the trajectories layout (base_dates, variables, ensembles, steps, values); got {data.shape}")
    nb, nv, ne, ns, npt = data.shape
    assert nv == len(vars_), (nv, len(vars_))
    bd = None
    for key in ("base_dates", "dates"):
        if key in g and g[key].shape[0] == nb:
            bd = np.asarray(g[key][:]).astype("datetime64[D]")
            break
    if bd is None:
        for key in ("base_dates", "start_date"):
            if key in attrs:
                if isinstance(attrs[key], list) and len(attrs[key]) == nb:
                    bd = np.asarray(attrs[key], "datetime64[D]")
                    break
        if bd is None and "start_date" in attrs:
            bd = np.datetime64(str(attrs["start_date"])[:10], "D") + np.arange(nb)
    if bd is None:
        raise SystemExit(f"base dates not found: arrays {sorted(g.keys())}, attrs {sorted(attrs)}")
    steps = None
    for key in ("steps", "lead_times", "leadtimes"):
        if key in g and g[key].shape[0] == ns:
            steps = [int(np.asarray(v) / (np.timedelta64(1, "h") if np.issubdtype(np.asarray(v).dtype, np.timedelta64) else 1))
                     for v in np.asarray(g[key][:])]
            break
        if key in attrs and len(attrs[key]) == ns:
            steps = [int(str(v).rstrip("h")) if not isinstance(v, (int, float)) else int(v) for v in attrs[key]]
            break
    if steps is None:
        raise SystemExit(f"steps not found: arrays {sorted(g.keys())}, attrs {sorted(attrs)}")
    return dict(data=data, base_dates=bd, steps=steps, variables=vars_, n_points=npt)


def chunk_exists(store_path, data, b, s):
    """True when the chunk (base b, all variables, ensemble 0, step s, all points) is on disk (zarr v2 or v3 directory
    store). Missing (base date, step) slots of the v4 stores have no chunk file."""
    d = Path(store_path) / "data"
    if (d / ".zarray").exists():
        zr = json.loads((d / ".zarray").read_text())
        ch, sep, prefix = zr["chunks"], zr.get("dimension_separator", "."), ""
    else:
        zr = json.loads((d / "zarr.json").read_text())
        ch = zr["chunk_grid"]["configuration"]["chunk_shape"]
        enc = zr.get("chunk_key_encoding", {"name": "default"})
        sep = enc.get("configuration", {}).get("separator", "/" if enc["name"] == "default" else ".")
        prefix = "c" + sep if enc["name"] == "default" else ""
    key = prefix + sep.join(str(i // c) for i, c in zip((b, 0, 0, s, 0), ch))
    return (d / key).exists()


def probe(path):
    import zarr
    g = zarr.open_group(path, mode="r")
    print("STORE", path)
    print("  arrays", {k: (g[k].shape, str(g[k].dtype)) for k in g.array_keys()})
    print("  attrs", sorted(g.attrs.keys()))
    print("  data chunks", g["data"].chunks, "dtype", g["data"].dtype)
    L = layout(g)
    print("  base dates", L["base_dates"][0], "..", L["base_dates"][-1], len(L["base_dates"]), "steps", L["steps"])
    print("  variables", L["variables"])
    have = [b for b in range(0, len(L["base_dates"]), 7) if chunk_exists(path, L["data"], b, L["steps"].index(24))]
    print(f"  chunk check every 7th base date at +24 h: {len(have)} of {len(range(0, len(L['base_dates']), 7))} exist")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gin", required=True)
    ap.add_argument("--gout", required=True)
    ap.add_argument("--out")
    ap.add_argument("--month", type=int)
    ap.add_argument("--n-base", type=int, default=40)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        probe(a.gin)
        probe(a.gout)
        return
    import zarr
    gi, go = zarr.open_group(a.gin, mode="r"), zarr.open_group(a.gout, mode="r")
    Li, Lo = layout(gi), layout(go)
    assert np.array_equal(Li["base_dates"], Lo["base_dates"]), "the two stores have different base-date axes"
    for L in (Li, Lo):
        for v in PHYS:
            assert v in L["variables"], f"{v} not in store variables"
    si = [Li["steps"].index(h) for h in LEADS]
    so = [Lo["steps"].index(h) for h in LEADS]
    out = Path(a.out) / "samples"
    out.mkdir(parents=True, exist_ok=True)
    wfile = Path(a.out) / "idw_o320_o1280_k4.npz"
    hlat, hlon = np.asarray(go["latitudes"][:]), np.asarray(go["longitudes"][:])
    if not wfile.exists():
        llat, llon = np.asarray(gi["latitudes"][:]), np.asarray(gi["longitudes"][:])
        t0 = time.time()
        idx, w = idw_weights(llat, llon, hlat, hlon)
        part = f"{wfile}.{os.getpid()}.part.npz"
        np.savez(part, idx=idx, w=w, n_src=llat.size, n_dst=hlat.size)
        os.replace(part, wfile)
        print(f"IDW weights {hlat.size} x 4 in {time.time() - t0:.0f}s", flush=True)
    W = np.load(wfile)
    idx, w = W["idx"], W["w"]
    m = a.month
    months = Li["base_dates"].astype("datetime64[M]").astype(int) % 12 + 1
    cand = [b for b in np.flatnonzero(months == m)
            if chunk_exists(a.gin, Li["data"], int(b), si[0]) and chunk_exists(a.gout, Lo["data"], int(b), so[0])]
    rng = np.random.default_rng(SEED + m)
    pick = sorted(rng.choice(cand, size=min(a.n_base, len(cand)), replace=False).tolist())
    n_rows = len(pick) * len(LEADS)
    npt = hlat.size
    mm = {}
    for v in VARS:
        for side in ("in", "t"):
            f = out / f"m{m:02d}_{v}_{side}.f16"
            mode = "r+" if f.exists() else "w+"
            mm[(v, side)] = np.lib.format.open_memmap(f, mode=mode, dtype=np.float16, shape=(n_rows, npt)) if mode == "w+" \
                else np.load(f, mmap_mode="r+")
    ixf = out / f"m{m:02d}_index.json"
    index = json.loads(ixf.read_text()) if ixf.exists() else dict(
        month=m, seed=SEED + m, n_candidates=len(cand), base_dates=[str(Li["base_dates"][b]) for b in pick],
        base_index=[int(b) for b in pick], leads=list(LEADS), done=[False] * n_rows, gin=a.gin, gout=a.gout)
    assert index["base_index"] == [int(b) for b in pick], "sample draw changed since the first run"
    vi = [Li["variables"].index(v) for v in PHYS]
    vo = [Lo["variables"].index(v) for v in PHYS]
    t0 = time.time()
    for r, (b, li) in enumerate([(b, li) for b in pick for li in range(len(LEADS))]):
        if index["done"][r]:
            continue
        xin = np.asarray(Li["data"][b, :, 0, si[li], :], np.float32)[vi]           # (5, O320)
        xt = np.asarray(Lo["data"][b, :, 0, so[li], :], np.float32)[vo]            # (5, O1280)
        if not (np.isfinite(xin).all() and np.isfinite(xt).all()):
            print(f"  skip row {r}: non-finite (base {Li['base_dates'][b]}, +{LEADS[li]} h)", flush=True)
            continue
        pin = dict(zip(PHYS, idw_apply(xin, idx, w)))
        pt = dict(zip(PHYS, xt))
        for v in VARS:
            mm[(v, "in")][r] = to_work(v, pin).astype(np.float16)
            mm[(v, "t")][r] = to_work(v, pt).astype(np.float16)
        index["done"][r] = True
        if r % 12 == 11:
            for x in mm.values():
                x.flush()
            ixf.write_text(json.dumps(index))
            print(f"  m{m:02d} rows {sum(index['done'])}/{n_rows}, {time.time() - t0:.0f}s", flush=True)
    for x in mm.values():
        x.flush()
    ixf.write_text(json.dumps(index))
    print(f"EXTRACT_DONE m{m:02d}: {sum(index['done'])} of {n_rows} rows, {len(cand)} candidate base dates, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
