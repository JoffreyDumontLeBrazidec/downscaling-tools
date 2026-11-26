#!/usr/bin/env python3
"""CorrDiff baseline, data export (Leonardo CPU): the E2-v4 box stores -> one zarr on the regular box grids (cd_grid).

Sources (anemoi stores, layout data[row, variable, member, point], E2-v4 note s7): the target out_hres_o1280.zarr
(70 variables, 814,810 points), the input in_lres_o320_halo1p5.zarr (70 variables, 55,871 points incl. the 1.5 deg
margin) and E2's forcings in_hres_o1280_forcings.zarr (11 variables; the ones constant over rows are the invariants).
Rows: the 17,280 rows of the synthetic axis; the store attribute fake_hindcasts maps each row to (reference date,
hindcast date, step). Split by hindcast year, as RW50k and GL512: train <= 2021, valid 2022, test 2023; the counts
must be 15,870 / 864 / 546 (the parse is checked by them).

Writes OUT.zarr:
  output    [row, 10, 864, 1728] float32  GL512's 10 outputs (10u 10v 2d 2t msl skt sp tcw cp tp), store units
  input     [row, 70, 216, 432]  float32  all 70 input variables (the training config selects)
  invariant [n_inv, 864, 1728]   float32
  attrs: output_variables, input_variables, invariant_variables, grid (cd_grid), rows meta in rows.csv beside it
and OUT.stats.json (mean, std per channel over the training rows; hrrr_mini's layout).

    python3 cd_export.py probe --out-store S --in-store S --forcings S
    python3 cd_export.py init  --out-store S --in-store S --forcings S --dest OUT.zarr
    python3 cd_export.py fill  --out-store S --in-store S --dest OUT.zarr --part I --nparts N
    python3 cd_export.py stats --dest OUT.zarr [--nrows 1200]
    python3 cd_export.py verify --out-store S --in-store S --dest OUT.zarr
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cd_grid as G  # noqa: E402

OUT10 = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "cp", "tp"]
COUNTS = {"train": 15870, "valid": 864, "test": 546}


def open_store(p):
    import zarr
    return zarr.open_group(str(p), mode="r")


def variables(g):
    v = g.attrs.get("variables")
    if v is None:
        v = list(g.attrs["name_to_index"].keys())
    return list(v)


def _to_dt(x):
    if isinstance(x, (int, np.integer)):
        raise ValueError(x)
    s = str(x).replace("Z", "")
    for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d%H", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s[:19], f)
        except ValueError:
            pass
    raise ValueError(f"cannot read date {x!r}")


def _hours(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    s = str(x).strip().lower()
    if s.endswith("h"):
        s = s[:-1]
    if ":" in s:
        h, m = s.split(":")[:2]
        return int(h) + int(m) // 60
    return int(float(s))


def parse_entry(v):
    """fake_hindcasts value of one row -> (reference, hindcast, step hours). Accepts a dict with keys naming
    reference / hindcast / step, or a sequence (reference, hindcast, step) in the order of the E2 note."""
    if isinstance(v, dict):
        kr = [k for k in v if "ref" in k.lower()]
        kh = [k for k in v if "hindcast" in k.lower() or k.lower() in ("date", "base_date", "init")]
        ks = [k for k in v if "step" in k.lower() or "lead" in k.lower()]
        if len(kh) != 1 or len(ks) != 1:
            raise ValueError(f"fake_hindcasts entry keys not understood: {sorted(v)}")
        return (_to_dt(v[kr[0]]) if kr else None), _to_dt(v[kh[0]]), _hours(v[ks[0]])
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return _to_dt(v[0]), _to_dt(v[1]), _hours(v[2])
    raise ValueError(f"fake_hindcasts entry not understood: {v!r}")


def row_table(g):
    """[(row, synthetic date, reference, hindcast, step, valid, split)] for every row of the store."""
    dates = np.asarray(g["dates"][:])
    fh = g.attrs["fake_hindcasts"]
    out = []
    for r, d in enumerate(dates):
        key = str(np.datetime_as_string(d, unit="s")) if np.issubdtype(dates.dtype, np.datetime64) else str(d)
        cand = [key, key.replace("T", " "), key[:16], key[:13]]
        hit = next((c for c in cand if c in fh), None)
        if hit is None:
            raise SystemExit(f"row {r} date {key}: no fake_hindcasts entry (keys look like {list(fh)[:2]})")
        ref, hc, step = parse_entry(fh[hit])
        y = hc.year
        split = "train" if y <= 2021 else ("valid" if y == 2022 else ("test" if y == 2023 else "none"))
        out.append((r, key, ref, hc, step, hc + dt.timedelta(hours=step), split))
    n = {s: sum(1 for x in out if x[6] == s) for s in COUNTS}
    steps = sorted({x[4] for x in out})
    if os.environ.get("CD_TEST_ROWS") == "1":   # synthetic self-test only
        return out
    if n != COUNTS or steps != [12, 24, 36, 48, 60, 72]:
        raise SystemExit(f"row split check failed: counts {n} (want {COUNTS}), steps {steps}")
    return out


def mk(g, name, shape, chunks, dtype):
    try:
        return g.create_array(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=np.nan)
    except AttributeError:
        return g.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=np.nan)


def weights(gout, gin):
    wo = G.idw_to_grid(gout["latitudes"][:], gout["longitudes"][:], G.NY, G.NX)
    wi = G.idw_to_grid(gin["latitudes"][:], gin["longitudes"][:], G.NYC, G.NXC)
    return wo, wi


def regrid(field, w, ny, nx):
    idx, ww = w
    return (np.asarray(field, np.float32)[..., idx] * ww).sum(-1).reshape(field.shape[:-1] + (ny, nx))


def cmd_probe(a):
    for name, p in (("out", a.out_store), ("in", a.in_store), ("forcings", a.forcings)):
        g = open_store(p)
        lat, lon = np.asarray(g["latitudes"][:]), G.wrap180(g["longitudes"][:])
        print(f"{name}: data {g['data'].shape} {g['data'].dtype} chunks {g['data'].chunks}; lat {lat.min():.2f}..{lat.max():.2f} "
              f"lon {lon.min():.2f}..{lon.max():.2f}; variables {variables(g)}; attrs {sorted(g.attrs.keys())}")
    g = open_store(a.out_store)
    fh = g.attrs.get("fake_hindcasts")
    print("fake_hindcasts:", type(fh).__name__, len(fh) if fh is not None else None, list(fh.items())[:3] if isinstance(fh, dict) else None)
    rows = row_table(g)
    print("rows OK:", {s: sum(1 for x in rows if x[6] == s) for s in COUNTS}, "first", rows[0], "last", rows[-1])


def cmd_init(a):
    import zarr
    gout, gin, gf = open_store(a.out_store), open_store(a.in_store), open_store(a.forcings)
    rows = row_table(gout)
    rows_in = row_table(gin)
    assert [(x[3], x[4]) for x in rows] == [(x[3], x[4]) for x in rows_in], "input and target rows differ"
    vo, vi, vf = variables(gout), variables(gin), variables(gf)
    assert all(v in vo for v in OUT10), vo
    n = len(rows)
    # invariants: forcing variables equal on two distant rows
    f0, f1 = np.asarray(gf["data"][0][:, 0, :], np.float32), np.asarray(gf["data"][n // 2][:, 0, :], np.float32)
    inv = [v for i, v in enumerate(vf) if np.array_equal(f0[i], f1[i])]
    print("invariants:", inv, "time-varying (not used):", [v for v in vf if v not in inv], flush=True)
    wo, wi = weights(gout, gin)
    dest = Path(a.dest)
    np.savez(dest.with_suffix(".weights.npz"), o_idx=wo[0], o_w=wo[1], i_idx=wi[0], i_w=wi[1])
    root = zarr.open_group(str(dest), mode="w")
    mk(root, "output", (n, len(OUT10), G.NY, G.NX), (1, len(OUT10), G.NY, G.NX), "float32")
    mk(root, "input", (n, len(vi), G.NYC, G.NXC), (1, len(vi), G.NYC, G.NXC), "float32")
    a_inv = mk(root, "invariant", (len(inv), G.NY, G.NX), (len(inv), G.NY, G.NX), "float32")
    a_inv[:] = regrid(f0[[vf.index(v) for v in inv]], wo, G.NY, G.NX)
    lat, lon = G.axes(G.NY, G.NX)
    root.attrs.update(dict(output_variables=OUT10, input_variables=vi, invariant_variables=inv, box=list(G.BOX),
                           ny=G.NY, nx=G.NX, factor=G.FACTOR, lat0=float(lat[0]), lon0=float(lon[0]),
                           sources=dict(out=str(a.out_store), inp=str(a.in_store), forcings=str(a.forcings)),
                           regrid="inverse squared chord distance, 4 nearest points", created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    with open(dest.with_suffix(".rows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "synthetic", "reference", "hindcast", "step", "valid", "split"])
        for r in rows:
            w.writerow([r[0], r[1], r[2].isoformat() if r[2] else "", r[3].isoformat(), r[4], r[5].isoformat(), r[6]])
    print(f"INIT_DONE {dest}: {n} rows", flush=True)


def cmd_fill(a):
    import zarr
    gout, gin = open_store(a.out_store), open_store(a.in_store)
    root = zarr.open_group(str(a.dest), mode="r+")
    W = np.load(Path(a.dest).with_suffix(".weights.npz"))
    wo, wi = (W["o_idx"], W["o_w"]), (W["i_idx"], W["i_w"])
    vo = variables(gout)
    oi = [vo.index(v) for v in OUT10]
    n = root["output"].shape[0]
    mine = list(range(a.part, n, a.nparts))
    t0, done = time.time(), 0
    for r in mine:
        if not a.overwrite and np.isfinite(root["output"][r, 0, G.NY // 2, G.NX // 2]):
            continue
        yo = np.asarray(gout["data"][r][:, 0, :], np.float32)[oi]
        xi = np.asarray(gin["data"][r][:, 0, :], np.float32)
        root["input"][r] = regrid(xi, wi, G.NYC, G.NXC)   # input first: a finite output marks a complete row
        root["output"][r] = regrid(yo, wo, G.NY, G.NX)
        done += 1
        if done % 50 == 0:
            print(f"part {a.part}: {done} rows, {(time.time() - t0) / done:.2f} s/row", flush=True)
    print(f"FILL_DONE part {a.part}/{a.nparts}: {done} rows written of {len(mine)}", flush=True)


def cmd_stats(a):
    import zarr
    root = zarr.open_group(str(a.dest), mode="r")
    rows = list(csv.DictReader(open(Path(a.dest).with_suffix(".rows.csv"))))
    tr = [int(r["row"]) for r in rows if r["split"] == "train"]
    rng = np.random.default_rng(20261117)
    pick = np.sort(rng.choice(tr, size=min(a.nrows, len(tr)), replace=False))
    out = {}
    for grp, arr, names in (("output", "output", root.attrs["output_variables"]), ("input", "input", root.attrs["input_variables"])):
        s1 = np.zeros(len(names)); s2 = np.zeros(len(names)); cnt = 0
        for r in pick:
            x = np.asarray(root[arr][r], np.float64).reshape(len(names), -1)
            if not np.isfinite(x).all():
                raise SystemExit(f"row {r} of {arr} not filled or non-finite")
            s1 += x.sum(1); s2 += (x * x).sum(1); cnt += x.shape[1]
        m = s1 / cnt
        sd = np.sqrt(np.maximum(s2 / cnt - m * m, 0.0))
        out[grp] = {v: {"mean": float(m[i]), "std": float(max(sd[i], 1e-12))} for i, v in enumerate(names)}
    inv = np.asarray(root["invariant"][:], np.float64)
    out["invariant"] = {v: {"mean": float(inv[i].mean()), "std": float(max(inv[i].std(), 1e-12))}
                        for i, v in enumerate(root.attrs["invariant_variables"])}
    out["_meta"] = dict(rows=int(len(pick)), split="train", seed=20261117)
    Path(a.dest).with_suffix(".stats.json").write_text(json.dumps(out, indent=1))
    print("STATS_DONE", len(pick), "rows", flush=True)


def cmd_verify(a):
    """Every row filled; 6 random rows recomputed from the stores bit for bit; output tp >= 0."""
    import zarr
    gout, gin = open_store(a.out_store), open_store(a.in_store)
    root = zarr.open_group(str(a.dest), mode="r")
    W = np.load(Path(a.dest).with_suffix(".weights.npz"))
    vo = variables(gout)
    oi = [vo.index(v) for v in OUT10]
    n = root["output"].shape[0]
    bad = [r for r in range(n) if not (np.isfinite(root["output"][r, :, G.NY // 2, G.NX // 2]).all()
                                       and np.isfinite(root["input"][r, :, G.NYC // 2, G.NXC // 2]).all())]
    if bad:
        raise SystemExit(f"{len(bad)} rows not filled, first {bad[:10]}")
    for r in np.random.default_rng(1).choice(n, 6, replace=False):
        yo = regrid(np.asarray(gout["data"][r][:, 0, :], np.float32)[oi], (W["o_idx"], W["o_w"]), G.NY, G.NX)
        xi = regrid(np.asarray(gin["data"][r][:, 0, :], np.float32), (W["i_idx"], W["i_w"]), G.NYC, G.NXC)
        assert np.array_equal(yo, root["output"][r]) and np.array_equal(xi, root["input"][r]), f"row {r} differs"
        assert np.nanmin(root["output"][r, OUT10.index("tp")]) >= 0.0
    print(f"VERIFY_OK {n} rows", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["probe", "init", "fill", "stats", "verify"])
    ap.add_argument("--out-store")
    ap.add_argument("--in-store")
    ap.add_argument("--forcings")
    ap.add_argument("--dest")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--nrows", type=int, default=1200)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    {"probe": cmd_probe, "init": cmd_init, "fill": cmd_fill, "stats": cmd_stats, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
