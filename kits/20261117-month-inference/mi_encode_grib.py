#!/usr/bin/env python3
"""MI-S25 (17 Nov 2026): encode one model row and start date (the packed npz files of mi_predict_global.py, copied from
Jupiter) as GRIB on the global O1280 grid, for quaver source="marsfile" (route B; rule 30: a local file, never FDB).

Keys are cloned from real operational enfo O1280 messages (mi_template.sh: od enfo pf 1, 20250901 00 UTC, step 12, the 10
surface outputs), then set: class rd, stream enfo, type pf, expver (--expver, a label), number = input member, dataDate,
dataTime 0, 16 bits per value. Values are written in the template's point order (mapped from the model grid file by nearest
point; the mapping must be exact). tp and cp (12 h windows ending at each lead, clipped at 0) are written as accumulations
from step 0, so quaver's de-accumulation returns the windows; this needs the leads to tile 12..max by 12 (checked).
Usage: mi_encode_grib.py --rowdir ROWDIR --date 20250901 --template tmpl_o1280.grib --out OUT.grib [--expver xxxx] [--vars ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ACCUM = ("tp", "cp")


def unit_vectors(lat, lon):
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rowdir", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expver", default="xxxx")
    ap.add_argument("--vars", default="10u,10v,2d,2t,msl,tp")
    a = ap.parse_args()
    import eccodes
    from scipy.spatial import cKDTree
    t0 = time.time()
    rowdir, d = Path(a.rowdir), a.date
    want = a.vars.split(",")
    files = sorted((rowdir / d).glob(f"pred_{d}_step*.npz"))
    leads = [int(f.stem.split("step")[1]) for f in files]
    assert leads == list(range(12, 12 * len(leads) + 1, 12)), f"leads must tile 12..max by 12: {leads}"
    tm = {}
    with open(a.template, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            tm.setdefault(eccodes.codes_get(h, "shortName"), h)
    assert all(v in tm for v in want), (sorted(tm), want)
    h0 = tm[want[0]]
    glat, glon = eccodes.codes_get_array(h0, "latitudes"), eccodes.codes_get_array(h0, "longitudes")
    g = np.load(rowdir / "grid_o1280.npz")
    dist, idx = cKDTree(unit_vectors(g["lat"], g["lon"])).query(unit_vectors(glat, glon), k=1)
    assert dist.max() < 1e-6 and np.unique(idx).size == idx.size == g["lat"].size, (dist.max(), idx.size, g["lat"].size)
    print(f"grid: {glat.size} template points, model points mapped exactly (max chord {dist.max():.1e}), "
          f"out of order {int(np.sum(idx != np.arange(idx.size)))}", flush=True)
    acc, n = {}, 0
    with open(a.out + ".part", "wb") as fo:
        for f, L in zip(files, leads):
            z = np.load(f)
            states = [str(s) for s in z["states"]]
            q, lo, hi = z["q"], z["lo"], z["hi"]
            members = [int(m) for m in z["members"]]
            for v in want:
                j = states.index(v)
                for k, num in enumerate(members):
                    vals = lo[k, j] + q[k, j].astype(np.float64) / 65535.0 * (hi[k, j] - lo[k, j])
                    vals = vals[idx]
                    h = eccodes.codes_clone(tm[v])
                    for key, val in (("class", "rd"), ("stream", "enfo"), ("type", "pf"), ("expver", a.expver), ("number", num),
                                     ("dataDate", int(d)), ("dataTime", 0)):
                        eccodes.codes_set(h, key, val)
                    if v in ACCUM:
                        acc[(v, num)] = acc.get((v, num), 0.0) + np.clip(vals, 0.0, None)
                        vals = acc[(v, num)]
                        eccodes.codes_set(h, "stepRange", f"0-{L}")
                    else:
                        eccodes.codes_set(h, "step", L)
                    eccodes.codes_set(h, "bitsPerValue", 16)
                    eccodes.codes_set_values(h, vals)
                    eccodes.codes_write(h, fo)
                    eccodes.codes_release(h)
                    n += 1
            attrs = json.loads(str(z["attrs"]))
    Path(a.out + ".part").rename(a.out)
    print(f"ENCODE_DONE {a.out} fields={n} leads={leads[0]}..{leads[-1]} members={len(members)} vars={want} "
          f"bytes={Path(a.out).stat().st_size} seconds={time.time() - t0:.0f} model={attrs.get('model')} step={attrs.get('ckpt_step')}",
          flush=True)


if __name__ == "__main__":
    main()
