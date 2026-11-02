#!/usr/bin/env python3
"""o2560 box read: provenance of one cluster's training inputs (review fix 5). sha256 of the residual statistics file and
of the warm-start checkpoint, and of the statistics arrays (mean, stdev, minimum, maximum) and variable list of the two
box stores. Run once on Atos (inputs of the crop chain A) and once on Jupiter (inputs of the full-box chain B).
Usage: python provenance_side.py <side label> <out.json> <residual_statistics.npy> <warm_start.ckpt> <store dir>"""
import hashlib
import json
import sys

import numpy as np
import zarr

side, out, resid, warm, store = sys.argv[1:6]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


res = {"side": side, "residual_statistics": {"path": resid, "sha256": sha_file(resid)},
       "warm_start": {"path": warm, "sha256": sha_file(warm)}, "stores": {}}
for name in ("in_lres_o1280.zarr", "hres_o2560.zarr"):
    z = zarr.open(f"{store}/{name}", "r")
    st = {"variables": list(z.attrs["variables"])}
    for k in ("mean", "stdev", "minimum", "maximum"):
        a = np.ascontiguousarray(np.asarray(z[k][:], dtype=np.float64))
        st[f"{k}_sha256"] = hashlib.sha256(a.tobytes()).hexdigest()
    res["stores"][name] = st
json.dump(res, open(out, "w"), indent=1)
print("PROVENANCE_DONE", side, out)
