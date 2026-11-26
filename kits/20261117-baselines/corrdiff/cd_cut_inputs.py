#!/usr/bin/env python3
"""CorrDiff baseline, inference inputs (Atos CPU): per start date of an evaluation set (XSET-1 or S25), the 10 input
members on CorrDiff's coarse box grid and the evaluation points, in one file to copy to Leonardo.

Input: A/stores/in/<date>.zarr (6 leads, 70 variables, 10 eefo pf members, global O320) of the XSET kit. The O320
points inside the E2 box + 1.5 deg (the E2-v4 in_lres crop) are regridded to the 216 x 432 grid with the same rule as
the training export (cd_export: inverse squared chord distance, 4 nearest), so the network sees the same kind of field.
Evaluation points: lat / lon of the ENS9 row file of the date (all leads share them; checked).

    python3 cd_cut_inputs.py --work A --out DIR --date 20240804 [--date ...]
Writes DIR/cdin_<date>.npz: xc (6, 10, 70, 216, 432) float32, variables, leads, elat, elon (evaluation points).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "20261116-extremes-set"))
import cd_grid as G  # noqa: E402
from xset_common import LEADS  # noqa: E402

HALO = 1.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", action="append", required=True)
    a = ap.parse_args()
    import netCDF4
    import zarr
    A, O = Path(a.work), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    w = None
    for d in a.date:
        if (O / f"cdin_{d}.npz").exists():
            print("have", d, flush=True)
            continue
        g = zarr.open_group(str(A / "stores" / "in" / f"{d}.zarr"), mode="r")
        assert list(g.attrs["leads"]) == list(LEADS)
        lat, lon = np.asarray(g["latitudes"][:], np.float64), G.wrap180(g["longitudes"][:])
        s, n, wb, e = G.BOX
        sel = np.flatnonzero((lat >= s - HALO) & (lat <= n + HALO) & (lon >= wb - HALO) & (lon <= e + HALO))
        if w is None:
            w = G.idw_to_grid(lat[sel], lon[sel], G.NYC, G.NXC)
            w_sel = sel
        assert np.array_equal(sel, w_sel), "the O320 grid differs between dates"
        idx, ww = w
        xc = np.empty((len(LEADS), 10, len(g.attrs["variables"]), G.NYC, G.NXC), np.float32)
        for li in range(len(LEADS)):
            x = np.asarray(g["data"][li], np.float32)[..., sel]                   # (70, 10, n_sel)
            xc[li] = np.moveaxis((x[..., idx] * ww).sum(-1), 1, 0).reshape(10, -1, G.NYC, G.NXC)
        pts = []
        for L in LEADS:
            with netCDF4.Dataset(str(A / "rows" / "ENS9" / f"predictions_{d}_step{L:03d}.nc")) as t:
                pts.append((np.asarray(t["lat_hres"][:]), np.asarray(t["lon_hres"][:])))
        assert all(np.array_equal(p[0], pts[0][0]) and np.array_equal(p[1], pts[0][1]) for p in pts)
        assert np.isfinite(xc).all(), d
        np.savez(O / f"cdin_{d}.npz", xc=xc, variables=np.array(g.attrs["variables"]), leads=np.array(LEADS),
                 elat=pts[0][0], elon=pts[0][1], date=d, n_o320_box=sel.size)
        print(f"CUT {d}: {sel.size} O320 box points, {pts[0][0].size} evaluation points", flush=True)
    print("CUT_DONE", flush=True)


if __name__ == "__main__":
    main()
