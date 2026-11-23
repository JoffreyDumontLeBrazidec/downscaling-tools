#!/usr/bin/env python3
"""Check cd_dataset.time_fields against the forcings store GL512 reads (E2's in_hres_o1280_forcings.zarr, anemoi layout
data[row, variable, member, point]). Read-only, light (one store row = 36 MB per checked row; login node with nice).

Rows: the export rows (OUT.rows.csv) matched to the forcings rows by (hindcast date, step) of each store's own
fake_hindcasts (not by index). For each checked row and each of the 5 time inputs: max |store - formula| on the native
O1280 points for several candidate times (valid = hindcast + step, the synthetic date of the 1 h axis, reference + step,
hindcast), then on the 864 x 1728 box grid (store field regridded with the export's own weights vs the formula on the
grid), which adds the regrid error.

    python3 cd_check_time_inputs.py --forcings F.zarr --data OUT.zarr [--rows auto | --rows 12,345,...]
"""
import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cd_export as E  # noqa: E402
import cd_grid as G  # noqa: E402
from cd_dataset import TIME_INPUTS, time_fields  # noqa: E402


def pick_rows(rows):
    """Two rows at 00 UTC and two at 12 UTC in different seasons (+ the first two valid rows)."""
    want = [("train", 1, 0), ("train", 7, 0), ("train", 4, 12), ("train", 10, 12), ("valid", None, 12), ("valid", None, 0)]
    out = []
    for split, month, hour in want:
        for r in rows:
            v = dt.datetime.fromisoformat(r["valid"])
            if r["split"] == split and v.hour == hour and (month is None or v.month == month) and r not in out:
                out.append(r)
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forcings", required=True)
    ap.add_argument("--data", required=True, help="the export zarr (OUT.rows.csv, OUT.weights.npz beside it)")
    ap.add_argument("--rows", default="auto")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(Path(a.data).with_suffix(".rows.csv"))))
    sel = pick_rows(rows) if a.rows == "auto" else [rows[int(i)] for i in a.rows.split(",")]
    gf = E.open_store(a.forcings)
    vf = E.variables(gf)
    fi = [vf.index(v) for v in TIME_INPUTS]
    ftab = {(x[3], x[4]): x for x in E.row_table(gf)}
    flat, flon = np.asarray(gf["latitudes"][:], np.float64), np.asarray(gf["longitudes"][:], np.float64)
    gout = E.open_store(E.open_store(a.data).attrs["sources"]["out"])
    same_grid = np.array_equal(np.asarray(gout["latitudes"][:]), flat) and np.array_equal(np.asarray(gout["longitudes"][:]), flon)
    W = np.load(Path(a.data).with_suffix(".weights.npz"))
    glat, glon = G.axes(G.NY, G.NX)
    glat2, glon2 = np.repeat(glat[:, None], G.NX, 1), np.repeat(glon[None, :], G.NY, 0)
    print(f"forcings {a.forcings}: {gf['data'].shape}, variables {vf}; lon range {flon.min():.2f}..{flon.max():.2f}; "
          f"forcing grid == target store grid: {same_grid}")
    worst = {v: 0.0 for v in TIME_INPUTS}
    for r in sel:
        hc, step = dt.datetime.fromisoformat(r["hindcast"]), int(r["step"])
        fr = ftab[(hc, step)]
        valid = dt.datetime.fromisoformat(r["valid"])
        assert fr[5] == valid, (fr, r)
        f = np.asarray(gf["data"][fr[0]][fi, 0, :], np.float64)
        cands = {"valid": valid, "synthetic": dt.datetime.fromisoformat(r["synthetic"]),
                 "reference+step": dt.datetime.fromisoformat(r["reference"]) + dt.timedelta(hours=step), "hindcast": hc}
        print(f"\nexport row {r['row']} ({r['split']}) = forcings row {fr[0]}: hindcast {hc:%Y-%m-%d %H}Z step {step} "
              f"valid {valid:%Y-%m-%d %H}Z synthetic {r['synthetic']} reference {r['reference']}")
        for name, t in cands.items():
            d = np.abs(time_fields(t, flat, flon).astype(np.float64) - f).max(axis=1)
            print(f"  native, formula at {name:15s} {t:%Y-%m-%d %H}Z: " + "  ".join(f"{v} {x:.2e}" for v, x in zip(TIME_INPUTS, d)))
        if same_grid:
            g = E.regrid(f.astype(np.float32), (W["o_idx"], W["o_w"]), G.NY, G.NX).astype(np.float64)
            d = np.abs(time_fields(valid, glat2, glon2).astype(np.float64) - g).reshape(len(TIME_INPUTS), -1).max(axis=1)
            print(f"  box grid (store regridded vs formula on grid, valid time):  " +
                  "  ".join(f"{v} {x:.2e}" for v, x in zip(TIME_INPUTS, d)))
        dn = np.abs(time_fields(valid, flat, flon).astype(np.float64) - f).max(axis=1)
        for v, x in zip(TIME_INPUTS, dn):
            worst[v] = max(worst[v], float(x))
    print("\nWORST native max |store - formula at valid time| over the checked rows: " +
          "  ".join(f"{v} {x:.2e}" for v, x in worst.items()))


if __name__ == "__main__":
    main()
