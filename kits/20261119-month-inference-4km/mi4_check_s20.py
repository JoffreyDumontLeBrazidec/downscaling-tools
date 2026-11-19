#!/usr/bin/env python3
"""MI4-S26 processing check (19 Nov 2026): the mi4_build_in.py processing (Reader + lead_arrays) applied to the s20 member store's
own GRIB files (20261109-o2560-tc/grib/in_{sfc,plg,tp}_pf_<d>.grib, enfo pf at grid=O1280, the same MARS recipe as MI4) at the
s20 box points must reproduce the s20 rows of emo_in_lres_o1280_pf.zarr (37 variables) and emo_tp_o1280_pf.zarr (tp, cp) bit
for bit. Also checks that the s20 box variable order equals the global meta store order.
Usage: mi4_check_s20.py --meta META --date 20250808 --lead 24 [--lead 48]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mi4_build_in import Reader, lead_arrays

T = Path("/ec/res4/scratch/ecm5702/agent-work/20261109-o2560-tc")
ap = argparse.ArgumentParser(); ap.add_argument("--meta", required=True); ap.add_argument("--date", required=True)
ap.add_argument("--lead", type=int, action="append", required=True); a = ap.parse_args()
import zarr
gi = zarr.open_group(str(T / "stores_2025/emo_in_lres_o1280_pf.zarr"), mode="r")
gt = zarr.open_group(str(T / "stores_2025/emo_tp_o1280_pf.zarr"), mode="r")
meta = zarr.open_group(a.meta, mode="r")
bv, mv = list(gi.attrs["variables"]), list(meta.attrs["variables"])
print("variable order s20 box == global meta:", bv == mv, flush=True)
pos = {(r["hdate"], int(r["lead"])): i for i, r in enumerate(gi.attrs["row_map"])}
files = [T / f"grib/{n}_{a.date}.grib" for n in ("in_sfc_pf", "in_plg_pf", "in_tp_pf")]
rd = Reader(files, np.asarray(gi["latitudes"][:]), np.asarray(gi["longitudes"][:]))
res = dict(order_equal=bv == mv, date=a.date, leads={})
ok = bv == mv
for L in a.lead:
    x, t = lead_arrays(rd, mv, L)
    i = pos[(a.date, L)]
    ref_x = np.asarray(gi["data"][i])            # (37, 10, n)
    ref_t = np.asarray(gt["data"][i])            # (2, 10, n)
    if bv != mv:
        ref_x = ref_x[[bv.index(v) for v in mv]]
    dx = [int((x[j] != ref_x[j]).sum()) for j in range(37)]
    dt = [int((t[j] != ref_t[j]).sum()) for j in range(2)]
    res["leads"][L] = dict(unequal_points_37=dict(zip(mv, dx)), unequal_points_tp_cp=dt,
                           max_abs_37=float(np.abs(x.astype(np.float64) - ref_x).max()),
                           max_abs_tp=float(np.abs(t.astype(np.float64) - ref_t).max()))
    ok &= sum(dx) == 0 and sum(dt) == 0
    print(f"CHECK_S20 {a.date} +{L} h: unequal values 37 vars {sum(dx)} (max |diff| {res['leads'][L]['max_abs_37']:.3e}), "
          f"tp/cp {dt} (max {res['leads'][L]['max_abs_tp']:.3e}) over {x[0].size} member-points per variable", flush=True)
print(json.dumps(res)[:2000])
print("CHECK_S20_PASS" if ok else "CHECK_S20_FAIL", flush=True)
sys.exit(0 if ok else 1)
