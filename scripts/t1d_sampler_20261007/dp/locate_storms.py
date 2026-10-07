"""CPU preflight of the T1d diagnostic boxes: for each (date, lead) bundle, find the storm centre the
trajectory tool will find (argmin of the TRUTH msl inside the Idalia window, as interp's
detect_min_center does), and check that the 500 km disc stays 1 deg inside the cut graph and that
the centre is not on the window edge. No GPU, no model.

  python -m scripts.t1d_sampler_20261007.dp.locate_storms --bundle-dir <root>     # all four bundles
  python -m scripts.t1d_sampler_20261007.dp.locate_storms --print-window 20230826 024
Exit code 1 when a check fails. T1D_WINDOW="lat0,lat1,lon0,lon1" overrides the table for --print-window.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import BOX_RADIUS_KM, IDALIA_WINDOWS, box_checks, disc_extent  # type: ignore
else:
    from .common import BOX_RADIUS_KM, IDALIA_WINDOWS, box_checks, disc_extent


def window_for(date, step):
    env = os.environ.get("T1D_WINDOW")
    if env:
        return tuple(float(x) for x in env.split(","))
    return IDALIA_WINDOWS[(str(date), "%03d" % int(step))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-window", nargs=2, metavar=("DATE", "STEP"))
    ap.add_argument("--bundle-dir")
    ap.add_argument("--member", default="01")
    a = ap.parse_args(argv)
    if a.print_window:
        print(",".join("%g" % x for x in window_for(*a.print_window)))
        return
    import xarray as xr
    from manual_inference.input_data_construction.bundle import extract_target_from_bundle
    ok = True
    for (date, step), _ in sorted(IDALIA_WINDOWS.items()):
        win = window_for(date, step)
        hits = sorted(glob.glob(str(Path(a.bundle_dir) / f"*date{date}*mem{a.member}*step{step}h*input_bundle.nc")))
        if not hits:
            print(f"FAIL {date}/{step}: no bundle in {a.bundle_dir}")
            ok = False
            continue
        ds = xr.open_dataset(hits[0])
        lat = np.asarray(ds["lat_hres"].values, dtype=np.float64)
        lon = np.asarray(ds["lon_hres"].values, dtype=np.float64) % 360.0
        ds.close()
        msl = extract_target_from_bundle(hits[0], ["msl"])[0][:, 0]
        sel = (lat >= win[0]) & (lat <= win[1]) & (lon >= win[2]) & (lon <= win[3])
        i = int(np.argmin(np.where(sel & np.isfinite(msl), msl, np.inf)))
        clat, clon = lat[i], lon[i]
        e = disc_extent(clat, clon, BOX_RADIUS_KM)
        print(f"{date}/{step}: window {win} -> centre {clat:.2f}N {clon:.2f}E, truth msl min {msl[i] / 100:.1f} hPa, "
              f"disc {e[0]:.2f}..{e[1]:.2f}N {e[2]:.2f}..{e[3]:.2f}E  ({Path(hits[0]).name})")
        for good, msg in box_checks(clat, clon, BOX_RADIUS_KM, win):
            print(("  PASS " if good else "  FAIL ") + msg)
            ok &= bool(good)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
