#!/usr/bin/env python3
"""Input member store of one start date of the extremes set XSET-1 (16 Nov 2026), from the MARS GRIB files of
xset_retrieve_date.sh (never FDB, rule 30).

  <out>/in/<date>.zarr   data (6 leads, 70 variables, 10 members, 421120 points) float32: operational eefo pf 1-10 on O320,
                         the 70 variables and the point order of the global in_lres store GL512 was trained on
                         (--gin, read only for its variable list, coordinates and statistics); leads 12..72 h.

Processing = the v4 store recipe, same functions as build_gl_member_stores.py (4 Nov 2026), which reproduced the stored
control bit for bit on 60 rows of 2023: instantaneous fields at the step; tp and cp = acc(step) - acc(step - 12), floored at 0.
No stored control exists for 2024-2025, so the check here is statistical: every variable's mean and standard deviation over
points and members against the store's own statistics (check/<date>.json; a variable whose mean is more than 3 store standard
deviations away, or whose standard deviation is outside 0.2-5 times the store's, stops the build before anything is marked done).

Usage (Atos, module load python3 ecmwf-toolbox):
  xset_build_in.py --gin GLOBAL_IN_STORE --grib GRIBDIR --out OUTDIR --date 20240804 [--date ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LEADS = (12, 24, 36, 48, 60, 72)
N_PF = 10
ACCUM = ("tp", "cp")


def unit_vectors(lat, lon):
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def read_grib(path, want, tlat, tlon):
    """{(number, key, endStep): values at the target points, in target order} for messages whose key is in `want`."""
    import eccodes
    from scipy.spatial import cKDTree
    out, idx, n_grib = {}, None, None
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                sn = eccodes.codes_get(h, "shortName")
                lt = eccodes.codes_get(h, "typeOfLevel")
                key = f"{sn}_{eccodes.codes_get(h, 'level')}" if lt == "isobaricInhPa" else sn
                if key not in want:
                    continue
                if idx is None:
                    glat, glon = eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes")
                    d, idx = cKDTree(unit_vectors(glat, glon)).query(unit_vectors(tlat, tlon), k=1)
                    assert d.max() < 1e-6, f"{path}: target points not on the GRIB grid (max chord {d.max():.2e})"
                    n_grib = glat.size
                    print(f"  grid {Path(path).name}: {glat.size} GRIB points -> {idx.size} targets, max chord {d.max():.1e}, "
                          f"points out of store order {int(np.sum(idx != np.arange(idx.size))) if idx.size == glat.size else -1}",
                          flush=True)
                vals = eccodes.codes_get_values(h)
                assert vals.size == n_grib, f"{path}: grid size changed within the file"
                num = int(eccodes.codes_get(h, "number")) if eccodes.codes_get(h, "dataType") == "pf" else 0
                out[(num, key, int(eccodes.codes_get(h, "endStep")))] = vals[idx]
            finally:
                eccodes.codes_release(h)
    return out


def field(fields, num, key, lead):
    if key in ACCUM:
        if lead == 12:
            return np.clip(fields[(num, key, 12)], 0.0, None)
        return np.clip(fields[(num, key, lead)] - fields[(num, key, lead - 12)], 0.0, None)
    return fields[(num, key, lead)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gin", required=True)
    ap.add_argument("--grib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", action="append", required=True)
    a = ap.parse_args()
    import zarr
    s = zarr.open_group(a.gin, mode="r")
    vars_ = list(s.attrs["variables"])
    tlat, tlon = np.asarray(s["latitudes"][:]), np.asarray(s["longitudes"][:])
    stats = {k: np.asarray(s[k][:], np.float64) for k in ("mean", "stdev") if k in s and s[k].shape == (len(vars_),)}
    assert len(stats) == 2, f"store statistics mean / stdev not found in {a.gin}: {sorted(s.keys())}"
    out, grib = Path(a.out), Path(a.grib)
    (out / "in").mkdir(parents=True, exist_ok=True)
    (out / "check").mkdir(parents=True, exist_ok=True)
    for d in a.date:
        done = out / "check" / f"in_{d}.done"
        if done.exists():
            print("have", d, flush=True)
            continue
        t0 = time.time()
        fields = {}
        for f in ("in_pl_pf", "in_sfc_pf"):
            p = grib / f"{f}_{d}.grib"
            assert p.exists() and Path(str(p) + ".ok").exists(), f"missing or incomplete {p}"
            fields.update(read_grib(p, set(vars_), tlat, tlon))
        tmp = out / "in" / f"{d}.zarr.part"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)                      # an unfinished build of this same date, created by this script
        g = zarr.open_group(str(tmp), mode="w-", zarr_format=2)
        data = g.create_array("data", shape=(len(LEADS), len(vars_), N_PF, tlat.size), chunks=(1, len(vars_), 1, tlat.size),
                              dtype="float32", fill_value=np.nan)
        g.create_array("latitudes", data=tlat)
        g.create_array("longitudes", data=tlon)
        rep, bad = {}, []
        for li, lead in enumerate(LEADS):
            x = np.empty((len(vars_), N_PF, tlat.size), np.float32)
            for k in range(1, N_PF + 1):
                x[:, k - 1, :] = np.stack([field(fields, k, v, lead) for v in vars_]).astype(np.float32)
            assert np.isfinite(x).all(), f"non-finite input {d} +{lead}"
            m = x.astype(np.float64).mean(axis=(1, 2))
            sd = x.astype(np.float64).std(axis=(1, 2))
            z = (m - stats["mean"]) / stats["stdev"]
            r = sd / stats["stdev"]
            # member identity: members must differ from each other on every prognostic variable
            same = [v for i, v in enumerate(vars_) if np.array_equal(x[i, 0], x[i, 1])]
            rep[lead] = dict(z_mean={v: round(float(q), 3) for v, q in zip(vars_, z)},
                             sd_ratio={v: round(float(q), 3) for v, q in zip(vars_, r)}, identical_members_1_2=same)
            for i, v in enumerate(vars_):
                if abs(z[i]) > 3.0 or not (0.2 <= r[i] <= 5.0):
                    bad.append((lead, v, round(float(z[i]), 2), round(float(r[i]), 2)))
            bad += [(lead, v, "members 1 and 2 identical") for v in same]
            for k in range(N_PF):
                data[li, :, k, :] = x[:, k, :]
            print(f"  {d} +{lead} h: max |z of the mean| {np.abs(z).max():.2f} ({vars_[int(np.abs(z).argmax())]}), "
                  f"sd ratio {r.min():.2f}..{r.max():.2f}, tp max {x[vars_.index('tp')].max() * 1000:.1f} mm", flush=True)
        g.attrs.update(dict(variables=vars_, members=list(range(1, N_PF + 1)), member_type="pf", leads=list(LEADS), date=d,
                            row_map=[dict(date=d, lead=int(L)) for L in LEADS], source_store=str(a.gin),
                            stream="od eefo 0001 00 UTC, O320 (pressure levels grid=O320)",
                            processing="instantaneous at step; tp,cp = acc(step)-acc(step-12), clip >= 0",
                            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        (out / "check" / f"in_{d}.json").write_text(json.dumps(dict(date=d, bad=bad, leads=rep), indent=1))
        if bad:
            sys.exit(f"CHECK FAILED {d}: {bad[:8]} (store left as {tmp})")
        tmp.rename(out / "in" / f"{d}.zarr")
        done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        del fields
        print(f"BUILT in {d} in {time.time() - t0:.0f}s", flush=True)
    print("BUILD_IN_DONE", flush=True)


if __name__ == "__main__":
    main()
