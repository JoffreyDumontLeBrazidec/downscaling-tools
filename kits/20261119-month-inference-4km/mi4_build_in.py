#!/usr/bin/env python3
"""MI4-S26 (19 Nov 2026): FULL-GLOBE input member store of one start date for the 4 km lane (o1280 -> o2560), from the MARS GRIB
files of mi4_retrieve_date.sh (never FDB, rule 30). Layout follows the 9 km month kit (kits/20261117-month-inference/mi_build_in.py),
processing follows the s20 member store builder (20261109-o2560-tc/kit/build_member_store_o2560_y2025.py), minus the box cut:

  <out>/in/<date>.zarr  data (39 leads, 37 variables, 10 members, 6599680 points) float32: od enfo pf 1-10 at native O1280,
                        the 37 in_lres variables of the global 4 km model G150e2 (Jupiter datasets/o1280_o2560_global/
                        in_lres_o1280.zarr, read through --meta for variable order, coordinates and statistics), same point order
                        tp    (39, 2, 10, 6599680) float32: tp, cp 6 h windows clip(acc(L) - acc(L-6), 0) in float64 (as s20), for
                        the input-ensemble baseline; not a model input
                        leads 12..240 h every 6 h; instantaneous fields at the step
Check per lead: every variable's mean and standard deviation over points and members against the training store's statistics
(|z of the mean| > 3 or sd ratio outside 0.2-5 stops the build), finite values, members 1 and 2 not identical.
The 37-variable and tp processing is shared with mi4_check_s20.py, which reproduces the s20 store rows bit for bit.

Usage (Atos, module load python3 ecmwf-toolbox; Slurm job, ~60 GB):
  mi4_build_in.py --meta META_ZARR --grib GRIBDIR --out OUTDIR --date 20260901 [--date ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

LEADS = tuple(range(12, 241, 6))
N_PF = 10
TP_VARS = ("tp", "cp")
GROUPS = [LEADS[i:i + 3] for i in range(0, len(LEADS), 3)]           # 13 groups = the 13 pl_gNN files
CHUNK_PTS = 412480                                                    # = training store chunks (16 grid blocks)


def unit_vectors(lat, lon):
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


class Reader:
    """Index of GRIB messages (member, key, endStep) -> (file, offset, length) by a headers-only pass; values at target points."""

    def __init__(self, files, tlat, tlon):
        import eccodes
        self.ec, self.tlat, self.tlon, self.idx, self.npts, self.loc = eccodes, tlat, tlon, None, None, {}
        for p in files:
            with open(p, "rb") as f:
                while True:
                    h = eccodes.codes_new_from_file(f, eccodes.CODES_PRODUCT_GRIB, headers_only=True)
                    if h is None:
                        break
                    try:
                        sn = eccodes.codes_get(h, "shortName")
                        lt = eccodes.codes_get(h, "typeOfLevel")
                        key = f"{sn}_{eccodes.codes_get(h, 'level')}" if lt == "isobaricInhPa" else sn
                        num = int(eccodes.codes_get(h, "number")) if eccodes.codes_get(h, "dataType") == "pf" else 0
                        k = (num, key, int(eccodes.codes_get(h, "endStep")))
                        assert k not in self.loc, f"duplicate message {k} in {p}"
                        self.loc[k] = (str(p), int(eccodes.codes_get(h, "offset")), int(eccodes.codes_get(h, "totalLength")))
                    finally:
                        eccodes.codes_release(h)

    def get(self, num, key, step, dtype=np.float32):
        p, off, ln = self.loc[(num, key, step)]
        with open(p, "rb") as f:
            f.seek(off)
            buf = f.read(ln)
        h = self.ec.codes_new_from_message(buf)
        try:
            assert self.ec.codes_get(h, "gridType") == "reduced_gg", (p, key, step)
            if self.idx is None:
                from scipy.spatial import cKDTree
                glat, glon = self.ec.codes_get_array(h, "latitudes"), self.ec.codes_get_array(h, "longitudes")
                d, self.idx = cKDTree(unit_vectors(glat, glon)).query(unit_vectors(self.tlat, self.tlon), k=1)
                assert d.max() < 1e-6, f"target points not on the GRIB grid (max chord {d.max():.2e})"
                self.npts = glat.size
                print(f"  grid: {glat.size} GRIB points -> {self.idx.size} targets, max chord {d.max():.1e}, points out of "
                      f"store order {int(np.sum(self.idx != np.arange(self.idx.size))) if self.idx.size == glat.size else -1}",
                      flush=True)
            v = self.ec.codes_get_values(h)
            assert v.size == self.npts, "grid size changed"
            return v[self.idx].astype(dtype)
        finally:
            self.ec.codes_release(h)


def lead_arrays(rd, vars_, lead, members=range(1, N_PF + 1)):
    """(37, M, n) float32 inputs and (2, M, n) float32 tp/cp 6 h windows of one lead: the s20 recipe."""
    x = np.stack([np.stack([rd.get(k, v, lead) for v in vars_]) for k in members], axis=1)
    t = np.stack([np.stack([np.clip(rd.get(k, v, lead, np.float64) - rd.get(k, v, lead - 6, np.float64), 0.0, None)
                            .astype(np.float32) for v in TP_VARS]) for k in members], axis=1)
    return x, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--grib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", action="append", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import numcodecs
    import zarr
    numcodecs.blosc.set_nthreads(a.threads)
    comp = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)   # = training store compressor
    s = zarr.open_group(a.meta, mode="r")
    vars_ = list(s.attrs["variables"])
    assert len(vars_) == 37 and "tp" not in vars_, vars_
    tlat, tlon = np.asarray(s["latitudes"][:]), np.asarray(s["longitudes"][:])
    assert tlat.size == 6599680, tlat.size
    mean, sdev = np.asarray(s["mean"][:], np.float64), np.asarray(s["stdev"][:], np.float64)
    assert mean.shape == (37,) and sdev.shape == (37,), (mean.shape, sdev.shape)
    out, grib = Path(a.out), Path(a.grib)
    (out / "in").mkdir(parents=True, exist_ok=True)
    (out / "check").mkdir(parents=True, exist_ok=True)
    for d in a.date:
        done = out / "check" / f"in_{d}.done"
        if done.exists():
            print("have", d, flush=True)
            continue
        t0 = time.time()
        files = [grib / f"pl_g{g:02d}_{d}.grib" for g in range(1, 14)] + [grib / f"{n}_{d}.grib" for n in ("sfq", "sfx", "tp")]
        for p in files:
            assert p.exists() and Path(str(p) + ".ok").exists(), f"missing or incomplete {p}"
        rd = Reader(files, tlat, tlon)
        want = {(k, v, L) for k in range(1, N_PF + 1) for L in LEADS for v in vars_} | \
               {(k, v, L) for k in range(1, N_PF + 1) for L in LEADS + (6,) for v in TP_VARS}
        miss = sorted(want - set(rd.loc))
        assert not miss, f"{d}: {len(miss)} messages missing, e.g. {miss[:5]}"
        print(f"  {d}: indexed {len(rd.loc)} messages in {time.time() - t0:.0f}s", flush=True)
        tmp = out / "in" / f"{d}.zarr.part"
        if tmp.exists():
            shutil.rmtree(tmp)                      # an unfinished build of this same date, created by this script
        g = zarr.open_group(str(tmp), mode="w-", zarr_format=2)
        n = tlat.size
        data = g.create_array("data", shape=(len(LEADS), 37, N_PF, n), chunks=(1, 37, 1, CHUNK_PTS), dtype="float32",
                              fill_value=np.nan, compressors=comp)
        tpa = g.create_array("tp", shape=(len(LEADS), 2, N_PF, n), chunks=(1, 2, 1, CHUNK_PTS), dtype="float32",
                             fill_value=np.nan, compressors=comp)
        g.create_array("latitudes", data=tlat)
        g.create_array("longitudes", data=tlon)
        rep, bad = {}, []
        for lead in LEADS:
            tl = time.time()
            x, t = lead_arrays(rd, vars_, lead)
            assert np.isfinite(x).all() and np.isfinite(t).all(), f"non-finite input {d} +{lead}"
            m = np.array([x[i].astype(np.float64).mean() for i in range(37)])     # per variable: no 20 GB float64 copy
            sd = np.array([x[i].astype(np.float64).std() for i in range(37)])
            z, r = (m - mean) / sdev, sd / sdev
            same = [v for i, v in enumerate(vars_) if np.array_equal(x[i, 0], x[i, 1])]
            rep[lead] = dict(z_mean={v: round(float(q), 3) for v, q in zip(vars_, z)},
                             sd_ratio={v: round(float(q), 3) for v, q in zip(vars_, r)}, identical_members_1_2=same,
                             tp_max_mm=round(float(t[0].max()) * 1000, 2))
            for i, v in enumerate(vars_):
                if abs(z[i]) > 3.0 or not (0.2 <= r[i] <= 5.0):
                    bad.append((lead, v, round(float(z[i]), 2), round(float(r[i]), 2)))
            bad += [(lead, v, "members 1 and 2 identical") for v in same]
            li = LEADS.index(lead)
            tw = time.time()
            for k in range(N_PF):
                data[li, :, k, :] = x[:, k, :]
                tpa[li, :, k, :] = t[:, k, :]
            print(f"  {d} +{lead} h: decode {tw - tl:.0f}s write {time.time() - tw:.0f}s; max |z of the mean| {np.abs(z).max():.2f} "
                  f"({vars_[int(np.abs(z).argmax())]}), sd ratio {r.min():.2f}..{r.max():.2f}, tp max {t[0].max() * 1000:.1f} mm",
                  flush=True)
            del x, t
        g.attrs.update(dict(variables=vars_, tp_variables=list(TP_VARS), members=list(range(1, N_PF + 1)), member_type="pf",
                            leads=list(LEADS), date=d, row_map=[dict(date=d, lead=int(L)) for L in LEADS], set="MI4-S26",
                            meta_store=str(a.meta), meta_source="Jupiter datasets/o1280_o2560_global/in_lres_o1280.zarr",
                            stream="od enfo 0001 pf 1-10 00 UTC, native O1280 (all requests grid=O1280; pl t/u/v/w/z spectral -> MIR)",
                            processing="instantaneous at step; tp,cp = clip(acc(L)-acc(L-6), 0) in float64 -> float32 (s20 recipe)",
                            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        (out / "check" / f"in_{d}.json").write_text(json.dumps(dict(date=d, bad=bad, leads=rep), indent=1))
        if bad:
            sys.exit(f"CHECK FAILED {d}: {bad[:8]} (store left as {tmp})")
        tmp.rename(out / "in" / f"{d}.zarr")
        done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        print(f"BUILT in {d} in {time.time() - t0:.0f}s", flush=True)
    print("BUILD_IN_DONE", flush=True)


if __name__ == "__main__":
    main()
