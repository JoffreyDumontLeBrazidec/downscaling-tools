#!/usr/bin/env python3
"""Ensemble means of the 9 km truth (ENS hindcast, stream enfh, O1280) and of the 36 km input (stream eefh, O320)
for the cases of a downscaling read, on the prediction files' own box points.

Why: our prediction files carry ONE truth forecast (the enfh control, y) and ONE input forecast (the eefh control,
x_interp), while the model gives 10 draws from that input. The truth and the input both come from 11-member
hindcast ensembles (cf + pf 1-10, 48r1). This tool fetches the 10 perturbed members of both, and writes the 11-member
means at the same points, so a model ensemble mean can be shown beside the truth's and the input's.

Three steps, three hosts (nothing is ever written to FDB, standing rule 30; MARS reads to files only):
  export   (where the prediction files are, e.g. Jupiter): per case, box-cut lat/lon, y, x_interp and the member mean
           of each listed row -> cases_<tag>.npz
  retrieve (Atos, MARS): enfh O1280 and eefh O320, type cf and pf 1-10, steps lead-12 and lead -> GRIB files
  build    (Atos): 11-member means at the exported points (input through the global O320->O1280 linear matrix, the
           one behind x_interp); the MARS control is checked against y and x_interp before anything is written
           -> refens_<tag>.npz (adds truth_ensmean, input_ensmean, truth_cf_mars, input_cf_mars, n_members)

tp only for now: 12 h window ending at the lead, mm, clipped at 0 (as in the prediction files).
Hindcast dates: hdate = the file date; MARS run date = same month-day of the next year (Mon/Thu runs of 48r1,
2023-06-29..2024-11-11; check with --run-date YYYYMMDD=YYYYMMDD if a date fails).

Usage:
  python refens.py export   --pred-root R --rows CA300,DB300 --cases cases.txt --out cases_tag.npz
  python refens.py retrieve --cases cases.txt --grib-dir G [--dry-run]
  python refens.py build    --cases-npz cases_tag.npz --grib-dir G --up-matrix M --out refens_tag.npz
cases.txt: one case per line "region yyyymmdd lead lat lon half_lon half_lat" (# comments allowed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np

STREAMS = {"truth": ("enfh", 6599680), "input": ("eefh", 421120)}   # stream, expected global points (O1280, O320)
N_PF = 10
MARGIN = 0.3                                                          # degrees kept around the plotted box


# ---------------------------------------------------------------- helpers (pure, tested)
def read_cases(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        reg, date, lead, lat, lon, hx, hy = line.split()
        out.append(dict(region=reg, date=date, lead=int(lead), lat=float(lat), lon=float(lon),
                        hx=float(hx), hy=float(hy)))
    return out


def case_key(c):
    return f"{c['region']}_{c['date']}_L{c['lead']:03d}"


def run_date(hdate, overrides=None):
    if overrides and hdate in overrides:
        return overrides[hdate]
    d = dt.date(int(hdate[:4]) + 1, int(hdate[4:6]), int(hdate[6:]))
    return d.strftime("%Y%m%d")


def to_mm(a):
    """tp to mm; an array already in mm (max > 5) is kept. Clip at 0."""
    a = np.asarray(a, dtype=np.float64)
    if np.nanmax(np.abs(a)) < 5.0:
        a = a * 1000.0
    return np.clip(a, 0, None)


def tp_window_mm(acc_end_m, acc_start_m):
    """12 h window from accumulated tp in metres (start may be None for step 0)."""
    end = np.asarray(acc_end_m, dtype=np.float64)
    start = np.zeros_like(end) if acc_start_m is None else np.asarray(acc_start_m, dtype=np.float64)
    return np.clip((end - start) * 1000.0, 0, None)


def lon180(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def unit_vectors(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


# ---------------------------------------------------------------- export (prediction host)
def cmd_export(a):
    import netCDF4
    rows = a.rows.split(",")
    out = {}
    for c in read_cases(a.cases):
        k = case_key(c)
        for i, row in enumerate(rows):
            f = Path(a.pred_root) / f"{row}_L{c['lead']:03d}" / f"predictions_{c['date']}_step{c['lead']:03d}.nc"
            with netCDF4.Dataset(str(f)) as ds:
                ds.set_auto_mask(False)
                st = [str(s) for s in ds["weather_state"][:]]
                j = st.index("tp")
                if i == 0:
                    lat = np.asarray(ds["lat_hres"][:], dtype=np.float64)
                    lon = lon180(ds["lon_hres"][:])
                    b = (np.abs(lat - c["lat"]) <= c["hy"] + MARGIN) & (np.abs(lon - c["lon"]) <= c["hx"] + MARGIN)
                    out[f"{k}/lat"], out[f"{k}/lon"] = lat[b], lon[b]
                    out[f"{k}/truth_cf"] = to_mm(ds["y"][0, :, j])[b].astype(np.float32)
                    out[f"{k}/input_cf"] = to_mm(ds["x_interp"][0, :, j])[b].astype(np.float32)
                yp = to_mm(ds["y_pred"][0, :, :, j])                       # unit decided on the whole field
                out[f"{k}/{row}_mean"] = yp.mean(0)[b].astype(np.float32)
                out[f"{k}/{row}_n"] = np.int32(yp.shape[0])
        print("exported", k, int(b.sum()), "points", flush=True)
    out["rows"] = np.array(rows)
    out["cases"] = np.array([case_key(c) for c in read_cases(a.cases)])
    out["cases_txt"] = np.array(Path(a.cases).read_text())
    np.savez_compressed(a.out, **out)
    print("EXPORT_DONE", a.out, flush=True)


# ---------------------------------------------------------------- retrieve (Atos, MARS to files)
def grib_name(grib_dir, who, hdate, lead, typ):
    return Path(grib_dir) / f"{STREAMS[who][0]}_{typ}_{hdate}_L{lead:03d}_tp.grib"


def mars_request(who, hdate, lead, typ, target, overrides=None):
    stream = STREAMS[who][0]
    steps = "/".join(str(s) for s in sorted({max(lead - 12, 0), lead}))
    r = [f"retrieve,class=od,expver=0001,stream={stream},type={typ}",
         f"date={run_date(hdate, overrides)},hdate={hdate},time=00,step={steps}",
         "levtype=sfc,param=tp"]
    if typ == "pf":
        r.append(f"number=1/to/{N_PF}")
    r.append(f'target="{target}"')
    return ",\n  ".join(r) + "\n"


def parse_overrides(items):
    return dict(s.split("=") for s in (items or []))


def cmd_retrieve(a):
    ov = parse_overrides(a.run_date)
    Path(a.grib_dir).mkdir(parents=True, exist_ok=True)
    for c in read_cases(a.cases):
        for who in STREAMS:
            for typ in ("cf", "pf"):
                tgt = grib_name(a.grib_dir, who, c["date"], c["lead"], typ)
                if tgt.exists() and tgt.stat().st_size > 0:
                    print("have", tgt.name, flush=True)
                    continue
                req = mars_request(who, c["date"], c["lead"], typ, tgt, ov)
                print(req, flush=True)
                if a.dry_run:
                    continue
                subprocess.run(["mars"], input=req, text=True, check=True)
    print("RETRIEVE_DONE", flush=True)


# ---------------------------------------------------------------- build (Atos)
def read_grib(path):
    """-> dict {(number, endStep): values}, lat, lon (from the first message)."""
    import eccodes
    out, lat, lon = {}, None, None
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                num = int(eccodes.codes_get(h, "number")) if eccodes.codes_get(h, "dataType") == "pf" else 0
                step = int(eccodes.codes_get(h, "endStep"))
                out[(num, step)] = eccodes.codes_get_values(h).astype(np.float64)
                if lat is None:
                    lat = eccodes.codes_get_array(h, "latitudes").astype(np.float64)
                    lon = eccodes.codes_get_array(h, "longitudes").astype(np.float64)
            finally:
                eccodes.codes_release(h)
    return out, lat, lon


def member_windows(grib_dir, who, hdate, lead):
    """{member: tp window mm (global)}, lat, lon; member 0 = control."""
    fields, lat, lon = {}, None, None
    for typ in ("cf", "pf"):
        f, la, lo = read_grib(grib_name(grib_dir, who, hdate, lead, typ))
        fields.update(f)
        if lat is None:
            lat, lon = la, lo
    s0, s1 = max(lead - 12, 0), lead
    mem = sorted({m for m, _ in fields})
    win = {}
    for m in mem:
        start = fields.get((m, s0)) if s0 > 0 else fields.get((m, 0))
        win[m] = tp_window_mm(fields[(m, s1)], start)
    return win, lat, lon


def cmd_build(a):
    from scipy.sparse import load_npz
    from scipy.spatial import cKDTree
    z = dict(np.load(a.cases_npz, allow_pickle=False))
    up = load_npz(a.up_matrix).tocsr()
    trees, out, report = {}, dict(z), []
    for k in z["cases"]:
        k = str(k)
        reg, date, lead = k.split("_")[0], k.split("_")[1], int(k.split("_L")[1])
        lat, lon = z[f"{k}/lat"], z[f"{k}/lon"]
        res = {}
        for who in STREAMS:
            win, glat, glon = member_windows(a.grib_dir, who, date, lead)
            npts = STREAMS[who][1]
            assert glat.size == npts, (who, glat.size, npts)
            if who == "truth":
                if "o1280" not in trees:
                    trees["o1280"] = cKDTree(unit_vectors(glat, lon180(glon)))
                d, gidx = trees["o1280"].query(unit_vectors(lat, lon), k=1)
                assert d.max() < 1e-6, f"box points not on the O1280 GRIB grid (max chord {d.max():.2e})"
                trees["gidx"] = gidx
                pick = lambda v: v[gidx]
            else:
                assert up.shape[1] == npts, up.shape
                pick = lambda v: np.clip(np.asarray(up @ v), 0, None)[trees["gidx"]]
            vals = np.stack([pick(win[m]) for m in sorted(win)])            # members x points
            res[who] = (vals, sorted(win))
        for who in STREAMS:
            vals, mem = res[who]
            assert mem == list(range(N_PF + 1)), (who, mem)
            cf_ref = z[f"{k}/{who}_cf"].astype(np.float64)
            dif = np.abs(vals[0] - cf_ref)
            scale = max(cf_ref.max(), 1.0)
            line = (f"{k} {who}: MARS control vs prediction file: max |diff| {dif.max():.4f} mm "
                    f"({dif.max() / scale:.2e} of max {cf_ref.max():.1f} mm), mean |diff| {dif.mean():.5f} mm")
            print(line, flush=True); report.append(line)
            if dif.max() > a.tol_mm:
                sys.exit(f"CHECK FAILED ({who}, {k}): the MARS control does not match the stored field; wrong run "
                         f"date, grid order or matrix. Nothing written.")
            out[f"{k}/{who}_ensmean"] = vals.mean(0).astype(np.float32)
            out[f"{k}/{who}_pfmean"] = vals[1:].mean(0).astype(np.float32)
            out[f"{k}/{who}_cf_mars"] = vals[0].astype(np.float32)
            out[f"{k}/{who}_n"] = np.int32(vals.shape[0])
    out["check_report"] = np.array("\n".join(report))
    np.savez_compressed(a.out, **out)
    print("BUILD_DONE", a.out, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    e = s.add_parser("export"); e.add_argument("--pred-root", required=True); e.add_argument("--rows", required=True)
    e.add_argument("--cases", required=True); e.add_argument("--out", required=True)
    r = s.add_parser("retrieve"); r.add_argument("--cases", required=True); r.add_argument("--grib-dir", required=True)
    r.add_argument("--run-date", nargs="*", help="hdate=rundate overrides"); r.add_argument("--dry-run", action="store_true")
    b = s.add_parser("build"); b.add_argument("--cases-npz", required=True); b.add_argument("--grib-dir", required=True)
    b.add_argument("--up-matrix", required=True); b.add_argument("--out", required=True)
    b.add_argument("--tol-mm", type=float, default=0.05, help="max |MARS control - stored field| allowed (mm)")
    a = p.parse_args(argv)
    {"export": cmd_export, "retrieve": cmd_retrieve, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
