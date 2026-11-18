# Reduced field files for the wind/cyclone realism read (17 Nov 2026). Login-node CPU, kit/pyenv.sh.
# Batch 1 (no arguments): the 8 PAIRS below into $W/fields/ plus orog.npz.
# Batch 2: python extract.py --lead 24 --dates 20230101,20230122 --out $W/fields24 [--no-orog]
#   same keys, same checks; each date line ends with CHECKS_OK or CHECK_FAIL(<reasons>); files written as .part then renamed.
import argparse, json, numpy as np, netCDF4 as nc, zarr, os, sys
# Paths and row from the environment (eval.cli final_battery); defaults = the 17 Nov Jupiter reference run.
U = os.environ.get("FB_USER_ROOT", "/e/data1/jureap-data/ecmwf/users/jdumont")
X = os.environ.get("FB_PRED_ROOT", U + "/outputs/ens_member_box_20261104")
A = os.environ.get("FB_MEMBER_STORES", U + "/agent-work/20261104-ens-member-box")
W = os.environ.get("FB_WIND_WORK", U + "/agent-work/20261117-realism-wind")
ROW = os.environ.get("FB_ROW", "GLB450e2_m")
F = os.environ.get("FB_FORCINGS", U + "/datasets/o320_o1280_europe_e2/in_hres_o1280_forcings.zarr")
PAIRS = [("20230226",24),("20230829",24),("20230909",24),("20231021",24),("20231031",48),("20230111",24),("20230822",24),("20230829",72)]
NAMES = ["10u","10v","msl"]
ap = argparse.ArgumentParser()
ap.add_argument("--lead", type=int, default=None); ap.add_argument("--dates", default="")
ap.add_argument("--out", default=f"{W}/fields"); ap.add_argument("--no-orog", action="store_true")
ap.add_argument("--orog-only", action="store_true", help="write <out>/orog.npz only (no field files)")
a = ap.parse_args()
if a.orog_only:
    PAIRS = []
elif a.lead is not None:
    PAIRS = [(d, a.lead) for d in a.dates.split(",") if d]
    a.no_orog = True
os.makedirs(a.out, exist_ok=True)
stores = {s: zarr.open(f"{A}/{s}/emb_out_hres_o1280_pf.zarr", mode="r") for s in ("stores","stores_leads")}
for s, z in stores.items():
    assert json.loads(z.attrs["variables"].replace("'", '"')) if isinstance(z.attrs["variables"], str) else True
    print(s, "vars", z.attrs["variables"], "leads", z.attrs["leads"], "shape", z["data"].shape, "dates[:3]", z["dates"][:3], "e2_row[:3]", z["e2_row"][:3])
def find_row(z, fake):
    d = [str(np.datetime64(v, "s")) for v in z["dates"][:]]
    hits = [i for i, v in enumerate(d) if v == fake[:19]]
    return hits
for date, lead in PAIRS:
    f = f"{X}/pred/{ROW}_L{lead:03d}/predictions_{date}_step{lead:03d}.nc"
    if not os.path.exists(f):
        print("MISSING", f, "CHECK_FAIL(missing)", flush=True); continue
    fail = []
    d = nc.Dataset(f); ws = [str(x) for x in d["weather_state"][:]]; ix = [ws.index(n) for n in NAMES]
    lat = d["lat_hres"][:].astype(np.float32); lon = d["lon_hres"][:].astype(np.float32)
    pred = d["y_pred"][0][:, :, ix].astype(np.float32)
    inp = d["x_interp_members"][0][:, :, ix].astype(np.float32)
    y = d["y"][0][:, ix].astype(np.float32)
    fake = d.getncattr("fake_date"); idx = d.getncattr("idx")
    out = dict(lat=lat, lon=lon, names=np.array(NAMES), pred=pred, inp=inp, y=y)
    msg = f"{date} L{lead:03d} fake={fake} idx={idx}"
    if pred.shape[0] != 10 or inp.shape[0] != 10: fail.append(f"members pred={pred.shape} inp={inp.shape}")
    if not (np.isfinite(pred).all() and np.isfinite(inp).all() and np.isfinite(y).all()): fail.append("nonfinite_pred_inp_y")
    got = False
    for s in ("stores", "stores_leads"):
        z = stores[s]; hits = find_row(z, fake)
        if not hits: continue
        if len(hits) > 1: fail.append(f"{len(hits)}_rows_match")
        r = hits[0]; vs = z.attrs["variables"]; vs = json.loads(vs.replace("'", '"')) if isinstance(vs, str) else list(vs)
        vi = [vs.index(n) for n in NAMES]
        t = np.asarray(z["data"][r])[vi].transpose(1, 2, 0).astype(np.float32)  # (10, P, 3)
        zl = np.asarray(z["latitudes"][:]); zo = np.asarray(z["longitudes"][:])
        dlat = float(np.abs(zl - lat).max()); dlon = float(np.abs(((zo - lon) + 180) % 360 - 180).max())
        if dlat > 1e-3 or dlon > 1e-3: fail.append(f"grid dlat={dlat:.2e} dlon={dlon:.2e}")
        dmem = [float(np.nanmax(np.abs(t[k] - y))) for k in range(10)]
        cf = ""
        if "data_cf" in z:
            c = np.asarray(z["data_cf"][r])[vi].T
            cfd = np.nanmax(np.abs(c - y), axis=0)
            cf = " cf_vs_y_maxabs=" + ",".join(f"{v:.4g}" for v in cfd)
            if float(cfd.max()) > 0: fail.append("cf_ne_y")
        nant = int(np.isnan(t).sum())
        if nant: fail.append(f"nan_truth={nant}")
        msg += (f" truth={s} row={r} e2_row={z['e2_row'][r]} dlat={dlat:.2e} dlon={dlon:.2e} "
                f"member_vs_y_maxabs(min over members, all 3 vars)={min(dmem):.4g} per-member={[round(v,3) for v in dmem]}{cf}"
                f" nan_truth={nant}")
        out["truth"] = t; got = True; break
    if not got: msg += " TRUTH NOT MATCHED"; fail.append("truth_not_matched")
    msg += f" pred_vs_inp_members_mean_absdiff={[round(float(v),3) for v in np.abs(pred-inp).mean(axis=(0,1))]}"
    path = f"{a.out}/{ROW}_{date}_L{lead:03d}.npz"
    with open(path + ".part", "wb") as fh:
        np.savez_compressed(fh, **out)
    os.replace(path + ".part", path)
    msg += " CHECKS_OK" if not fail else " CHECK_FAIL(" + "; ".join(fail) + ")"
    print(msg, flush=True)
if a.no_orog:
    print("done"); sys.exit(0)
# orography and land-sea mask
z = zarr.open(F, mode="r"); vs = z.attrs["variables"]; print("forcing vars", vs, "shape", z["data"].shape)
vi = {n: vs.index(n) for n in ("z", "lsm") if n in vs}
row = np.asarray(z["data"][0])  # (vars, ens, P)
o = dict(lat=np.asarray(z["latitudes"][:]).astype(np.float32), lon=np.asarray(z["longitudes"][:]).astype(np.float32))
for n, i in vi.items(): o[n] = row[i, 0].astype(np.float32)
o["orog_m"] = o["z"] / 9.80665 if "z" in o else None
import glob as _glob
d = nc.Dataset(sorted(_glob.glob(f"{X}/pred/{ROW}_L*/predictions_*.nc"))[0])
print("orog points", o["lat"].shape, "dlat vs pred", float(np.abs(o["lat"] - d["lat_hres"][:]).max()), "dlon", float(np.abs(((o["lon"] - d["lon_hres"][:]) + 180) % 360 - 180).max()),
      "z range", float(o["z"].min()), float(o["z"].max()), "lsm range", float(o["lsm"].min()), float(o["lsm"].max()))
np.savez_compressed(f"{a.out}/orog.npz", **o)
print("done")
