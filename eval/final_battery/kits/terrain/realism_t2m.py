#!/usr/bin/env python
"""Terrain / coast realism tables for GL512 450k EMA-2000 (row GLB450e2_m), E2 box, 9 km lane.

Subcommands
  statics            build out/statics.npz (orography, masks, coast distance, high-pass operator)
  case DATE LEAD     compute one (date, lead) partial -> out/parts/<date>_L<lead>.pkl (+ .done)
  run [--workers N]  all missing cases in a process pool
  aggregate          per-date CSVs (out/full/) and date-aggregated CSVs (out/tables/)

Definitions (see README): zhr = O1280 orography (m, forcing store z / g); zlr = O320 eefh orography
through the O320->O1280 truncation matrix (same as x_interp); dz = zhr - zlr. Interior = box minus a
1-degree strip. High-pass = judge's zoom_maps operator: nearest-neighbour regrid to 0.08 deg, Gaussian
sigma = 0.6 / 0.08 / 2 cells, bilinear back to points, subtracted.
Pairing: eefh member k and enfh member k are NOT paired (20261104_ens_member_box README). err = model k
minus truth pf k is therefore an unpaired pair; floor = truth pf k minus truth pf k+1 (cyclic). The
only paired truth/input couple is the control: y (9 km enfh cf) minus x_interp (eefh cf interpolated).
"""
import os, sys, glob, json, pickle, time, argparse
import numpy as np

G = 9.80665
RD = 287.05
# Paths and the row are read from the environment (set by `python -m eval.cli final_battery`); the defaults are the
# Jupiter values of the 17 Nov 2026 reference run (row GLB450e2_m), so the kit run by hand there is unchanged.
_env = os.environ.get
U = _env("FB_USER_ROOT", "/e/data1/jureap-data/ecmwf/users/jdumont")
A = _env("FB_MEMBER_STORES", U + "/agent-work/20261104-ens-member-box")   # stores/, stores_leads/ (truth + input pf)
X = _env("FB_PRED_ROOT", U + "/outputs/ens_member_box_20261104")          # X/pred/<ROW>_L<lead>/predictions_*.nc
D = _env("FB_DATASETS", U + "/datasets")
S = _env("FB_WORK", U + "/agent-work/20261117-realism-t2m")
OUT = _env("FB_TERRAIN_OUT", S + "/out")
ROW = _env("FB_ROW", "GLB450e2_m")
LEADS = [int(x) for x in _env("FB_LEADS", "12 24 36 48 60 72").split()]
W = ["10u", "10v", "2t", "msl", "sp", "tp", "cp"]
V3 = ["2t", "msl", "sp"]
CV = ["2t", "msl", "sp", "2d", "skt"]
MAT = D + "/o320_o1280_europe_e2/assets/inter_mat/interpol_O320_to_O1280_linear-europe_e2_10_70N_80W_40E_halo1p5.mat.npz"
OROG320 = _env("FB_OROG320", U + "/station_caches/orography_O320_eefh.grib")
FORC = D + "/o320_o1280_europe_e2/in_hres_o1280_forcings.zarr"
OUTV4 = D + "/o320_o1280_europe_e2_v4/out_hres_o1280.zarr"
INV4 = D + "/o320_o1280_europe_e2_v4/in_lres_o320_halo1p5.zarr"
G320 = D + "/o320_o1280_europe_e2/assets/grids/grid-o320-europe_e2_10_70N_80W_40E_halo1p5.npz"

REGIONS = {  # lat0, lat1, lon0, lon1
    "alps": (43, 49, 4, 16), "norway": (58, 69, 4, 20), "iceland": (63, 67, -25, -13),
    "s_greenland": (59.5, 69, -55, -40), "pyrenees": (41.5, 43.5, -2, 3.5), "atlas": (29, 36, -10, 6),
    "turkey_caucasus": (36, 44, 26, 39), "appalachians": (34, 46, -79, -70),
}
ZB = [0, 200, 500, 1000, 1500, 2000, 3000, 1e9]
ZB_L = ["0-200", "200-500", "500-1000", "1000-1500", "1500-2000", "2000-3000", "3000+"]
DZB = [-1e9, -400, -200, -50, 50, 200, 400, 1e9]
DZB_L = ["<-400", "-400..-200", "-200..-50", "-50..50", "50..200", "200..400", ">400"]
CB = [-1e9, -100, -30, -10, 0, 10, 30, 100, 1e9]
CB_L = ["sea<-100", "sea-100..-30", "sea-30..-10", "sea-10..0", "land0..10", "land10..30", "land30..100", "land>100"]
D3B = [0, 3, 6, 9, 12, 15, 18, 1e9]
D3B_L = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18+"]
QS = [0.0001, 0.001, 0.01, 0.99, 0.999, 0.9999]
QS_L = ["q0.01", "q0.1", "q1", "q99", "q99.9", "q99.99"]
HP_RES, HP_CUT, LAT_SCALE = 0.08, 0.6, 1.4


def season(month):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA"}.get(month, "SON")


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def states(d):
    """Variable names of a prediction file, in file order (weather_state); the 7-state list W when absent.
    Indexing by name lets rows that save more outputs (10-output rows, 12-output exports) run unchanged."""
    if "weather_state" not in d.variables:
        return list(W)
    v = d["weather_state"][:]
    try:
        return [str(x) for x in v]
    except Exception:  # noqa: char arrays
        import netCDF4
        return [str(x) for x in netCDF4.chartostring(v)]


def load_matrix():
    import scipy.sparse as sp
    m = np.load(MAT)
    return sp.csr_matrix((m["data"], m["indices"], m["indptr"]), shape=tuple(m["shape"]))


# ----------------------------------------------------------------------------------------- statics
def build_statics():
    import zarr, eccodes
    from scipy.spatial import cKDTree
    t0 = time.time()
    M = load_matrix()
    fz = zarr.open(FORC, mode="r")
    fv = list(fz.attrs["variables"])
    lat = fz["latitudes"][:].astype(np.float64)
    lon = fz["longitudes"][:].astype(np.float64)
    lon = np.where(lon >= 180.0, lon - 360.0, lon)
    zhr = fz["data"][0, fv.index("z"), 0, :].astype(np.float64) / G
    lsm = fz["data"][0, fv.index("lsm"), 0, :].astype(np.float64)
    g3 = np.load(G320)
    with open(OROG320, "rb") as f:
        h = eccodes.codes_grib_new_from_file(f)
        la, lo, va = (eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes"),
                      eccodes.codes_get_values(h))
        eccodes.codes_release(h)
    dd, ii = cKDTree(xyz(la, lo)).query(xyz(g3["latitudes"], g3["longitudes"]))
    assert dd.max() * 6371 < 0.01, dd.max()
    z320 = va[ii] / G
    zlr = M @ z320
    interior = (lat >= 11) & (lat <= 69) & (lon >= -79) & (lon <= 39)
    land = lsm >= 0.5
    water = ~land
    # inland-water proxy (no lake-cover field on disk): connected water bodies on the 0.08-degree grid
    # (8-connectivity) that do not touch the box edge and are smaller than 5000 cells (~250,000 km2):
    # Great Lakes, Ladoga, Alpine lakes in; Black Sea, Caspian, Mediterranean out.
    from scipy import ndimage
    gx0 = np.arange(-80.0, 40.0 + 1e-9, HP_RES); gy0 = np.arange(10.0, 70.0 + 1e-9, HP_RES)
    GX0, GY0 = np.meshgrid(gx0, gy0)
    _, nn0 = cKDTree(np.c_[lon, lat * LAT_SCALE]).query(np.c_[GX0.ravel(), GY0.ravel() * LAT_SCALE], workers=8)
    wg = water[nn0].reshape(GX0.shape)
    labw, nlab = ndimage.label(wg, structure=np.ones((3, 3)))
    size = np.bincount(labw.ravel())
    edge = np.unique(np.r_[labw[0], labw[-1], labw[:, 0], labw[:, -1]])
    small = (size < 5000); small[0] = False; small[edge] = False
    iy0 = np.clip(np.rint((lat - gy0[0]) / HP_RES).astype(int), 0, len(gy0) - 1)
    ix0 = np.clip(np.rint((lon - gx0[0]) / HP_RES).astype(int), 0, len(gx0) - 1)
    inland = water & small[labw[iy0, ix0]]
    P = xyz(lat, lon)
    dl, _ = cKDTree(P[water]).query(P[land])
    dw, _ = cKDTree(P[land]).query(P[water])
    coast = np.zeros_like(lat)
    coast[land] = dl * 6371.0
    coast[water] = -dw * 6371.0
    d3, _ = cKDTree(xyz(g3["latitudes"], g3["longitudes"])).query(P)
    d320 = d3 * 6371.0
    # high-pass operator, whole box
    gx = np.arange(-80.0, 40.0 + 1e-9, HP_RES)
    gy = np.arange(10.0, 70.0 + 1e-9, HP_RES)
    GX, GY = np.meshgrid(gx, gy)
    _, nn = cKDTree(np.c_[lon, lat * LAT_SCALE]).query(np.c_[GX.ravel(), GY.ravel() * LAT_SCALE], workers=8)
    fx = np.clip((lon - gx[0]) / HP_RES, 0, len(gx) - 1.000001)
    fy = np.clip((lat - gy[0]) / HP_RES, 0, len(gy) - 1.000001)
    np.savez(OUT + "/statics.npz", lat=lat, lon=lon, zhr=zhr, zlr=zlr, lsm=lsm, interior=interior, land=land,
             inland=inland, coast=coast, d320=d320, nn=nn.astype(np.int32), gshape=np.array(GX.shape),
             fx=fx, fy=fy, z320=z320)
    print(f"STATICS_DONE n={lat.size} interior={interior.sum()} land_int={(land & interior).sum()} "
          f"inland_int={(inland & interior).sum()} zhr[{zhr.min():.0f},{zhr.max():.0f}] zlr[{zlr.min():.0f},{zlr.max():.0f}] "
          f"d320max={d320.max():.1f}km grid={GX.shape} {time.time() - t0:.0f}s", flush=True)


class Stat:
    def __init__(self):
        s = np.load(OUT + "/statics.npz")
        for k in s.files:
            setattr(self, k, s[k])
        self.gshape = tuple(int(x) for x in self.gshape)
        self.dz = self.zhr - self.zlr
        ix = np.floor(self.fx).astype(np.int64)
        iy = np.floor(self.fy).astype(np.int64)
        wx, wy = self.fx - ix, self.fy - iy
        self.bl = (iy, ix, wx, wy)
        I, L = self.interior, self.land & self.interior
        self.I, self.L = I, L
        self.sea = I & ~self.land & ~self.inland
        self.inl = I & self.inland
        lab = lambda v, b: np.digitize(v, b[1:-1])
        # scheme -> labels (int; -1 = not counted), labels list
        zb = np.where(L, lab(self.zhr, ZB), -1); zb = np.where(self.sea, len(ZB_L), zb)
        dzb = np.where(L, lab(self.dz, DZB), -1)
        cb = np.where(I & ~self.inland, lab(self.coast, CB), -1); cb = np.where(self.inl, len(CB_L), cb)
        allb = np.full(self.lat.size, -1); allb[L] = 0; allb[self.sea] = 1; allb[self.inl] = 2
        self.schemes = {
            "zhr_band": (zb, ZB_L + ["sea"]),
            "dz_band": (dzb, DZB_L),
            "coast_km": (cb, CB_L + ["inland_water_proxy"]),
            "all": (allb, ["land", "sea", "inland_water_proxy"]),
        }
        self.lst_base = np.where(L, 0, np.where(self.sea, 1, -1))  # LST scheme built per lead
        self.regions = {}
        for r, (a, b, c, d) in REGIONS.items():
            self.regions[r] = L & (self.lat >= a) & (self.lat <= b) & (self.lon >= c) & (self.lon <= d)
        self.regions["box_land"] = L
        self.d3 = np.where(L, lab(self.d320, D3B), -1)

    def lst_scheme(self, lead):
        hloc = np.mod((lead % 24) + self.lon / 15.0, 24.0)
        b = np.minimum((hloc // 3).astype(int), 7)
        lab = np.where(self.lst_base == 0, b, np.where(self.lst_base == 1, 8 + b, -1))
        names = [f"land_{3*i:02d}-{3*i+3:02d}" for i in range(8)] + [f"sea_{3*i:02d}-{3*i+3:02d}" for i in range(8)]
        return lab, names

    def lowpass(self, v):
        """Judge low-pass at points (v: (..., N)); returns array same shape."""
        from scipy.ndimage import gaussian_filter
        v = np.asarray(v, dtype=np.float64)
        flat = v.reshape(-1, v.shape[-1])
        out = np.empty_like(flat)
        iy, ix, wx, wy = self.bl
        sig = HP_CUT / HP_RES / 2.0
        for i in range(flat.shape[0]):
            g = gaussian_filter(flat[i][self.nn].reshape(self.gshape), sig, mode="nearest")
            out[i] = ((1 - wy) * ((1 - wx) * g[iy, ix] + wx * g[iy, ix + 1])
                      + wy * ((1 - wx) * g[iy + 1, ix] + wx * g[iy + 1, ix + 1]))
        return out.reshape(v.shape)


def binacc(lab, nb, e):
    """e: (..., N) -> (..., nb, 3) [n, sum, sumsq] over points with lab >= 0."""
    m = lab >= 0
    l = lab[m]
    e2 = e.reshape(-1, e.shape[-1])[:, m]
    out = np.zeros((e2.shape[0], nb, 3))
    for i in range(e2.shape[0]):
        x = e2[i].astype(np.float64)
        ok = np.isfinite(x)
        out[i, :, 0] = np.bincount(l[ok], minlength=nb)
        out[i, :, 1] = np.bincount(l[ok], weights=x[ok], minlength=nb)
        out[i, :, 2] = np.bincount(l[ok], weights=x[ok] ** 2, minlength=nb)
    return out.reshape(e.shape[:-1] + (nb, 3))


def slope_fit(x, y):
    x = x - x.mean(); y = y - y.mean()
    vx, vy = (x * x).sum(), (y * y).sum()
    c = (x * y).sum()
    return c / vx, (c * c / (vx * vy) if vy > 0 else np.nan)


def store_row(lead, hdate):
    import zarr
    sub = "stores" if lead in (24, 48) else "stores_leads"
    zi = zarr.open(f"{A}/{sub}/emb_in_lres_o320_halo1p5_pf.zarr", mode="r")
    zo = zarr.open(f"{A}/{sub}/emb_out_hres_o1280_pf.zarr", mode="r")
    rm = zo.attrs["row_map"]
    r = [i for i, x in enumerate(rm) if x["hdate"] == hdate and int(x["lead"]) == lead]
    assert len(r) == 1, (hdate, lead, r)
    return zi, zo, r[0]


# ----------------------------------------------------------------------------------------- one case
def do_case(hdate, lead, st=None, M=None):
    import zarr, netCDF4 as nc
    t0 = time.time()
    st = st or Stat()
    M = M if M is not None else load_matrix()
    fn = f"{X}/pred/{ROW}_L{lead:03d}/predictions_{hdate}_step{lead:03d}.nc"
    with nc.Dataset(fn) as d:
        d.set_auto_mask(False)
        yp = d["y_pred"][0]           # (10, N, 7)
        xm = d["x_interp_members"][0]  # (10, N, 7)
        yc = d["y"][0]                 # (N, 7) truth control
        xc = d["x_interp"][0]          # (N, 7) input control
        assert np.abs(np.where(d["lon_hres"][:] < 0, d["lon_hres"][:] + 360, d["lon_hres"][:])
                      - np.where(st.lon < 0, st.lon + 360, st.lon)).max() < 1e-6
        names = states(d)
    iv = [names.index(v) for v in V3]
    i2t = names.index("2t")
    mod = np.moveaxis(yp[:, :, iv], 2, 0).astype(np.float64)   # (3, 10, N)
    inp = np.moveaxis(xm[:, :, iv], 2, 0).astype(np.float64)
    zi, zo, r = store_row(lead, hdate)
    ov = list(zo.attrs["variables"])
    tpf = zo["data"][r][[ov.index(v) for v in V3]].astype(np.float64)  # (3, 10, N)
    er = int(zo["e2_row"][r])
    o4 = zarr.open(OUTV4, mode="r"); o4v = list(o4.attrs["variables"])
    i4 = zarr.open(INV4, mode="r"); i4v = list(i4.attrs["variables"])
    tc = np.stack([o4["data"][er, o4v.index(v), 0, :] for v in CV]).astype(np.float64)       # (5, N)
    ic = np.stack([M @ i4["data"][er, i4v.index(v), 0, :].astype(np.float64) for v in CV])   # (5, N)
    assert np.abs(tc[0] - yc[:, i2t]).max() < 1e-3 and np.abs(ic[0] - xc[:, i2t]).max() < 1e-3, "control mismatch"
    ziv = list(zi.attrs["variables"])
    i2d = (M @ zi["data"][r, ziv.index("2d"), :, :].astype(np.float64).T).T   # (10, N) input member 2d
    iskt = (M @ zi["data"][r, ziv.index("skt"), :, :].astype(np.float64).T).T
    del yp, xm
    flo = tpf[:, np.roll(np.arange(10), -1), :]
    res = {"hdate": hdate, "lead": lead, "season": season(int(hdate[4:6]))}

    # T1 ------------------------------------------------------------------------------------------
    schemes = dict(st.schemes); schemes["lst"] = st.lst_scheme(lead)
    t1 = {}
    for sname, (lab, names) in schemes.items():
        nb = len(names)
        t1[sname] = {
            "model": binacc(lab, nb, mod - tpf), "input": binacc(lab, nb, inp - tpf),
            "floor": binacc(lab, nb, tpf - flo), "ctrl_pair": binacc(lab, nb, tc - ic),
        }
    res["t1"] = t1

    # high-pass / low-pass of the fields needed -------------------------------------------------------
    T2i, MSLi, SPi = 0, 1, 2
    low = {}
    low["mod2t"] = st.lowpass(mod[0]); low["tpf2t"] = st.lowpass(tpf[0]); low["inp2t"] = st.lowpass(inp[0])
    low["tc2t"] = st.lowpass(tc[0]); low["ic2t"] = st.lowpass(ic[0])
    low["modmsl"] = st.lowpass(mod[1]); low["tpfmsl"] = st.lowpass(tpf[1]); low["inpmsl"] = st.lowpass(inp[1])
    low["tcmsl"] = st.lowpass(tc[1]); low["icmsl"] = st.lowpass(ic[1])
    hp = {"mod2t": mod[0] - low["mod2t"], "tpf2t": tpf[0] - low["tpf2t"], "inp2t": inp[0] - low["inp2t"],
          "tc2t": tc[0] - low["tc2t"],
          "modmsl": mod[1] - low["modmsl"], "tpfmsl": tpf[1] - low["tpfmsl"], "inpmsl": inp[1] - low["inpmsl"],
          "tcmsl": tc[1] - low["tcmsl"]}
    I, L = st.I, st.L

    # T2 lapse slopes -------------------------------------------------------------------------------
    t2 = []
    ksc = {"2t": 1000.0, "sp": 10.0, "2d": 1000.0, "skt": 1000.0}  # K/km ; Pa/m -> hPa/km
    for reg, rm in st.regions.items():
        if reg == "box_land":
            continue
        m = rm & (np.abs(st.dz) > 100)
        n = int(m.sum())
        if n < 30:
            continue
        x = st.dz[m]
        for vi, v in ((0, "2t"), (2, "sp")):
            for k in range(10):
                s, r2 = slope_fit(x, (mod[vi, k] - inp[vi, k])[m]); t2.append(("model", reg, v, k, s * ksc[v], r2, n))
                s, r2 = slope_fit(x, (tpf[vi, k] - inp[vi, k])[m]); t2.append(("truth_pf_unpaired", reg, v, k, s * ksc[v], r2, n))
        for ci, v in enumerate(CV):
            if v == "msl":
                continue
            s, r2 = slope_fit(x, (tc[ci] - ic[ci])[m]); t2.append(("truth_ctrl", reg, v, -1, s * ksc[v], r2, n))
    res["t2"] = t2

    # T3 hydrostatic consistency ---------------------------------------------------------------------
    def rfield(msl, sp, t2m, z):
        return np.log(msl / sp) - G * z / (RD * (t2m + 0.00325 * z))
    rm_ = rfield(mod[1], mod[2], mod[0], st.zhr)
    rt_ = rfield(tpf[1], tpf[2], tpf[0], st.zhr)
    rf_ = rt_[np.roll(np.arange(10), -1)]
    rtc = rfield(tc[1], tc[2], tc[0], st.zhr)
    ri_ = rfield(inp[1], inp[2], inp[0], st.zlr)
    mt = L & (st.zhr > 500); fl = L & (st.zhr < 200)
    hpr_m = rm_ - st.lowpass(rm_); hpr_t = rt_ - st.lowpass(rt_); hpr_tc = rtc - st.lowpass(rtc)
    hpr_i = ri_ - st.lowpass(ri_)
    t3 = []
    for k in range(10):
        t3 += [("model", k, "hp_r_std_mtn", np.std(hpr_m[k][mt])), ("truth_pf", k, "hp_r_std_mtn", np.std(hpr_t[k][mt])),
               ("input", k, "hp_r_std_mtn", np.std(hpr_i[k][mt])),
               ("model_minus_truth_pf", k, "r_diff_std_mtn", np.std((rm_[k] - rt_[k])[mt])),
               ("floor_pf_pair", k, "r_diff_std_mtn", np.std((rt_[k] - rf_[k])[mt])),
               ("model_minus_truth_pf", k, "hp_r_diff_std_mtn", np.std((hpr_m[k] - hpr_t[k])[mt])),
               ("floor_pf_pair", k, "hp_r_diff_std_mtn", np.std((hpr_t[k] - hpr_t[(k + 1) % 10])[mt]))]
        for nm, key in (("model", "modmsl"), ("truth_pf", "tpfmsl"), ("input", "inpmsl")):
            t3 += [(nm, k, "hp_msl_std_mtn", np.std(hp[key][k][mt])), (nm, k, "hp_msl_std_flat", np.std(hp[key][k][fl]))]
    t3 += [("truth_ctrl", -1, "hp_r_std_mtn", np.std(hpr_tc[mt])), ("truth_ctrl", -1, "hp_msl_std_mtn", np.std(hp["tcmsl"][mt])),
           ("truth_ctrl", -1, "hp_msl_std_flat", np.std(hp["tcmsl"][fl]))]
    res["t3"] = t3
    res["t3_n"] = (int(mt.sum()), int(fl.sum()))

    # T4 2d vs 2t, skt - 2t ----------------------------------------------------------------------------
    t4 = []
    lstlab, lstn = st.lst_scheme(lead)
    for sname, smask in (("land", L), ("sea", st.sea)):
        def frac(t2m, d2m, src, k):
            ex = (d2m - t2m)[smask]
            t4.append((src, k, sname, "2d_gt_2t+0.1", float((ex > 0.1).mean())))
            t4.append((src, k, sname, "2d_gt_2t+1", float((ex > 1.0).mean())))
            t4.append((src, k, sname, "max_2d_minus_2t", float(ex.max())))
        frac(tc[0], tc[3], "truth_ctrl", -1)
        frac(ic[0], ic[3], "input_ctrl", -1)
        for k in range(10):
            frac(mod[0, k], i2d[k], "proxy_model2t_vs_input2d", k)
            frac(tpf[0, k], i2d[k], "proxy_truthpf2t_vs_input2d", k)
            frac(inp[0, k], i2d[k], "input_member", k)
    # skt - 2t by land/sea x lst
    t4s = []
    for b, nm in enumerate(lstn):
        m = lstlab == b
        if m.sum() < 100:
            continue
        for src, e in (("truth_ctrl", tc[4] - tc[0]), ("input_ctrl", ic[4] - ic[0])):
            x = e[m]; t4s.append((src, -1, nm, int(m.sum()), x.mean(), *np.percentile(x, [1, 99])))
        for k in range(10):
            for src, e in (("proxy_inputskt_minus_model2t", iskt[k] - mod[0, k]),
                           ("proxy_inputskt_minus_truthpf2t", iskt[k] - tpf[0, k])):
                x = e[m]; t4s.append((src, k, nm, int(m.sum()), x.mean(), *np.percentile(x, [1, 99])))
    res["t4"] = t4; res["t4s"] = t4s

    # T5 imprint vs distance to nearest O320 point ---------------------------------------------------------
    sub = {"land": st.d3 >= 0, "mountain": (st.d3 >= 0) & (np.abs(st.dz) > 100), "flat": (st.d3 >= 0) & (np.abs(st.dz) <= 100)}
    t5 = {}
    for sn, sm in sub.items():
        lab = np.where(sm, st.d3, -1)
        t5[sn] = {"hp_model": binacc(lab, 7, hp["mod2t"]), "hp_truth_pf": binacc(lab, 7, hp["tpf2t"]),
                  "hp_input": binacc(lab, 7, hp["inp2t"]), "hp_truth_ctrl": binacc(lab, 7, hp["tc2t"]),
                  "abs_hp_model": binacc(lab, 7, np.abs(hp["mod2t"])), "abs_hp_truth_pf": binacc(lab, 7, np.abs(hp["tpf2t"])),
                  "abs_hp_input": binacc(lab, 7, np.abs(hp["inp2t"])), "abs_hp_truth_ctrl": binacc(lab, 7, np.abs(hp["tc2t"])),
                  "err_model": binacc(lab, 7, mod[0] - tpf[0]), "err_floor": binacc(lab, 7, tpf[0] - flo[0]),
                  "hp_model_minus_input": binacc(lab, 7, hp["mod2t"] - hp["inp2t"])}
    res["t5"] = t5

    # T6 tails ------------------------------------------------------------------------------------------
    t6 = []
    for reg, rm in st.regions.items():
        n = int(rm.sum())
        if n < 50:
            continue
        for fld, arrs in (("raw", (("model", mod[0]), ("truth_pf", tpf[0]), ("input", inp[0]))),
                          ("hp", (("model", hp["mod2t"]), ("truth_pf", hp["tpf2t"]), ("input", hp["inp2t"])))):
            for src, a in arrs:
                q = np.quantile(a[:, rm], QS, axis=1).T  # (10, 6)
                for k in range(10):
                    qq = [q[k, j] if n * min(QS[j], 1 - QS[j]) >= 10 else np.nan for j in range(6)]
                    t6.append((src, reg, fld, k, n, *qq))
    res["t6"] = t6

    # T7 large-scale move -------------------------------------------------------------------------------
    t7 = []
    for v, a, b, c, dd in (("2t", "mod2t", "inp2t", "tpf2t", ("tc2t", "ic2t")), ("msl", "modmsl", "inpmsl", "tpfmsl", ("tcmsl", "icmsl"))):
        for sname, sm in (("interior", I), ("land", L)):
            lm = low[a][:, sm] - low[b][:, sm]; lt = low[c][:, sm] - low[b][:, sm]
            for k in range(10):
                t7.append(("model_minus_input", v, sname, k, float(np.sqrt((lm[k] ** 2).mean())), float(lm[k].mean())))
                t7.append(("truth_pf_minus_input_unpaired", v, sname, k, float(np.sqrt((lt[k] ** 2).mean())), float(lt[k].mean())))
            lc = low[dd[0]][sm] - low[dd[1]][sm]
            t7.append(("truth_ctrl_minus_input_ctrl", v, sname, -1, float(np.sqrt((lc ** 2).mean())), float(lc.mean())))
    res["t7"] = t7

    # T8 spread -----------------------------------------------------------------------------------------
    lab, names = st.schemes["zhr_band"]
    t8 = {}
    for src, a in (("model", mod), ("truth_pf", tpf), ("input", inp)):
        var = a.var(axis=1, ddof=1)  # (3, N)
        t8[src] = binacc(lab, len(names), var)
    res["t8"] = t8

    os.makedirs(OUT + "/parts", exist_ok=True)
    p = f"{OUT}/parts/{hdate}_L{lead:03d}.pkl"
    with open(p + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(p + ".tmp", p)
    open(p + ".done", "w").write(f"{time.time() - t0:.1f}\n")
    print(f"CASE_DONE {hdate} L{lead:03d} {time.time() - t0:.0f}s", flush=True)
    return p


def all_cases():
    cases = []
    for L in LEADS:
        for f in sorted(glob.glob(f"{X}/pred/{ROW}_L{L:03d}/predictions_*_step{L:03d}.nc")):
            cases.append((os.path.basename(f).split("_")[1], L))
    return cases


_ST = _M = None
def _worker(c):
    global _ST, _M
    if _ST is None:
        _ST, _M = Stat(), load_matrix()
    try:
        return do_case(c[0], c[1], _ST, _M)
    except Exception as e:  # noqa
        import traceback
        print(f"CASE_FAIL {c[0]} L{c[1]:03d} {e!r}\n{traceback.format_exc()}", flush=True)
        return None


def run(workers):
    from multiprocessing import Pool
    todo = [c for c in all_cases() if not os.path.exists(f"{OUT}/parts/{c[0]}_L{c[1]:03d}.pkl.done")]
    print(f"RUN todo={len(todo)} of {len(all_cases())} workers={workers}", flush=True)
    with Pool(workers) as p:
        r = list(p.imap_unordered(_worker, todo))
    nf = sum(x is None for x in r)
    print(f"RUN_{'DONE' if nf == 0 else 'FAIL'} failed={nf}", flush=True)


# ----------------------------------------------------------------------------------------- aggregate
def aggregate():
    import pandas as pd
    parts = sorted(glob.glob(OUT + "/parts/*.pkl"))
    full = OUT + "/full"; tabs = OUT + "/tables"
    os.makedirs(full, exist_ok=True); os.makedirs(tabs, exist_ok=True)
    st = Stat()
    names = {k: v[1] for k, v in st.schemes.items()}
    names["lst"] = st.lst_scheme(0)[1]
    R1, R2, R3, R4, R4s, R5, R6, R7, R8 = ([] for _ in range(9))
    for p in parts:
        res = pickle.load(open(p, "rb"))
        h, L, se = res["hdate"], res["lead"], res["season"]
        for sname, dct in res["t1"].items():
            for src, acc in dct.items():
                vs = CV if src == "ctrl_pair" else V3
                for vi, v in enumerate(vs):
                    a = acc[vi] if src == "ctrl_pair" else acc[vi].sum(axis=0)  # pool members
                    for b, bn in enumerate(names[sname]):
                        n, s, ss = a[b]
                        if n > 0:
                            R1.append((h, L, se, src, v, sname, bn, int(n), s / n, np.sqrt(ss / n)))
                    if src == "floor":  # single-pair truth differences pf j - pf j+5 (telescoping sums of the cyclic pairs)
                        for j in range(5):
                            aj = acc[vi, j:j + 5].sum(axis=0)
                            for b, bn in enumerate(names[sname]):
                                n5, s5, _ = aj[b]
                                if n5 > 0:
                                    R1.append((h, L, se, f"floorpair{j}", v, sname, bn, int(n5 / 5), s5 / (n5 / 5), np.nan))
        R2 += [(h, L, se, *r) for r in res["t2"]]
        R3 += [(h, L, se, *r) for r in res["t3"]]
        R4 += [(h, L, se, *r) for r in res["t4"]]
        R4s += [(h, L, se, *r) for r in res["t4s"]]
        for sn, dct in res["t5"].items():
            for src, acc in dct.items():
                a = acc.sum(axis=0) if acc.ndim == 3 else acc
                for b, bn in enumerate(D3B_L):
                    n, s, ss = a[b]
                    if n > 0:
                        R5.append((h, L, se, sn, src, bn, int(n), s / n, np.sqrt(ss / n)))
        R6 += [(h, L, se, *r) for r in res["t6"]]
        R7 += [(h, L, se, *r) for r in res["t7"]]
        for src, acc in res["t8"].items():
            for vi, v in enumerate(V3):
                a = acc[vi]
                for b, bn in enumerate(names["zhr_band"]):
                    n, s, _ = a[b]
                    if n > 0:
                        R8.append((h, L, se, src, v, bn, int(n), np.sqrt(s / n)))
    c0 = ["date", "lead", "season"]
    t1 = pd.DataFrame(R1, columns=c0 + ["source", "var", "scheme", "bin", "n", "bias", "rmse"])
    t1["std"] = np.sqrt(np.maximum(t1.rmse ** 2 - t1.bias ** 2, 0))
    t2 = pd.DataFrame(R2, columns=c0 + ["source", "region", "var", "member", "slope", "r2", "n"])
    t3 = pd.DataFrame(R3, columns=c0 + ["source", "member", "metric", "value"])
    t4 = pd.DataFrame(R4, columns=c0 + ["source", "member", "surface", "metric", "value"])
    t4s = pd.DataFrame(R4s, columns=c0 + ["source", "member", "lst_bin", "n", "mean", "p1", "p99"])
    t5 = pd.DataFrame(R5, columns=c0 + ["subset", "quantity", "d320_km", "n", "mean", "rms"])
    t6 = pd.DataFrame(R6, columns=c0 + ["source", "region", "field", "member", "n"] + QS_L)
    t7 = pd.DataFrame(R7, columns=c0 + ["source", "var", "subset", "member", "lowpass_rms", "lowpass_mean"])
    t8 = pd.DataFrame(R8, columns=c0 + ["source", "var", "zhr_band", "n", "spread"])
    for nm, df in (("t1", t1), ("t2", t2), ("t3", t3), ("t4", t4), ("t4s", t4s), ("t5", t5), ("t6", t6), ("t7", t7), ("t8", t8)):
        df.to_csv(f"{full}/{nm}_per_date.csv.gz", index=False, float_format="%.6g")
    nd = t1.date.nunique()
    print(f"AGG cases={len(parts)} dates={nd}", flush=True)

    # T1 aggregated: per lead (and all leads) per source/var/scheme/bin over dates
    def dagg(df, keys, val):
        g = df.groupby(keys)[val]
        o = g.agg(["mean", "std", "count"]).rename(columns={"mean": f"{val}_mean", "std": f"{val}_sd_dates", "count": "n_dates"})
        o[f"{val}_npos"] = g.apply(lambda x: int((x > 0).sum())).values
        return o
    t1a = t1.copy()
    t1all = t1a.groupby(["date", "source", "var", "scheme", "bin"]).apply(
        lambda d: pd.Series({"n": d.n.sum(), "bias": np.average(d.bias, weights=d.n),
                             "rmse": np.sqrt(np.average(d.rmse ** 2, weights=d.n))}), include_groups=False).reset_index()
    t1all["lead"] = 0
    outs = []
    for df in (t1a, t1all):
        k = ["source", "var", "lead", "scheme", "bin"]
        b = dagg(df, k, "bias"); r = df.groupby(k).agg(rmse_mean=("rmse", "mean"), rmse_sd_dates=("rmse", "std"), n_points_per_date=("n", "mean"))
        outs.append(b.join(r).reset_index())
    T1 = pd.concat(outs)
    # artefact test vs floor (date as unit): bias exceeds 2x floor date-sd AND >=20 of 30 dates share its sign;
    # bias floor: sd across dates of a single truth-pair difference (pf j - pf j+5), mean over the 5 disjoint pairs, / sqrt(10)
    # = the date-to-date sd expected for a 10-member mean minus an independent 10-member mean of exchangeable members.
    fp = T1[T1.source.str.startswith("floorpair")].groupby(["var", "lead", "scheme", "bin"]).bias_sd_dates.mean() / np.sqrt(10)
    T1 = T1[~T1.source.str.startswith("floorpair")]
    fl = T1[T1.source == "floor"].set_index(["var", "lead", "scheme", "bin"])
    T1 = T1.set_index(["var", "lead", "scheme", "bin"])
    T1["floor_bias_sd_dates"] = fp.reindex(T1.index).values
    T1["floor_rmse_mean"] = fl["rmse_mean"].reindex(T1.index).values
    T1 = T1.reset_index()
    nsame = np.maximum(T1.bias_npos, T1.n_dates - T1.bias_npos)
    T1["bias_flag"] = (np.abs(T1.bias_mean) > 2 * T1.floor_bias_sd_dates) & (nsame >= np.ceil(20 / 30 * T1.n_dates))
    # rmse: dates where model rmse > floor rmse
    m_ = t1[t1.source.isin(["model", "input"])].merge(t1[t1.source == "floor"][["date", "lead", "var", "scheme", "bin", "rmse"]],
                                                       on=["date", "lead", "var", "scheme", "bin"], suffixes=("", "_floor"))
    m_["gt"] = (m_.rmse > m_.rmse_floor).astype(int)
    rg = m_.groupby(["source", "var", "lead", "scheme", "bin"]).gt.sum().rename("rmse_dates_above_floor").reset_index()
    T1 = T1.merge(rg, on=["source", "var", "lead", "scheme", "bin"], how="left")
    T1["bias_flag"] = T1["bias_flag"].astype(object)
    T1.loc[T1.source == "ctrl_pair", ["floor_bias_sd_dates", "floor_rmse_mean", "bias_flag"]] = np.nan
    T1.loc[T1.source == "floor", "bias_flag"] = np.nan
    T1.to_csv(f"{tabs}/T1_bias_rmse_binned.csv", index=False, float_format="%.5g")
    # season table: scheme all, all leads pooled, per season
    s = t1[t1.scheme.isin(["all", "zhr_band"])].copy()
    sa = s.groupby(["season", "source", "var", "scheme", "bin", "date"]).apply(
        lambda d: pd.Series({"bias": np.average(d.bias, weights=d.n), "rmse": np.sqrt(np.average(d.rmse ** 2, weights=d.n)), "n": d.n.sum()}),
        include_groups=False).reset_index()
    sg = sa.groupby(["season", "source", "var", "scheme", "bin"]).agg(bias_mean=("bias", "mean"), bias_sd_dates=("bias", "std"),
                                                                     rmse_mean=("rmse", "mean"), n_dates=("date", "nunique"),
                                                                     n_points_per_date=("n", "mean")).reset_index()
    sg.to_csv(f"{tabs}/T1e_season.csv", index=False, float_format="%.5g")

    # T2: member-mean model slope per case-region-var, truth ctrl, truth pf mean
    g = t2.groupby(["date", "lead", "season", "region", "var", "source"]).agg(slope=("slope", "mean"), slope_sd_members=("slope", "std"),
                                                                             r2=("r2", "mean"), n=("n", "first")).reset_index()
    T2 = g.pivot_table(index=["date", "lead", "season", "region", "var", "n"], columns="source",
                       values=["slope", "slope_sd_members", "r2"]).reset_index()
    T2.columns = ["_".join([c for c in col if c]).strip("_") for col in T2.columns.values]
    T2.to_csv(f"{tabs}/T2_lapse_slopes.csv", index=False, float_format="%.4g")

    # T3: per date/lead/source/metric (member mean)
    T3 = t3.groupby(["date", "lead", "season", "source", "metric"]).value.mean().reset_index()
    T3.to_csv(f"{tabs}/T3_hydrostatic.csv", index=False, float_format="%.5g")
    T3s = T3.groupby(["lead", "source", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    T3s.to_csv(f"{tabs}/T3_hydrostatic_summary.csv", index=False, float_format="%.5g")
    # T4
    T4 = t4.groupby(["lead", "source", "surface", "metric"]).value.agg(["mean", "std", "max", "count"]).reset_index()
    T4.to_csv(f"{tabs}/T4_2d_vs_2t.csv", index=False, float_format="%.5g")
    T4s = t4s.groupby(["source", "lst_bin"]).agg(n_cases=("date", "count"), mean=("mean", "mean"), p1=("p1", "mean"), p99=("p99", "mean")).reset_index()
    T4s.to_csv(f"{tabs}/T4_skt_minus_2t.csv", index=False, float_format="%.5g")
    # T5: per lead and all leads, mean over dates
    T5 = t5.groupby(["lead", "subset", "quantity", "d320_km"]).agg(mean=("mean", "mean"), mean_sd_dates=("mean", "std"), rms=("rms", "mean"),
                                                                  rms_sd_dates=("rms", "std"), n_dates=("date", "nunique"), n_points=("n", "mean")).reset_index()
    T5b = t5.groupby(["subset", "quantity", "d320_km"]).agg(mean=("mean", "mean"), mean_sd_dates=("mean", "std"), rms=("rms", "mean"),
                                                           rms_sd_dates=("rms", "std"), n_dates=("date", "nunique"), n_points=("n", "mean")).reset_index()
    T5b["lead"] = 0
    pd.concat([T5, T5b]).to_csv(f"{tabs}/T5_o320_imprint.csv", index=False, float_format="%.5g")
    # T6: member mean per case
    T6 = t6.groupby(["date", "lead", "season", "source", "region", "field", "n"])[QS_L].mean().reset_index()
    T6.to_csv(f"{tabs}/T6_tails.csv", index=False, float_format="%.5g")
    # T7
    T7 = t7.groupby(["lead", "source", "var", "subset"]).agg(lowpass_rms=("lowpass_rms", "mean"), lowpass_rms_sd_dates=("lowpass_rms", "std"),
                                                             n_dates=("date", "nunique")).reset_index()
    T7.to_csv(f"{tabs}/T7_large_scale.csv", index=False, float_format="%.5g")
    # T8
    T8 = t8.groupby(["lead", "source", "var", "zhr_band"]).agg(spread=("spread", "mean"), spread_sd_dates=("spread", "std"),
                                                               n_dates=("date", "nunique"), n_points=("n", "mean")).reset_index()
    T8.to_csv(f"{tabs}/T8_spread.csv", index=False, float_format="%.5g")
    for f in sorted(glob.glob(tabs + "/*.csv")):
        print("TABLE", os.path.basename(f), os.path.getsize(f), flush=True)
    print("AGG_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "statics":
        build_statics()
    elif a.cmd == "case":
        do_case(a.args[0], int(a.args[1]))
    elif a.cmd == "run":
        run(a.workers)
    elif a.cmd == "aggregate":
        aggregate()
