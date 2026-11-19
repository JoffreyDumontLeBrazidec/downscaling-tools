"""A small synthetic copy of the Jupiter inputs of the final-model battery, for the smoke test.

Same layout, names and array shapes as the real stores (E2 box 10-70 N, 80 W-40 E), on coarse grids: the "O320" input
grid is 1 degree, the "O1280" output grid 0.25 degree. Fields are smooth weather plus terrain plus member noise, with
two lows so the storm code has something to find. Nothing here is a physical claim; it only exercises the code paths.

    build(root, dates=("20230226", "20230829"), leads=(12, 24)) -> dict of paths
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

G, RD = 9.80665, 287.05
W10 = ["10u", "10v", "2t", "msl", "sp", "tp", "cp", "2d", "skt", "tcw"]
PF_OUT = ["10u", "10v", "2t", "msl", "sp", "tp", "cp"]
PF_IN = ["10u", "10v", "2t", "msl", "sp", "tp", "cp", "2d", "skt", "tcw"]
V4_OUT = ["10u", "10v", "2t", "msl", "sp", "tp", "cp", "2d", "skt", "tcw"]
V4_IN = ["10u", "10v", "2t", "msl", "sp", "tp", "cp", "2d", "skt", "tcw", "t_850", "t_700", "z_850", "z_700"]
FORC = ["cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude", "insolation", "lsm", "slor", "sdor", "cos_jd",
        "sin_jd", "z"]
NAME_MAT = "interpol_O320_to_O1280_linear-europe_e2_10_70N_80W_40E_halo1p5.mat.npz"
NAME_G320 = "grid-o320-europe_e2_10_70N_80W_40E_halo1p5.npz"


def grids():
    la_c, lo_c = np.arange(8.5, 71.5 + 1e-9, 1.0), np.arange(-81.5, 41.5 + 1e-9, 1.0)
    la_f, lo_f = np.arange(10.0, 70.0 + 1e-9, 0.25), np.arange(-80.0, 40.0 + 1e-9, 0.25)
    LC, OC = np.meshgrid(la_c, lo_c, indexing="ij")
    LF, OF = np.meshgrid(la_f, lo_f, indexing="ij")
    return (la_c, lo_c, LC.ravel(), OC.ravel()), (la_f, lo_f, LF.ravel(), OF.ravel())


def bilinear(coarse, fine):
    import scipy.sparse as sp
    la_c, lo_c = coarse[0], coarse[1]
    lat, lon = fine[2], fine[3]
    fy = (lat - la_c[0]) / 1.0
    fx = (lon - lo_c[0]) / 1.0
    iy, ix = np.floor(fy).astype(int), np.floor(fx).astype(int)
    wy, wx = fy - iy, fx - ix
    nx = len(lo_c)
    rows, cols, vals = [], [], []
    n = lat.size
    for dy, dx, w in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx), (1, 0, wy * (1 - wx)), (1, 1, wy * wx)):
        rows.append(np.arange(n)); cols.append((iy + dy) * nx + ix + dx); vals.append(w)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, len(la_c) * nx))


def terrain(lat, lon):
    def bump(la0, lo0, h, r):
        return h * np.exp(-(((lat - la0) / r) ** 2 + ((lon - lo0) / (1.4 * r)) ** 2))
    z = bump(46.5, 10.0, 2600, 1.6) + bump(64.0, -42.0, 3200, 3.0) + bump(61.0, 9.0, 1400, 2.5) + bump(40.0, -79.0, 900, 2.0) \
        + bump(64.8, -18.0, 1200, 1.0) + bump(31.5, -6.0, 2200, 1.5)
    land = (((lon > -10) & (lat > 36)) | ((lon < -55) & (lat > 25)) | ((lon > -55) & (lon < -25) & (lat > 59)) | ((lat < 36) & (lon > -15) & (lat > 12))
            | ((lat > 63.3) & (lat < 66.5) & (lon > -24.5) & (lon < -13.5)))
    lake = ((lat - 46.4) ** 2 + (lon - 6.5) ** 2) < 0.35 ** 2
    land = land & ~lake
    z = np.where(land, z + 50, 0.0)
    return z, land.astype(np.float32)


def weather(lat, lon, z, seed, low_centres):
    r = np.random.default_rng(seed)
    t2 = 300 - 0.55 * (lat - 10) - 0.0065 * z + 2 * np.sin(np.radians(lon * 3 + seed))
    msl = 101500 + 600 * np.sin(np.radians(lat * 4 + seed)) * np.cos(np.radians(lon * 2))
    u = np.zeros_like(lat); v = np.zeros_like(lat)
    for la0, lo0, depth, rad in low_centres:
        dy = (lat - la0) * 111.2; dx = (lon - lo0) * 111.2 * np.cos(np.radians(la0))
        rr = np.hypot(dx, dy) + 1e-3
        msl = msl - depth * np.exp(-(rr / rad) ** 2)
        vt = 25 * (rr / rad) * np.exp(0.5 * (1 - (rr / rad) ** 2))
        u += -vt * dy / rr; v += vt * dx / rr
    u += 5 + r.normal(0, 0.3, lat.size); v += r.normal(0, 0.3, lat.size)
    sp = msl * np.exp(-G * z / (RD * (t2 + 0.00325 * z)))
    tp = np.clip(0.004 * np.sin(np.radians(lat * 9 + lon * 5 + seed)) + r.normal(0, 0.0008, lat.size), 0, None)
    cp = 0.4 * tp
    d2 = t2 - np.abs(3 + 2 * np.cos(np.radians(lon + seed)))
    skt = t2 + 0.5
    tcw = 25 + 10 * np.cos(np.radians(lat * 3))
    return {"10u": u, "10v": v, "2t": t2, "msl": msl, "sp": sp, "tp": tp, "cp": cp, "2d": d2, "skt": skt, "tcw": tcw,
            "t_850": t2 - 10, "t_700": t2 - 20, "z_850": np.full_like(lat, 1450 * G), "z_700": np.full_like(lat, 3000 * G)}


def _arr(g, name, data, **kw):
    a = g.create_array(name, shape=data.shape, dtype=data.dtype, **kw)
    a[...] = data
    return a


def build(root, dates=("20230226", "20230829"), leads=(12, 24), row="SYN_m", nmem=10):
    import netCDF4
    import zarr

    root = Path(root)
    U = root / "user"
    coarse, fine = grids()
    latc, lonc, latf, lonf = coarse[2], coarse[3], fine[2], fine[3]
    M = bilinear(coarse, fine)
    zc, lsmc = terrain(latc, lonc)
    zf, lsmf = terrain(latf, lonf)
    D = U / "datasets"
    assets = D / "o320_o1280_europe_e2" / "assets"
    (assets / "inter_mat").mkdir(parents=True, exist_ok=True)
    (assets / "grids").mkdir(parents=True, exist_ok=True)
    np.savez(assets / "inter_mat" / NAME_MAT, data=M.data, indices=M.indices, indptr=M.indptr, shape=np.array(M.shape))
    np.savez(assets / "grids" / NAME_G320, latitudes=latc, longitudes=lonc)
    # forcings (O1280 z as geopotential, lsm)
    fz = zarr.open_group(str(D / "o320_o1280_europe_e2" / "in_hres_o1280_forcings.zarr"), mode="w")
    fz.attrs["variables"] = FORC
    fd = np.zeros((1, len(FORC), 1, latf.size), np.float32)
    fd[0, FORC.index("lsm"), 0] = lsmf; fd[0, FORC.index("z"), 0] = zf * G
    _arr(fz, "data", fd); _arr(fz, "latitudes", latf); _arr(fz, "longitudes", np.where(lonf < 0, lonf + 360, lonf))
    # O320 orography grib on the coarse grid (regular_ll, north to south as GRIB scans)
    import eccodes
    (U / "station_caches").mkdir(parents=True, exist_ok=True)
    with open(U / "station_caches" / "orography_O320_eefh.grib", "wb") as fh:
        h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        la_c, lo_c = coarse[0], coarse[1]
        for k, v in (("Ni", len(lo_c)), ("Nj", len(la_c)), ("latitudeOfFirstGridPointInDegrees", float(la_c[-1])),
                     ("longitudeOfFirstGridPointInDegrees", float(lo_c[0] % 360)), ("latitudeOfLastGridPointInDegrees", float(la_c[0])),
                     ("longitudeOfLastGridPointInDegrees", float(lo_c[-1] % 360)), ("iDirectionIncrementInDegrees", 1.0),
                     ("jDirectionIncrementInDegrees", 1.0), ("jScansPositively", 0)):
            eccodes.codes_set(h, k, v)
        eccodes.codes_set_values(h, (zc.reshape(len(la_c), len(lo_c))[::-1] * G).ravel())
        eccodes.codes_write(h, fh)
        eccodes.codes_release(h)
    # stores
    A = U / "agent-work" / "ens-member-box"
    cases = [(d, L) for d in dates for L in leads]
    v4_out, v4_in = [], []
    pf = {"stores": [], "stores_leads": []}
    pred_root = U / "outputs" / "box"
    for e2, (date, lead) in enumerate(cases):
        seed = int(date) % 9973 + lead
        lows = [(55.0 + lead / 24, -30.0, 2500.0, 500.0), (25.0, -60.0 + lead / 12, 3000.0, 250.0)]
        base_c = weather(latc, lonc, zc, seed, lows)
        base_f = weather(latf, lonf, zf, seed, lows)
        v4_out.append(np.stack([base_f[v] for v in V4_OUT]))
        v4_in.append(np.stack([base_c[v] for v in V4_IN]))
        tru, inp_c, inp_f, mod = [], [], [], []
        for k in range(nmem):
            shift = [(la + 0.4 * np.sin(k), lo + 0.5 * np.cos(k), d, r) for la, lo, d, r in lows]
            tk = weather(latf, lonf, zf, seed + 101 * (k + 1), shift)
            ik = weather(latc, lonc, zc, seed + 211 * (k + 1), shift)
            tru.append(np.stack([tk[v] for v in PF_OUT]))
            inp_c.append(np.stack([ik[v] for v in PF_IN]))
            xi = np.stack([M @ ik[v] for v in W10])
            rng = np.random.default_rng(seed + k)
            yp = xi + rng.normal(0, 1, xi.shape) * np.array([0.5, 0.5, 0.3, 30, 30, 0.0002, 0.0001, 0.3, 0.3, 0.5])[:, None]
            mod.append(yp); inp_f.append(xi)
        sub = "stores" if lead in (24, 48) else "stores_leads"
        valid = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00", "s") + np.timedelta64(lead, "h")
        pf[sub].append(dict(hdate=date, lead=lead, e2=e2, valid=valid, out=np.stack(tru, 1), inp=np.stack(inp_c, 1)))
        # prediction file
        pdir = pred_root / "pred" / f"{row}_L{lead:03d}"
        pdir.mkdir(parents=True, exist_ok=True)
        with netCDF4.Dataset(pdir / f"predictions_{date}_step{lead:03d}.nc", "w") as d:
            d.createDimension("sample", 1); d.createDimension("ensemble_member", nmem)
            d.createDimension("grid_point_hres", latf.size); d.createDimension("weather_state", len(W10))
            ws = d.createVariable("weather_state", str, ("weather_state",))
            for i, n in enumerate(W10):
                ws[i] = n
            d.createVariable("lat_hres", "f4", ("grid_point_hres",))[:] = latf
            d.createVariable("lon_hres", "f4", ("grid_point_hres",))[:] = np.where(lonf < 0, lonf + 360, lonf)
            d.createVariable("y_pred", "f4", ("sample", "ensemble_member", "grid_point_hres", "weather_state"))[:] = \
                np.moveaxis(np.stack(mod), 1, 2)[None]
            d.createVariable("x_interp_members", "f4", ("sample", "ensemble_member", "grid_point_hres", "weather_state"))[:] = \
                np.moveaxis(np.stack(inp_f), 1, 2)[None]
            d.createVariable("y", "f4", ("sample", "grid_point_hres", "weather_state"))[:] = np.stack([base_f[v] for v in W10], -1)[None]
            d.createVariable("x_interp", "f4", ("sample", "grid_point_hres", "weather_state"))[:] = \
                np.stack([M @ base_c[v] for v in W10], -1)[None]
            d.setncattr("fake_date", str(valid)); d.setncattr("idx", e2); d.setncattr("seeds", json.dumps([seed]))
    for sub, rows in pf.items():
        if not rows:
            continue
        for kind, names, grid in (("out", PF_OUT, (latf, lonf)), ("inp", PF_IN, (latc, lonc))):
            fn = "emb_out_hres_o1280_pf.zarr" if kind == "out" else "emb_in_lres_o320_halo1p5_pf.zarr"
            g = zarr.open_group(str(A / sub / fn), mode="w")
            g.attrs["variables"] = names
            g.attrs["row_map"] = [{"hdate": r["hdate"], "lead": r["lead"]} for r in rows]
            g.attrs["leads"] = sorted({r["lead"] for r in rows})
            _arr(g, "data", np.stack([r[kind] for r in rows]).astype(np.float32))
            _arr(g, "e2_row", np.array([r["e2"] for r in rows], np.int64))
            _arr(g, "dates", np.array([r["valid"].astype("int64") for r in rows], np.int64))
            _arr(g, "latitudes", grid[0]); _arr(g, "longitudes", grid[1])
    for fn, names, data in (("out_hres_o1280.zarr", V4_OUT, v4_out), ("in_lres_o320_halo1p5.zarr", V4_IN, v4_in)):
        g = zarr.open_group(str(D / "o320_o1280_europe_e2_v4" / fn), mode="w")
        g.attrs["variables"] = names
        _arr(g, "data", np.stack(data)[:, :, None, :].astype(np.float32))
    return {"user_root": str(U), "member_stores": str(A), "pred_root": str(pred_root), "row": row}
