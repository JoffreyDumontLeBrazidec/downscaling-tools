#!/usr/bin/env python3
"""Realism accumulators for 10 m wind and cyclones, GL512 450k box row (thread "Hunt wind and cyclone artefacts at 9 km", 17 Nov 2026).

Input: reduced field files <fields>/GLB450e2_m_<date>_L<LLL>.npz (keys lat, lon, names=[10u,10v,msl],
pred[10,P,3], inp[10,P,3], truth[10,P,3], y[P,3]) written by extract.py, and orog.npz (lat, lon, orog_m, lsm).
Output in <out>: tiles_total.npz (per-pixel sums over all files and members), tile_dates.csv (per tile, date, source: band
variances), spectra.npz (per tile and source isotropic power), cases.csv + profiles.npz (storm-centred), crops/*.npz.
Sources: truth = 9 km enfh perturbed members 1-10, model = GLB450e2_m draw k from input member k, input = interpolated input member k.
No file is modified. numpy + scipy only.
"""
import argparse, glob, os, re, sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

R_E = 6371.0
KM_DEG = np.pi * R_E / 180.0
PIX = 8.0            # km
MARGIN = 160.0       # km taper margin around each 5-degree tile
BANDS = [(17.0, 28.0), (28.0, 45.0), (45.0, 90.0), (90.0, 200.0)]   # wavelength km
SPEC_EDGES = np.geomspace(16.0, 400.0, 25)   # wavelength bin edges km
SRC = ["truth", "model", "input"]
BOX = (10.0, 70.0, -80.0, 40.0)
SPECKLE_THR = 2.0    # m/s, |ws - median of 3x3| above this = speckle pixel

CROPS = [  # name, date, lead, lat0, lat1, lon0, lon1
    ("alps", "20230226", 24, 40.0, 52.0, 1.4, 18.6),
    ("ciaran", "20231031", 48, 43.0, 55.0, -14.0, 6.0),
    ("norway", "20230111", 24, 56.0, 68.0, 0.0, 18.0),
    ("ocean_quiet", "20230829", 24, 36.0, 46.0, -45.0, -25.0),
    ("iberia", "20230829", 24, 35.0, 45.0, -11.0, 5.0),
]


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class Tile:
    def __init__(self, lat0, lon0, tree):
        self.lat0, self.lon0 = lat0, lon0
        hy = 2.5 * KM_DEG + MARGIN
        hx = 2.5 * KM_DEG * np.cos(np.radians(max(abs(lat0) - 2.5, 0))) + MARGIN
        ny, nx = int(2 * hy / PIX) // 2 * 2, int(2 * hx / PIX) // 2 * 2
        y = (np.arange(ny) - ny / 2 + 0.5) * PIX
        x = (np.arange(nx) - nx / 2 + 0.5) * PIX
        X, Y = np.meshgrid(x, y)
        lat = lat0 + Y / KM_DEG
        lon = lon0 + X / (KM_DEG * np.cos(np.radians(lat)))
        d, idx = tree.query(xyz(lat.ravel(), lon.ravel()))
        self.idx = idx.reshape(ny, nx)
        self.lat, self.lon = lat, lon
        self.far = (d.reshape(ny, nx) * R_E) > 12.0
        self.core = (np.abs(lat - lat0) <= 2.5) & (np.abs(lon - lon0) <= 2.5)
        # taper only inside the margin: distance to tile edge in km
        dy = np.minimum(np.arange(ny), np.arange(ny)[::-1]) * PIX
        dx = np.minimum(np.arange(nx), np.arange(nx)[::-1]) * PIX
        ty = np.clip(dy / (0.8 * MARGIN), 0, 1); tx = np.clip(dx / (0.8 * MARGIN), 0, 1)
        self.win = np.outer(0.5 - 0.5 * np.cos(np.pi * ty), 0.5 - 0.5 * np.cos(np.pi * tx)).astype(np.float32)
        ky = np.fft.fftfreq(ny, d=PIX); kx = np.fft.rfftfreq(nx, d=PIX)
        self.KX, self.KY = np.meshgrid(kx, ky)
        k = np.hypot(self.KX, self.KY)
        with np.errstate(divide="ignore"):
            wl = np.where(k > 0, 1.0 / k, np.inf)
        self.bandmask = [((wl >= a) & (wl < b)) for a, b in BANDS]
        self.specbin = np.digitize(wl, SPEC_EDGES) - 1          # -1 or 24 = outside
        # rfft weights (double count of non-self-conjugate columns)
        w = np.full(k.shape, 2.0); w[:, 0] = 1.0
        if nx % 2 == 0: w[:, -1] = 1.0
        self.w = w / (ny * nx) ** 2
        self.ny, self.nx = ny, nx
        self.ncore = int(self.core.sum())

    def spec(self, P):
        """binned power of a rfft2 power array P (already |F|^2)."""
        sb = self.specbin.ravel(); ok = (sb >= 0) & (sb < len(SPEC_EDGES) - 1)
        return np.bincount(sb[ok], weights=(P * self.w).ravel()[ok], minlength=len(SPEC_EDGES) - 1)


def detrend(f):
    ny, nx = f.shape
    y = np.linspace(-1, 1, ny)[:, None]; x = np.linspace(-1, 1, nx)[None, :]
    f = f - f.mean()
    f = f - y * (f * y).sum() / (y * y).sum() / nx * 1.0 - x * (f * x).sum() / (x * x).sum() / ny * 1.0
    return f


def process_tiles(tiles, data, acc, rows, spectra, date, lead):
    """data: dict src -> array [M, P, 3]"""
    nb = len(BANDS)
    for ti, T in enumerate(tiles):
        for si, s in enumerate(SRC):
            A = data[s]
            M = A.shape[0]
            bandvar = np.zeros(nb); zvar = np.zeros(nb); dvar = np.zeros(nb); kevar = np.zeros(nb); mslvar = np.zeros(nb)
            sp = np.zeros((5, len(SPEC_EDGES) - 1))
            sws = 0.0; n_speck = 0
            for m in range(M):
                u = A[m, :, 0][T.idx]; v = A[m, :, 1][T.idx]; p = A[m, :, 2][T.idx]
                ws = np.hypot(u, v)
                acc["sum_ws"][si][ti] += ws[T.core]
                acc["sum_ws2"][si][ti] += ws[T.core] ** 2
                med = ndimage.median_filter(ws, size=3, mode="nearest")
                sk = (np.abs(ws - med) > SPECKLE_THR)[T.core]
                acc["speckle"][si][ti] += sk
                n_speck += int(sk.sum())
                sws += float(ws[T.core].mean())
                F = np.fft.rfft2(detrend(ws) * T.win)
                Fu = np.fft.rfft2(detrend(u) * T.win); Fv = np.fft.rfft2(detrend(v) * T.win)
                Fp = np.fft.rfft2(detrend(p) * T.win)
                Pz = np.abs(T.KX * Fv - T.KY * Fu) ** 2 * (2 * np.pi) ** 2      # (m/s per km)^2
                Pd = np.abs(T.KX * Fu + T.KY * Fv) ** 2 * (2 * np.pi) ** 2
                Pk = np.abs(Fu) ** 2 + np.abs(Fv) ** 2
                Pw = np.abs(F) ** 2; Pp = np.abs(Fp) ** 2
                for j, P in enumerate((Pw, Pk, Pz, Pd, Pp)):
                    sp[j] += T.spec(P)
                for b in range(nb):
                    bm = T.bandmask[b]
                    fb = np.fft.irfft2(np.where(bm, F, 0), s=(T.ny, T.nx))
                    b2 = fb[T.core] ** 2
                    acc["band2"][si][b][ti] += b2
                    bandvar[b] += float(b2.mean())
                    wsum = T.w * bm
                    zvar[b] += float((Pz * wsum).sum()); dvar[b] += float((Pd * wsum).sum())
                    kevar[b] += float((Pk * wsum).sum()); mslvar[b] += float((Pp * wsum).sum())
            acc["n"][si][ti] += M
            spectra[si][ti] += sp; 
            r = dict(date=date, lead=lead, tile=ti, lat0=T.lat0, lon0=T.lon0, src=s, members=M, mean_ws=sws / M, speckle_px=n_speck / M)
            for b in range(nb):
                tag = "%d_%d" % BANDS[b]
                r["ws_var_" + tag] = bandvar[b] / M; r["vort_pow_" + tag] = zvar[b] / M; r["div_pow_" + tag] = dvar[b] / M
                r["ke_pow_" + tag] = kevar[b] / M; r["msl_pow_" + tag] = mslvar[b] / M
            rows.append(r)


# ---------------------------------------------------------------- storms
def find_lows(lat, lon, msl, lsm, tree):
    """candidate lows in the control field: local minima on a 0.25-degree min-binned grid."""
    la0, la1, lo0, lo1 = BOX
    d = 0.25
    ny, nx = int((la1 - la0) / d), int((lo1 - lo0) / d)
    iy = np.clip(((lat - la0) / d).astype(int), 0, ny - 1); ix = np.clip(((lon - lo0) / d).astype(int), 0, nx - 1)
    g = np.full((ny, nx), np.inf); np.minimum.at(g, (iy, ix), msl)
    g[~np.isfinite(g)] = 1.1e5
    mn = ndimage.minimum_filter(g, size=(33, 49), mode="nearest")   # about +-4 degrees
    cand = np.argwhere((g == mn) & (g < 1.008e5))
    out = []
    for cy, cx in cand:
        clat, clon = la0 + (cy + 0.5) * d, lo0 + (cx + 0.5) * d
        if not (la0 + 5 <= clat <= la1 - 5 and lo0 + 6 <= clon <= lo1 - 6):
            continue
        ii = np.array(tree.query_ball_point(xyz(clat, clon), 500.0 / R_E))
        dist = gc_km(clat, clon, lat[ii], lon[ii])
        k = ii[np.argmin(msl[ii])]
        ring = msl[ii][(dist > 400)]
        depth = (ring.mean() - msl[k]) / 100.0
        seafrac = float((lsm[ii][dist < 150] < 0.5).mean())
        if depth < 4.0 or seafrac < 0.6:
            continue
        out.append(dict(lat=float(lat[k]), lon=float(lon[k]), pmin=float(msl[k]) / 100.0, depth=depth, seafrac=seafrac))
    out.sort(key=lambda c: c["pmin"])
    return out[:8]


def gc_km(lat0, lon0, lat, lon):
    a = xyz(lat0, lon0); b = xyz(lat, lon)
    return 2 * R_E * np.arcsin(np.clip(np.linalg.norm(b - a, axis=-1) / 2, 0, 1))


def centre_near(lat, lon, msl, tree, clat, clon, rad):
    ii = np.array(tree.query_ball_point(xyz(clat, clon), rad / R_E))
    k = ii[np.argmin(msl[ii])]
    onedge = gc_km(clat, clon, lat[k], lon[k]) > 0.97 * rad
    return float(lat[k]), float(lon[k]), bool(onedge)


RB = np.arange(0, 510, 10.0)


def structure(lat, lon, u, v, msl, tree, clat, clon):
    ii = np.array(tree.query_ball_point(xyz(clat, clon), 500.0 / R_E))
    la, lo = lat[ii], lon[ii]
    x = (lo - clon) * KM_DEG * np.cos(np.radians(clat)); y = (la - clat) * KM_DEG
    r = np.hypot(x, y); th = np.arctan2(y, x)
    uu, vv, pp = u[ii], v[ii], msl[ii]
    ws = np.hypot(uu, vv)
    vt = -uu * np.sin(th) + vv * np.cos(th); vr = uu * np.cos(th) + vv * np.sin(th)
    b = np.clip((r / 10.0).astype(int), 0, len(RB) - 2)
    cnt = np.bincount(b, minlength=len(RB) - 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = {k: np.bincount(b, weights=w, minlength=len(RB) - 1) / cnt for k, w in (("ws", ws), ("vt", vt), ("vr", vr), ("msl", pp))}
    wsp = np.where(cnt >= 3, prof["ws"], np.nan)
    jb = int(np.nanargmax(wsp[1:40]) + 1)
    rmw = RB[jb] + 5.0
    res = dict(pmin=float(pp.min()) / 100.0, vmax300=float(ws[r <= 300].max()), rmw=rmw, vazi_max=float(wsp[jb]),
               vt_max=float(np.nanmax(np.where(cnt >= 3, prof["vt"], np.nan)[1:40])))
    inner = r < max(0.3 * rmw, 12.0)
    res["eye_ratio"] = float(ws[inner].mean() / wsp[jb]) if inner.any() else np.nan
    res["ws_centre_min"] = float(ws[r < max(0.5 * rmw, 15.0)].min()) if (r < max(0.5 * rmw, 15.0)).any() else np.nan
    ann = (r >= 0.7 * rmw) & (r <= 1.3 * rmw)
    if ann.sum() >= 8:
        G = np.stack([np.ones(ann.sum()), np.cos(th[ann]), np.sin(th[ann])], 1)
        c = np.linalg.lstsq(G, ws[ann], rcond=None)[0]
        res["asym1"] = float(np.hypot(c[1], c[2]) / c[0]); res["asym_dir_deg"] = float(np.degrees(np.arctan2(c[2], c[1])))
        res["ring_rough"] = float(np.std(ws[ann] - G @ c) / c[0])
        res["ring_max_over_mean"] = float(ws[ann].max() / c[0])
    out_ann = (r >= rmw) & (r <= min(3 * rmw, 450.0))
    if out_ann.sum() >= 8:
        res["inflow_deg"] = float(np.degrees(np.arctan2(-vr[out_ann].mean(), vt[out_ann].mean())))
    # gradient wind from the azimuthal-mean pressure profile
    f = 2 * 7.292e-5 * np.sin(np.radians(clat))
    rc = (RB[:-1] + 5.0) * 1000.0
    pm = np.where(cnt >= 3, prof["msl"], np.nan)
    ok = np.isfinite(pm)
    if ok.sum() > 10:
        pmf = np.interp(rc, rc[ok], pm[ok]); pmf = ndimage.uniform_filter1d(pmf, 3, mode="nearest")
        dpdr = np.gradient(pmf, rc)
        vg = -f * rc / 2 + np.sqrt(np.maximum((f * rc / 2) ** 2 + rc * dpdr / 1.15, 0))
        sel = (rc / 1000.0 >= 1.5 * rmw) & (rc / 1000.0 <= min(4 * rmw, 450.0)) & (vg > 3)
        if sel.sum() >= 3:
            res["vt_over_vgrad"] = float(np.nanmean(prof["vt"][sel]) / np.nanmean(vg[sel]))
        prof["vgrad"] = vg
        res["dp_300"] = float((np.interp(300e3, rc, pmf) - pp.min()) / 100.0)
    return res, prof, (ii, r)


def process_storms(lat, lon, lsm, tree, D, date, lead, cases, profiles, cropdir):
    yc = D["y"]
    lows = find_lows(lat, lon, yc[:, 2], lsm, tree)
    for li, L in enumerate(lows):
        cid = "%s_L%03d_%d" % (date, lead, li)
        kind = "tc" if (L["lat"] < 36.0 and 6 <= int(date[4:6]) <= 11) else "xt"
        keep = None
        for si, s in enumerate(SRC):
            A = D[s]
            for m in range(A.shape[0]):
                u, v, p = A[m, :, 0], A[m, :, 1], A[m, :, 2]
                # every field around its own centre; the model centre is searched from its input member's centre
                ilat, ilon, iedge = centre_near(lat, lon, D["input"][m, :, 2], tree, L["lat"], L["lon"], 400.0)
                if s == "truth":
                    clat, clon, edge = centre_near(lat, lon, p, tree, L["lat"], L["lon"], 400.0)
                elif s == "input":
                    clat, clon, edge = ilat, ilon, iedge
                else:
                    clat, clon, edge = centre_near(lat, lon, p, tree, ilat, ilon, 250.0)
                res, prof, (ii, r) = structure(lat, lon, u, v, p, tree, clat, clon)
                res.update(case=cid, date=date, lead=lead, low=li, kind=kind, src=s, member=m + 1, clat=clat, clon=clon, on_edge=int(edge),
                           ctl_lat=L["lat"], ctl_lon=L["lon"], ctl_pmin=L["pmin"], ctl_depth=L["depth"], seafrac=L["seafrac"],
                           d_from_input_km=float(gc_km(clat, clon, ilat, ilon)), d_from_ctl_km=float(gc_km(clat, clon, L["lat"], L["lon"])))
                cases.append(res)
                for k in ("ws", "vt", "vr", "msl", "vgrad"):
                    if k in prof:
                        profiles.setdefault(k, []).append(prof[k].astype(np.float32))
                profiles.setdefault("key", []).append("%s|%s|%d" % (cid, s, m + 1))
        if kind == "tc" or L["pmin"] < 975.0:
            ii = np.array(tree.query_ball_point(xyz(L["lat"], L["lon"]), 700.0 / R_E))
            save_crop(os.path.join(cropdir, "storm_%s.npz" % cid), lat, lon, lsm, D, ii, meta=dict(ctl_lat=L["lat"], ctl_lon=L["lon"], kind=kind))


def save_crop(path, lat, lon, lsm, D, ii, meta=None, nm=3):
    out = dict(lat=lat[ii].astype(np.float32), lon=lon[ii].astype(np.float32), lsm=lsm[ii].astype(np.float16))
    for s in SRC:
        A = D[s][:nm][:, ii, :]
        out[s + "_uv"] = A[:, :, :2].astype(np.float16)
        out[s + "_msl"] = A[:, :, 2].astype(np.float32)
    out["ctl_uv"] = D["y"][ii, :2].astype(np.float16); out["ctl_msl"] = D["y"][ii, 2].astype(np.float32)
    if meta:
        out.update({k: np.array(v) for k, v in meta.items()})
    np.savez_compressed(path, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True); ap.add_argument("--orog", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "crops"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.fields, os.environ.get("FB_ROW", "GLB450e2_m") + "_*_L*.npz")))
    if a.only:
        files = [f for f in files if re.search(a.only, f)]
    O = np.load(a.orog)
    lat, lon = O["lat"].astype(np.float64), O["lon"].astype(np.float64)
    lon = np.where(lon > 180, lon - 360, lon)
    lsm = O["lsm"].astype(np.float32); orog = O["orog_m"].astype(np.float32)
    tree = cKDTree(xyz(lat, lon))
    t0 = time.time()
    tiles = [Tile(la, lo, tree) for la in np.arange(12.5, 70, 5.0) for lo in np.arange(-77.5, 40, 5.0)]
    print("tiles", len(tiles), "built in %.0f s" % (time.time() - t0), flush=True)
    nt = len(tiles); nb = len(BANDS)
    acc = dict(sum_ws=[[np.zeros(T.ncore) for T in tiles] for _ in SRC], sum_ws2=[[np.zeros(T.ncore) for T in tiles] for _ in SRC],
               speckle=[[np.zeros(T.ncore) for T in tiles] for _ in SRC],
               band2=[[[np.zeros(T.ncore) for T in tiles] for _ in range(nb)] for _ in SRC], n=[[0] * nt for _ in SRC])
    spectra = np.zeros((len(SRC), nt, 5, len(SPEC_EDGES) - 1))
    rows, cases, profiles = [], [], {}
    done = []
    for f in files:
        mm = re.search(r"_(\d{8})_L(\d{3})\.npz$", f); date, lead = mm.group(1), int(mm.group(2))
        t1 = time.time()
        Z = np.load(f)
        flon = np.where(Z["lon"] > 180, Z["lon"] - 360, Z["lon"])
        assert np.abs(Z["lat"] - lat).max() < 1e-3 and np.abs(flon - lon).max() < 1e-3, "grid mismatch " + f
        D = dict(truth=Z["truth"], model=Z["pred"], input=Z["inp"], y=Z["y"])
        assert all(np.isfinite(D[s]).all() for s in SRC), "non-finite " + f
        process_tiles(tiles, D, acc, rows, spectra, date, lead)
        process_storms(lat, lon, lsm, tree, D, date, lead, cases, profiles, os.path.join(a.out, "crops"))
        for name, cd, cl, la0, la1, lo0, lo1 in CROPS:
            if cd == date and cl == lead:
                ii = np.where((lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1))[0]
                save_crop(os.path.join(a.out, "crops", "box_%s_%s_L%03d.npz" % (name, date, lead)), lat, lon, lsm, D, ii,
                          meta=dict(orog=orog[ii].astype(np.float32)))
        done.append("%s_L%03d" % (date, lead))
        print("FILE_DONE", os.path.basename(f), "%.0f s" % (time.time() - t1), "lows", sum(1 for c in cases if c["date"] == date and c["lead"] == lead) // 30, flush=True)
    import csv
    def wcsv(path, rws):
        keys = []
        for r in rws:
            for k in r:
                if k not in keys: keys.append(k)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rws)
    wcsv(os.path.join(a.out, "tile_dates.csv"), rows)
    wcsv(os.path.join(a.out, "cases.csv"), cases)
    np.savez_compressed(os.path.join(a.out, "profiles.npz"), rb=RB, **{k: np.array(v) for k, v in profiles.items()})
    np.savez_compressed(os.path.join(a.out, "spectra.npz"), spectra=spectra, edges=SPEC_EDGES, src=np.array(SRC),
                        names=np.array(["ws", "ke", "vort", "div", "msl"]), tile_lat=np.array([T.lat0 for T in tiles]), tile_lon=np.array([T.lon0 for T in tiles]),
                        n=np.array(acc["n"]), files=np.array(done))
    # per-pixel totals, concatenated over tiles
    cat = lambda L: np.concatenate(L).astype(np.float32)
    px = dict(lat=cat([T.lat[T.core] for T in tiles]), lon=cat([T.lon[T.core] for T in tiles]),
              tile=np.concatenate([np.full(T.ncore, i, np.int16) for i, T in enumerate(tiles)]),
              far=np.concatenate([T.far[T.core] for T in tiles]),
              lsm=cat([lsm[T.idx][T.core] for T in tiles]), orog=cat([orog[T.idx][T.core] for T in tiles]),
              n=np.array(acc["n"]), bands=np.array(BANDS), src=np.array(SRC), files=np.array(done))
    for si, s in enumerate(SRC):
        px["sum_ws_" + s] = cat(acc["sum_ws"][si]); px["sum_ws2_" + s] = cat(acc["sum_ws2"][si]); px["speckle_" + s] = cat(acc["speckle"][si])
        for b in range(nb):
            px["band2_%s_%d" % (s, b)] = cat(acc["band2"][si][b])
    np.savez_compressed(os.path.join(a.out, "tiles_total.npz"), **px)
    # the same sums binned on a 0.125-degree grid (small enough to commit); divide by cnt for pixel means
    dg = 0.125; ny, nx = int(60 / dg), int(120 / dg)
    iy = np.clip(((px["lat"] - 10.0) / dg).astype(int), 0, ny - 1); ix = np.clip(((px["lon"] + 80.0) / dg).astype(int), 0, nx - 1)
    flat = iy * nx + ix
    g = dict(dg=dg, lat0=10.0, lon0=-80.0, n=px["n"], bands=px["bands"], src=px["src"], files=px["files"])
    cnt = np.bincount(flat, minlength=ny * nx).reshape(ny, nx)
    g["cnt"] = cnt.astype(np.int16)
    for k in list(px):
        if k.startswith(("sum_ws_", "sum_ws2_", "speckle_", "band2_")) or k in ("lsm", "orog"):
            g[k] = (np.bincount(flat, weights=px[k].astype(np.float64), minlength=ny * nx).reshape(ny, nx) / np.maximum(cnt, 1)).astype(np.float32)
    np.savez_compressed(os.path.join(a.out, "tiles_grid0125.npz"), **g)
    print("ALL_DONE files", len(done), "tile rows", len(rows), "case rows", len(cases), "%.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
