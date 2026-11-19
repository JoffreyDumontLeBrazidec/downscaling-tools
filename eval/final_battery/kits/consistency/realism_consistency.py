#!/usr/bin/env python3
"""Ensemble and physical consistency of a 9 km member row (thread "Check ensemble and physical consistency at 9 km", 17 Nov 2026).

CPU only, reads existing box prediction files (one draw per input member, `x_interp_members` present) and the truth-member
store(s). Three sources are treated alike, 10 members each:
  model = y_pred, truth = enfh pf 1-10 from the zarr store, input = x_interp_members (36 km input, interpolated).

One worker per date walks the leads in order and writes `row_<date>_<lead>.npz` (scalars + small arrays) and
`date_<date>.npz` (stationary-pattern accumulators). `--aggregate` writes scalars.csv, arrays.npz and summary.md.

Sections
  S  spread: point variance across members, full / fine (< 125 km) / large (> 400 km); uniformity (CV of the local sd),
     map correlations between sources, by terrain class, subsampled variance maps.
  P  physics: tp sign and intensity histogram (drizzle share), cp vs tp and convective share, hypsometric msl-sp-2t,
     geostrophic relation (speed ratio, turning angle, fine-scale wind-pressure coupling), divergence / vorticity,
     convergence-rain coupling, rain objects, optional 2d / tcw / tcc bounds when the file has them.
  T  lead-to-lead: rms change of each band between consecutive leads per member, lag correlation, box extremes change.
  E  edges: fine-scale variance and member-paired error against distance from the box edge.
  F  stationary pattern: date sums of the grain band (< 34 km), model and truth, to expose a fixed imprint.
Bands as in the 29 Oct fresh-eyes kit (Gaussian differences on a 0.05 x 0.065 deg grid, half-power about 400 / 125 / 34 km).
"""
import argparse, glob, json, os, sys, traceback
import numpy as np

SIG_KM = {'L': 53.0, 'M': 17.0, 'F': 4.5}
CELL_KM = 5.55
LAT0, LAT1, DLAT = 10.0, 70.0, 0.05
LON0, LON1, DLON = -80.0, 40.0, 0.065
EDGE = float(os.environ.get('RC_EDGE', '1.0'))  # degrees dropped at the box edge
FORCINGS = os.environ.get('RC_FORCINGS', '/e/data1/jureap-data/ecmwf/users/jdumont/datasets/o320_o1280_europe_e2/in_hres_o1280_forcings.zarr')
SRC = ['model', 'truth', 'input']
SVARS = ['ws', '2t', 'msl', 'tp']
TP_EDGES = np.array([-1e9, -1.0, -0.1, -0.01, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e9])
SHARE_EDGES = np.array([-1e9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001, 1e9])
ANG_EDGES = np.arange(-180.0, 180.1, 10.0)
EDGE_BINS = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9])
SUB = 8
TCW_EDGES = np.array([-1e9, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 1e9])
NDER = 4     # members used for derivative-based statistics
REARTH = 6.371e6


def gauss(f, sig_km):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(f, sigma=sig_km / CELL_KM, mode='nearest')


def corr(a, b, m=None):
    if m is not None:
        a = a[m]; b = b[m]
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def build_grid(lat, lon, cache):
    if os.path.exists(cache):
        d = np.load(cache)
        return d['idx'], d['glat'], d['glon']
    from scipy.spatial import cKDTree
    glat = np.arange(LAT0, LAT1 + 1e-9, DLAT); glon = np.arange(LON0, LON1 + 1e-9, DLON)
    LA, LO = np.meshgrid(glat, glon, indexing='ij')
    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.sin(lo) * np.cos(la), np.sin(la)], -1)
    _, idx = cKDTree(xyz(lat, lon)).query(xyz(LA.ravel(), LO.ravel()), k=1)
    idx = idx.reshape(LA.shape)
    np.savez(cache, idx=idx, glat=glat, glon=glon)
    return idx, glat, glon


def static(idx, npoints, cache):
    if os.path.exists(cache):
        d = np.load(cache); return d['lsm'], d['h']
    import zarr
    zf = zarr.open(FORCINGS, mode='r'); data = zf['data']
    assert data.shape[-1] == npoints, f'forcings points {data.shape} != prediction points {npoints}'
    fv = list(zf.attrs.get('variables', []))
    il, iz = (fv.index('lsm'), fv.index('z')) if {'lsm', 'z'} <= set(fv) else (5, 10)   # 5 / 10 = the E2 forcings store order
    lsm = np.asarray(data[0, il, 0, :])[idx]; h = (np.asarray(data[0, iz, 0, :]) / 9.80665)[idx]
    np.savez(cache, lsm=lsm, h=h)
    return lsm, h


def names_of(ds):
    import netCDF4
    v = ds['weather_state']
    try:
        return [str(s) for s in v[:]]
    except Exception:
        return [str(s) for s in netCDF4.chartostring(v[:])]


def to_fields(a, names, idx):
    """a: (points, states) -> dict var -> float32 grid; tp, cp in mm."""
    k = {n: i for i, n in enumerate(names)}
    out = {n: a[idx, i].astype(np.float32) for n, i in k.items()}
    if 'tp' in out and float(np.nanmax(np.abs(out['tp']))) < 5.0:
        for n in ('tp', 'cp', 'lsp'):
            if n in out:
                out[n] = out[n] * 1000.0
    out['ws'] = np.hypot(out['10u'], out['10v'])
    return out


class Truth:
    def __init__(self, stores):
        import zarr
        self.map = {}; self.z = []
        for p in stores:
            g = zarr.open_group(p, mode='r')
            tv = list(g.attrs['variables'])
            for i, r in enumerate(g.attrs['row_map']):
                self.map[(str(r['hdate']), int(r['lead']))] = (len(self.z), i)
            self.z.append((g['data'], tv))
            self.g = g

    def get(self, date, lead):
        key = (str(date), int(lead))
        if key not in self.map:
            return None, None
        s, i = self.map[key]
        data, tv = self.z[s]
        return np.asarray(data[i]), tv      # (vars, members, points)


def hypso(sp, t2, h):
    g, Rd, L = 9.80665, 287.05, 0.0065
    return sp * np.exp(g * h / (Rd * (t2 + 0.5 * L * h)))


def ddx(f, glat):
    dx = REARTH * np.cos(np.radians(glat))[:, None] * np.radians(DLON)
    return np.gradient(f, axis=1) / dx


def ddy(f):
    return np.gradient(f, axis=0) / (REARTH * np.radians(DLAT))


def process_date(args):
    date, leads, pat, stores, outdir, instores = args
    try:
        return _process_date(date, leads, pat, stores, outdir, instores)
    except Exception:
        traceback.print_exc()
        return f'FAIL {date}'


def _process_date(date, leads, pat, stores, outdir, instores=()):
    import netCDF4
    from scipy.ndimage import label
    truth = Truth(stores)
    inp = Truth(instores) if instores else None
    prev = None
    acc = {}
    nacc = 0
    for lead in leads:
        out = os.path.join(outdir, f'row_{date}_{lead}.npz')
        p = pat.format(date=date, lead=lead)
        if not os.path.exists(p):
            print('MISSING', p, flush=True); prev = None; continue
        ds = netCDF4.Dataset(p); ds.set_auto_mask(False)
        names = names_of(ds)
        lat = np.asarray(ds['lat_hres'][:], np.float64); lon = np.asarray(ds['lon_hres'][:], np.float64)
        lon = np.where(lon > 180, lon - 360, lon)
        yp = np.asarray(ds['y_pred'][0]); xm = np.asarray(ds['x_interp_members'][0]); yc = np.asarray(ds['y'][0])
        ds.close()
        idx, glat, glon = build_grid(lat, lon, os.path.join(outdir, 'grid.npz'))
        lsm, h = static(idx, lat.size, os.path.join(outdir, 'static.npz'))
        tr, tv = truth.get(date, int(lead))
        if tr is None:
            print('NO_TRUTH_MEMBERS', date, lead, flush=True); prev = None; continue
        assert tr.shape[-1] == lat.size, (tr.shape, lat.size)
        M = min(yp.shape[0], tr.shape[1], xm.shape[0])
        D = {'model': [to_fields(yp[m], names, idx) for m in range(M)],
             'input': [to_fields(xm[m], names, idx) for m in range(M)],
             'truth': [to_fields(np.stack([tr[tv.index(v), m] for v in tv], -1), tv, idx) for m in range(M)]}
        ctrl = to_fields(yc, names, idx)
        tcw = None
        if inp is not None:
            xr, xv = inp.get(date, int(lead))
            if xr is not None and 'tcw' in xv:
                cl = os.path.join(outdir, 'grid_lres.npz')
                if os.path.exists(cl):
                    idxl = np.load(cl)['idxl']
                else:
                    from scipy.spatial import cKDTree
                    la = np.asarray(inp.g['latitudes'][:], np.float64); lo = np.asarray(inp.g['longitudes'][:], np.float64)
                    def xyz(a_, o_):
                        a_, o_ = np.radians(a_), np.radians(o_)
                        return np.stack([np.cos(a_) * np.cos(o_), np.cos(a_) * np.sin(o_), np.sin(a_)], -1)
                    LA_, LO_ = np.meshgrid(glat, glon, indexing='ij')
                    _, idxl = cKDTree(xyz(la, lo)).query(xyz(LA_.ravel(), LO_.ravel()), k=1)
                    idxl = idxl.reshape(LA_.shape); np.savez(cl + '.tmp%d.npz' % os.getpid(), idxl=idxl); os.replace(cl + '.tmp%d.npz' % os.getpid(), cl)
                tcw = [xr[xv.index('tcw'), m][idxl].astype(np.float32) for m in range(M)]
            del xr
        del yp, xm, tr
        LA, LO = np.meshgrid(glat, glon, indexing='ij')
        edist = np.minimum(np.minimum(LA - LAT0, LAT1 - LA), np.minimum(LO - LON0, LON1 - LO))
        inner = edist > EDGE
        land = lsm > 0.5
        coast = land & (gauss((~land).astype(np.float32), 11.0) > 0.02)
        classes = {'sea': inner & ~land, 'land_lt300': inner & land & (h < 300), 'land_300_1000': inner & land & (h >= 300) & (h < 1000),
                   'land_gt1000': inner & land & (h >= 1000), 'coast': inner & coast}
        geo = inner & ~land & (LA >= 30) & (LA <= 65)
        fcor = (2 * 7.292e-5 * np.sin(np.radians(LA))).astype(np.float32)
        R = {'meta|members': M, 'meta|truth_ctrl_corr_2t': corr(D['truth'][0]['2t'], ctrl['2t'], inner),
             'meta|names': np.array(names), 'meta|truth_names': np.array(tv)}
        cur = {}
        for s in SRC:
            # ---------- bands
            B = {}
            for v in SVARS:
                gl, gm, gf, ff = [], [], [], []
                for m in range(M):
                    f = D[s][m][v]
                    gL = gauss(f, SIG_KM['L']); gM = gauss(f, SIG_KM['M']); gF = gauss(f, SIG_KM['F'])
                    gl.append(gL); gm.append(gM); gf.append(gF); ff.append(f)
                B[v] = (np.stack(ff), np.stack(gl), np.stack(gm), np.stack(gf))
            # ---------- S spread
            for v in SVARS:
                f, gL, gM, gF = B[v]
                parts = {'full': f, 'fine': f - gM, 'large': gL}
                for sc, a in parts.items():
                    var = a.var(axis=0, ddof=1)
                    R[f'{s}|S|{v}|{sc}|var'] = float(var[inner].mean())
                    loc = gauss(var, 25.0)
                    sd = np.sqrt(np.clip(loc, 0, None))
                    R[f'{s}|S|{v}|{sc}|cv_local_sd'] = float(sd[inner].std() / max(sd[inner].mean(), 1e-12))
                    cur[(s, v, sc)] = sd[::SUB, ::SUB].astype(np.float32)
                    if sc in ('full', 'fine'):
                        R[f'{s}|S|{v}|{sc}|varmap'] = var[::SUB, ::SUB].astype(np.float32)
                    if sc == 'fine':
                        for cn, cm in classes.items():
                            if cm.sum() > 1000:
                                R[f'{s}|S|{v}|fine|class|{cn}|var'] = float(var[cm].mean())
                # member amplitude of the fine part (variance in space, mean over members)
                R[f'{s}|S|{v}|fine|member_var'] = float(np.mean([(f[m] - gM[m])[inner].var() for m in range(M)]))
            # ---------- P physics
            hist = np.zeros(len(TP_EDGES) - 1); vol = np.zeros(len(TP_EDGES) - 1)
            hsea = np.zeros(len(SHARE_EDGES) - 1); hland = np.zeros(len(SHARE_EDGES) - 1)
            negv = []; cpg = []; cpn = []; share = []; cp_dry = []
            for m in range(M):
                tp = D[s][m]['tp']; cp = D[s][m].get('cp')
                t = tp[inner]
                hh, _ = np.histogram(t, TP_EDGES); hist += hh
                vv, _ = np.histogram(t, TP_EDGES, weights=t.astype(np.float64)); vol += vv
                negv.append(float(-t[t < 0].sum() / max(t[t > 0].sum(), 1e-9)))
                if cp is not None:
                    c = cp[inner]
                    cpg.append(float((c - t > 0.1).mean())); cpn.append(float((c < -0.01).mean()))
                    share.append(float(np.clip(c, 0, None).sum() / max(np.clip(t, 0, None).sum(), 1e-9)))
                    cp_dry.append(float(((c > 0.1) & (t < 0.01)).mean()))
                    for msk, acc_h in ((classes['sea'], hsea), (inner & land, hland)):
                        w = msk & (tp >= 1.0)
                        hh, _ = np.histogram(cp[w] / tp[w], SHARE_EDGES); acc_h += hh
            R[f'{s}|P|tp_hist'] = hist; R[f'{s}|P|tp_vol'] = vol
            if tcw is not None:
                # rain against the INPUT member's column water (36 km field, same member index); truth has no tcw of its own
                cnt = np.zeros(len(TCW_EDGES) - 1); vsum = np.zeros(len(TCW_EDGES) - 1); c10 = np.zeros(len(TCW_EDGES) - 1); cc = []
                for m in range(M):
                    t = np.clip(D[s][m]['tp'][inner], 0, None).astype(np.float64); w_ = tcw[m][inner]
                    cnt += np.histogram(w_, TCW_EDGES)[0]; vsum += np.histogram(w_, TCW_EDGES, weights=t)[0]
                    c10 += np.histogram(w_[t >= 10.0], TCW_EDGES)[0]
                    if m < 3:
                        cc.append(corr(gauss(np.clip(D[s][m]['tp'], 0, None), SIG_KM['M']), tcw[m], inner))
                R[f'{s}|P|tcw|count_hist'] = cnt; R[f'{s}|P|tcw|tp_sum_hist'] = vsum; R[f'{s}|P|tcw|ge10_hist'] = c10
                R[f'{s}|P|tcw|corr_tp125km_tcw'] = float(np.mean(cc))
                R[f'{s}|P|tcw|rain_volume_share_tcw_lt15'] = float(vsum[:2].sum() / max(vsum.sum(), 1e-9))
                R[f'{s}|P|tcw|ge10_share_tcw_lt20'] = float(c10[:3].sum() / max(c10.sum(), 1e-9))
            R[f'{s}|P|tp_neg_volume_share'] = float(np.mean(negv))
            R[f'{s}|P|tp_min'] = float(min(D[s][m]['tp'][inner].min() for m in range(M)))
            if cpg:
                R[f'{s}|P|cp_gt_tp_frac'] = float(np.mean(cpg)); R[f'{s}|P|cp_neg_frac'] = float(np.mean(cpn))
                R[f'{s}|P|conv_share_volume'] = float(np.mean(share)); R[f'{s}|P|cp_wet_tp_dry_frac'] = float(np.mean(cp_dry))
                R[f'{s}|P|share_hist_sea'] = hsea; R[f'{s}|P|share_hist_land'] = hland
            nd = min(NDER, M)
            if 'sp' in D[s][0]:
                for cn in ('sea', 'land_lt300', 'land_300_1000', 'land_gt1000'):
                    cm = classes[cn]
                    if cm.sum() > 1000:
                        r = np.concatenate([(D[s][m]['msl'] - hypso(D[s][m]['sp'], D[s][m]['2t'], h))[cm] for m in range(nd)])
                        R[f'{s}|P|hypso_rms|{cn}'] = float(np.sqrt(np.mean(r.astype(np.float64) ** 2)))
                        R[f'{s}|P|hypso_bias|{cn}'] = float(r.mean())
            ang_h = np.zeros(len(ANG_EDGES) - 1); st = {k: [] for k in ('c_ws_vg', 'ratio', 'ang_med', 'fine_cpl', 'div_sea', 'vort_sea', 'div_land', 'vort_land', 'conv_tp', 'conv_tp_fine')}
            nobj = {1: [], 10: []}; aobj = {1: [], 10: []}
            for m in range(nd):
                u = gauss(D[s][m]['10u'], 11.0); vv_ = gauss(D[s][m]['10v'], 11.0); pm = gauss(D[s][m]['msl'], 11.0)
                ug = -ddy(pm) / (1.225 * fcor); vg = ddx(pm, glat) / (1.225 * fcor)
                sg = np.hypot(ug, vg); sw = np.hypot(u, vv_)
                st['c_ws_vg'].append(corr(sw, sg, geo))
                w = geo & (sg > 5.0)
                st['ratio'].append(float(np.median(sw[w] / sg[w])))
                ang = np.degrees(np.arctan2(ug * vv_ - vg * u, ug * u + vg * vv_))[w]
                st['ang_med'].append(float(np.median(ang)))
                hh, _ = np.histogram(ang, ANG_EDGES); ang_h += hh
                # fine-scale coupling: wind and geostrophic wind, both minus their own > 125 km part
                uf = u - gauss(u, SIG_KM['M']); vf = vv_ - gauss(vv_, SIG_KM['M'])
                ugf = ug - gauss(ug, SIG_KM['M']); vgf = vg - gauss(vg, SIG_KM['M'])
                st['fine_cpl'].append(0.5 * (corr(uf, ugf, geo) + corr(vf, vgf, geo)))
                div = ddx(u, glat) + ddy(vv_); vort = ddx(vv_, glat) - ddy(u)
                for cn, msk in (('sea', classes['sea']), ('land', inner & land)):
                    st[f'div_{cn}'].append(float(div[msk].std())); st[f'vort_{cn}'].append(float(vort[msk].std()))
                tps = gauss(np.clip(D[s][m]['tp'], 0, None), 11.0)
                st['conv_tp'].append(corr(-gauss(div, SIG_KM['M']), gauss(tps, SIG_KM['M']), classes['sea']))
                st['conv_tp_fine'].append(corr(-(div - gauss(div, SIG_KM['M'])), tps - gauss(tps, SIG_KM['M']), classes['sea']))
                for thr in (1, 10):
                    lab, n = label((D[s][m]['tp'] >= thr) & inner)
                    nobj[thr].append(n); aobj[thr].append(float(((D[s][m]['tp'] >= thr) & inner).sum() / max(n, 1)))
            R[f'{s}|P|geo|corr_ws_vg'] = float(np.mean(st['c_ws_vg'])); R[f'{s}|P|geo|median_ws_over_vg'] = float(np.mean(st['ratio']))
            R[f'{s}|P|geo|median_turning_deg'] = float(np.mean(st['ang_med'])); R[f'{s}|P|geo|ang_hist'] = ang_h
            R[f'{s}|P|geo|fine_wind_pressure_corr'] = float(np.mean(st['fine_cpl']))
            for cn in ('sea', 'land'):
                R[f'{s}|P|div_std|{cn}'] = float(np.mean(st[f'div_{cn}'])); R[f'{s}|P|vort_std|{cn}'] = float(np.mean(st[f'vort_{cn}']))
            R[f'{s}|P|conv_rain_corr_large'] = float(np.mean(st['conv_tp'])); R[f'{s}|P|conv_rain_corr_fine'] = float(np.mean(st['conv_tp_fine']))
            for thr in (1, 10):
                R[f'{s}|P|rain_objects_ge{thr}|count'] = float(np.mean(nobj[thr])); R[f'{s}|P|rain_objects_ge{thr}|mean_cells'] = float(np.mean(aobj[thr]))
            if '2d' in D[s][0]:
                ex = np.stack([(D[s][m]['2d'] - D[s][m]['2t'])[inner] for m in range(M)])
                R[f'{s}|P|2d_gt_2t_frac_0p1K'] = float((ex > 0.1).mean()); R[f'{s}|P|2d_gt_2t_frac_1K'] = float((ex > 1.0).mean())
                R[f'{s}|P|2d_minus_2t_max'] = float(ex.max())
            for wn in ('tcw', 'tcwv'):
                if wn in D[s][0]:
                    tpa = np.stack([D[s][m]['tp'][inner] for m in range(nd)]); wa = np.stack([D[s][m][wn][inner] for m in range(nd)])
                    R[f'{s}|P|{wn}|corr_tp'] = corr(np.clip(tpa, 0, None), wa)
                    wet = tpa >= 10.0
                    R[f'{s}|P|{wn}|p05_where_tp_ge10'] = float(np.quantile(wa[wet], 0.05)) if wet.sum() > 100 else np.nan
                    R[f'{s}|P|{wn}|neg_frac'] = float((wa < 0).mean())
            if 'tcc' in D[s][0]:
                tpa = np.stack([D[s][m]['tp'][inner] for m in range(nd)]); ca = np.stack([D[s][m]['tcc'][inner] for m in range(nd)])
                R[f'{s}|P|tcc|frac_tp_ge1_with_tcc_lt0p5'] = float(((tpa >= 1) & (ca < 0.5)).sum() / max((tpa >= 1).sum(), 1))
                R[f'{s}|P|tcc|out_of_bounds_frac'] = float(((ca < -0.02) | (ca > 1.02)).mean())
            # ---------- extremes per member (for lead-to-lead and exchangeability)
            R[f'{s}|X|msl_min'] = np.array([D[s][m]['msl'][inner].min() for m in range(M)])
            R[f'{s}|X|ws_max'] = np.array([D[s][m]['ws'][inner].max() for m in range(M)])
            R[f'{s}|X|tp_max'] = np.array([D[s][m]['tp'][inner].max() for m in range(M)])
            R[f'{s}|X|area_ge50'] = np.array([float((D[s][m]['tp'][inner] >= 50).sum()) for m in range(M)])
            # ---------- T band fields kept for the next lead (subsampled x2)
            for v in SVARS:
                f, gL, gM, gF = B[v]
                cur[(s, v, 'bF')] = (gF - gM)[:, ::2, ::2].copy(); cur[(s, v, 'bM')] = (gM - gL)[:, ::2, ::2].copy(); cur[(s, v, 'bL')] = gL[:, ::4, ::4].copy()
            # ---------- E edges
            for v in SVARS:
                f, gL, gM, gF = B[v]
                fine = f[:nd] - gM[:nd]
                for i in range(len(EDGE_BINS) - 1):
                    mk = (edist >= EDGE_BINS[i]) & (edist < EDGE_BINS[i + 1])
                    R[f'{s}|E|{v}|fine_var|bin{i}'] = float(np.mean([fine[m][mk].var() for m in range(nd)]))
            # ---------- F stationary pattern accumulators (grain band, < 34 km)
            for v in ('ws', '2t', 'tp'):
                f, gL, gM, gF = B[v]
                g = (f - gF).sum(axis=0)
                acc[(s, v)] = acc.get((s, v), 0) + g.astype(np.float64)
            del B
        nacc += M
        # ---------- cross-source statistics
        for v in SVARS:
            for sc in ('full', 'fine', 'large'):
                for a, b in (('model', 'truth'), ('input', 'truth'), ('model', 'input')):
                    R[f'{a}_{b}|S|{v}|{sc}|corr_local_sd'] = corr(cur[(a, v, sc)], cur[(b, v, sc)], inner[::SUB, ::SUB])
            # member-paired error against the truth member, by edge distance
            nd = min(NDER, M)
            for s in ('model', 'input'):
                err = np.mean([(D[s][m][v] - D['truth'][m][v]) ** 2 for m in range(nd)], axis=0)
                R[f'{s}|E|{v}|paired_mse|inner'] = float(err[inner].mean())
                for i in range(len(EDGE_BINS) - 1):
                    mk = (edist >= EDGE_BINS[i]) & (edist < EDGE_BINS[i + 1])
                    R[f'{s}|E|{v}|paired_mse|bin{i}'] = float(err[mk].mean())
        # ---------- T lead-to-lead
        if prev is not None and int(lead) - int(prev['lead']) == 12:
            for s in SRC:
                for v in SVARS:
                    for b, step in (('bF', 2), ('bM', 2), ('bL', 4)):
                        a1 = cur[(s, v, b)]; a0 = prev[(s, v, b)]; mk = inner[::step, ::step]
                        R[f'{s}|T|{v}|{b}|rms_change'] = float(np.sqrt(np.mean([((a1[m] - a0[m])[mk] ** 2).mean() for m in range(M)])))
                        R[f'{s}|T|{v}|{b}|lag_corr'] = float(np.mean([corr(a1[m], a0[m], mk) for m in range(M)]))
                        R[f'{s}|T|{v}|{b}|rms_level'] = float(np.sqrt(np.mean([(a1[m][mk] ** 2).mean() for m in range(M)])))
                for x in ('msl_min', 'ws_max', 'tp_max'):
                    R[f'{s}|T|{x}|abs_change'] = float(np.mean(np.abs(R[f'{s}|X|{x}'] - prev[f'{s}|X|{x}'])))
        keep = {k: v for k, v in cur.items() if k[2] in ('bF', 'bM', 'bL')}
        keep['lead'] = lead
        for s in SRC:
            for x in ('msl_min', 'ws_max', 'tp_max'):
                keep[f'{s}|X|{x}'] = R[f'{s}|X|{x}']
        prev = keep
        R['date'] = date; R['lead'] = lead
        np.savez_compressed(out, **R)
        print('ROW', date, lead, flush=True)
        del D, cur
    if nacc:
        np.savez_compressed(os.path.join(outdir, f'date_{date}.npz'), n=nacc, **{f'{s}|{v}': (a / nacc).astype(np.float32) for (s, v), a in acc.items()})
    return f'DONE {date}'


def aggregate(outdir):
    files = sorted(glob.glob(os.path.join(outdir, 'row_*.npz')))
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rows.append({k: d[k] for k in d.files})
    leads = sorted({str(r['lead']) for r in rows}); dates = sorted({str(r['date']) for r in rows})
    print('rows', len(rows), 'dates', len(dates), 'leads', leads)
    scal = sorted({k for r in rows for k in r if r[k].ndim == 0 and '|' in k and not k.startswith('meta')})
    lines = ['key,lead,mean,se,n_dates']
    def stat(key, lead=None):
        per = {}
        for r in rows:
            if key in r and (lead is None or str(r['lead']) == lead) and np.isfinite(float(r[key])):
                per.setdefault(str(r['date']), []).append(float(r[key]))
        v = np.array([np.mean(x) for x in per.values()])
        if len(v) == 0:
            return np.nan, np.nan, 0
        return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan), len(v)
    for k in scal:
        for lead in [None] + leads:
            m, s, n = stat(k, lead)
            lines.append(f'{k},{lead or "all"},{m:.6g},{s:.6g},{n}')
    # paired ratios model/truth and model/input per date (ratio of date means), for every key that starts with model|
    lines_r = ['key,lead,ratio_model_truth,se,ratio_input_truth,se,ratio_model_input,se,n_dates']
    def ratio(key, a, b, lead):
        ka, kb = key.replace('model|', a + '|', 1), key.replace('model|', b + '|', 1)
        per = {}
        for r in rows:
            if ka in r and kb in r and (lead is None or str(r['lead']) == lead):
                per.setdefault(str(r['date']), []).append((float(r[ka]), float(r[kb])))
        v = np.array([np.mean([x[0] for x in p]) / np.mean([x[1] for x in p]) for p in per.values() if np.mean([x[1] for x in p]) != 0])
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return np.nan, np.nan, 0
        return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan), len(v)
    for k in [k for k in scal if k.startswith('model|')]:
        for lead in [None] + leads:
            a = ratio(k, 'model', 'truth', lead); b = ratio(k, 'input', 'truth', lead); c = ratio(k, 'model', 'input', lead)
            lines_r.append(f'{k[6:]},{lead or "all"},{a[0]:.5g},{a[1]:.3g},{b[0]:.5g},{b[1]:.3g},{c[0]:.5g},{c[1]:.3g},{a[2]}')
    open(os.path.join(outdir, 'scalars.csv'), 'w').write('\n'.join(lines) + '\n')
    open(os.path.join(outdir, 'ratios.csv'), 'w').write('\n'.join(lines_r) + '\n')
    # arrays: summed histograms (all rows and per lead), mean variance maps, per-member extremes, stationary maps
    A = {}
    for k in sorted({k for r in rows for k in r if r[k].ndim >= 1 and '|' in k and not k.startswith('meta')}):
        if 'hist' in k or 'tp_vol' in k:  # summed
            A[k] = np.sum([r[k] for r in rows if k in r], axis=0)
            for lead in leads:
                A[f'{k}|L{lead}'] = np.sum([r[k] for r in rows if k in r and str(r['lead']) == lead], axis=0)
        elif 'varmap' in k:
            A[k] = np.mean([r[k] for r in rows if k in r], axis=0).astype(np.float32)
            for lead in (leads[0], leads[-1]):
                A[f'{k}|L{lead}'] = np.mean([r[k] for r in rows if k in r and str(r['lead']) == lead], axis=0).astype(np.float32)
        elif '|X|' in k:
            A[k] = np.stack([r[k] for r in rows if k in r])
    A['X|date'] = np.array([str(r['date']) for r in rows]); A['X|lead'] = np.array([str(r['lead']) for r in rows])
    g = np.load(os.path.join(outdir, 'grid.npz')); A['glat'] = g['glat']; A['glon'] = g['glon']
    A['tcw_edges'] = TCW_EDGES; A['tp_edges'] = TP_EDGES; A['share_edges'] = SHARE_EDGES; A['ang_edges'] = ANG_EDGES; A['edge_bins'] = EDGE_BINS
    np.savez_compressed(os.path.join(outdir, 'arrays.npz'), **A)
    # stationary pattern
    dfiles = sorted(glob.glob(os.path.join(outdir, 'date_*.npz')))
    if dfiles:
        st = np.load(os.path.join(outdir, 'static.npz')); lsm, h = st['lsm'], st['h']
        LA, LO = np.meshgrid(g['glat'], g['glon'], indexing='ij')
        edist = np.minimum(np.minimum(LA - LAT0, LAT1 - LA), np.minimum(LO - LON0, LON1 - LO)); inner = edist > EDGE
        cls = {'sea': inner & (lsm < 0.5), 'land_lt300': inner & (lsm > 0.5) & (h < 300), 'land_gt1000': inner & (lsm > 0.5) & (h >= 1000)}
        S = {}; out = ['key,class,value']
        for v in ('ws', '2t', 'tp'):
            for s in SRC:
                stack = np.stack([np.load(f)[f'{s}|{v}'] for f in dfiles])
                S[(s, v)] = stack
            n = len(dfiles)
            for s in ('model', 'input'):
                dif = S[(s, v)] - S[('truth', v)]            # per date: mean grain of source minus truth
                mean = dif.mean(axis=0); null = dif.var(axis=0, ddof=1) / n
                for cn, cm in cls.items():
                    out.append(f'{s}_minus_truth|{v}|grain_mean_var,{cn},{float((mean[cm] ** 2).mean()):.6g}')
                    out.append(f'{s}_minus_truth|{v}|grain_null_var,{cn},{float(null[cm].mean()):.6g}')
                    out.append(f'{s}_minus_truth|{v}|stationary_ratio,{cn},{float((mean[cm] ** 2).mean() / null[cm].mean()):.4g}')
                if s == 'model':
                    np.save(os.path.join(outdir, f'stationary_model_minus_truth_{v}.npy'), mean.astype(np.float32))
            for s in SRC:
                mean = S[(s, v)].mean(axis=0)
                np.save(os.path.join(outdir, f'stationary_{s}_{v}.npy'), mean.astype(np.float32))
                for cn, cm in cls.items():
                    out.append(f'{s}|{v}|grain_mean_rms,{cn},{float(np.sqrt((mean[cm] ** 2).mean())):.6g}')
            for cn, cm in cls.items():
                out.append(f'model_truth|{v}|grain_mean_corr,{cn},{corr(S[("model", v)].mean(axis=0), S[("truth", v)].mean(axis=0), cm):.4f}')
        open(os.path.join(outdir, 'stationary.csv'), 'w').write('\n'.join(out) + '\n')
    r0 = rows[0]
    meta = {'rows': len(rows), 'dates': dates, 'leads': leads, 'names': [str(x) for x in r0['meta|names']], 'truth_names': [str(x) for x in r0['meta|truth_names']],
            'truth_ctrl_corr_2t_min': float(min(float(r['meta|truth_ctrl_corr_2t']) for r in rows))}
    json.dump(meta, open(os.path.join(outdir, 'meta.json'), 'w'), indent=1)
    print('AGGREGATE_DONE', json.dumps(meta)[:400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', help='pattern with {date} and {lead}')
    ap.add_argument('--truth-stores', nargs='*', default=[]); ap.add_argument('--input-stores', nargs='*', default=[])
    ap.add_argument('--dates', nargs='*'); ap.add_argument('--leads', nargs='*', default=['012', '024', '036', '048', '060', '072'])
    ap.add_argument('--outdir', required=True); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--aggregate', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.aggregate:
        aggregate(a.outdir); return
    dates = a.dates or sorted({os.path.basename(p).split('_')[1] for lead in a.leads for p in glob.glob(a.pred.format(date='*', lead=lead))})
    print('dates', len(dates), dates, flush=True)
    jobs = [(d, a.leads, a.pred, a.truth_stores, a.outdir, a.input_stores) for d in dates]
    if a.workers <= 1 or len(jobs) == 1:
        for j in jobs:
            print(process_date(j), flush=True)
        return
    # caches are built by the first date alone, then the pool runs
    if not os.path.exists(os.path.join(a.outdir, 'static.npz')):
        print(process_date((dates[0], a.leads[:1], a.pred, a.truth_stores, a.outdir, a.input_stores)), flush=True)
        for f in glob.glob(os.path.join(a.outdir, f'row_{dates[0]}_*.npz')) + glob.glob(os.path.join(a.outdir, f'date_{dates[0]}.npz')):
            os.remove(f)
    # spawn, not fork: workers start without the arrays, threads and locks of the first date that ran in this process.
    # Results do not depend on the start method. A worker killed for memory (about 5 GB each) leaves the pool waiting
    # for ever: give the kit fewer workers rather than more.
    from multiprocessing import get_context
    with get_context('spawn').Pool(a.workers) as pool:
        for r in pool.imap_unordered(process_date, jobs):
            print(r, flush=True)


if __name__ == '__main__':
    main()
