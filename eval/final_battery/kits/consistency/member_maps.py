#!/usr/bin/env python3
"""Member maps beside truth members (thread "Check ensemble and physical consistency at 9 km", 17 Nov 2026).

Needs grid.npz and static.npz written by realism_consistency.py in --cache.
  --case NAME,DATE,LEAD,VAR,LAT0,LAT1,LON0,LON1   members figure: rows truth / model / input, columns members 1..N
  --case NAME,DATE,LEAD,VAR,low                  same, box of +-6 deg around the truth-control msl minimum in 15-50 N, 80-30 W
  --seq  NAME,DATE,VAR,MEMBER,LAT0,LAT1,LON0,LON1 lead sequence of one member: rows truth / model / input, columns leads
VAR: ws, tp, 2t, msl, or fine_ws / fine_2t (field minus its own > 125 km part).
Members share an index only (truth pf k and input pf k are not the same trajectory beyond the index).
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realism_consistency import Truth, to_fields, names_of, gauss, SIG_KM

LEADS = os.environ.get('RC_LEADS', '012 024 036 048 060 072').split()


def load(pat, truth, date, lead, idx, members):
    import netCDF4
    ds = netCDF4.Dataset(pat.format(date=date, lead=lead)); ds.set_auto_mask(False)
    names = names_of(ds)
    yp = np.asarray(ds['y_pred'][0]); xm = np.asarray(ds['x_interp_members'][0]); yc = np.asarray(ds['y'][0]); ds.close()
    tr, tv = truth.get(date, int(lead))
    D = {'model': [to_fields(yp[m], names, idx) for m in members], 'input': [to_fields(xm[m], names, idx) for m in members],
         'truth': [to_fields(np.stack([tr[tv.index(v), m] for v in tv], -1), tv, idx) for m in members]}
    return D, to_fields(yc, names, idx)


def style(var):
    if var == 'tp':
        try:
            import cmocean
            cm = cmocean.cm.rain
        except Exception:
            try:   # embedded cmocean rain (overlay agent-work/20261027-db300-vs-ca300/overlay, on PYTHONPATH in run_jupiter.sbatch)
                from _cmocean_rain import RAIN_RGB
                from matplotlib.colors import ListedColormap
                cm = ListedColormap(RAIN_RGB, name='rain')
            except Exception:
                cm = 'YlGnBu'
        return cm, 'mm / 12 h'
    if var.startswith('fine'):
        return 'RdBu_r', {'fine_ws': 'm/s', 'fine_2t': 'K'}.get(var, '')
    return {'ws': 'viridis', '2t': 'RdYlBu_r', 'msl': 'Spectral_r'}[var], {'ws': 'm/s', '2t': 'K', 'msl': 'hPa'}[var]


def field(d, var):
    if var.startswith('fine_'):
        f = d[var[5:]]
        return f - gauss(f, SIG_KM['M'])
    if var == 'msl':
        return d['msl'] / 100.0
    if var == 'tp':
        return np.clip(d['tp'], 0, None)
    return d[var]


def panel(ax, f, glat, glon, sl, lsm, cm, vmin, vmax, norm=None):
    im = ax.pcolormesh(glon[sl[1]], glat[sl[0]], f[sl], cmap=cm, vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm, shading='nearest', rasterized=True)
    ax.contour(glon[sl[1]], glat[sl[0]], lsm[sl], levels=[0.5], colors='k', linewidths=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect(1.0 / np.cos(np.radians(0.5 * (glat[sl[0]][0] + glat[sl[0]][-1]))))
    return im


def limits(fs, var):
    a = np.concatenate([f.ravel() for f in fs])
    if var == 'tp':
        import matplotlib.colors as mc
        top = max(float(np.quantile(a, 0.9995)), 5.0)
        return None, None, mc.PowerNorm(0.5, vmin=0, vmax=top)
    if var.startswith('fine'):
        v = float(np.quantile(np.abs(a), 0.995)); return -v, v, None
    return float(np.quantile(a, 0.002)), float(np.quantile(a, 0.998)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True); ap.add_argument('--truth-stores', nargs='+', required=True)
    ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--case', action='append', default=[]); ap.add_argument('--seq', action='append', default=[])
    ap.add_argument('--nmem', type=int, default=4); ap.add_argument('--label', default='GL512 450k')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = np.load(os.path.join(a.cache, 'grid.npz')); idx, glat, glon = g['idx'], g['glat'], g['glon']
    lsm = np.load(os.path.join(a.cache, 'static.npz'))['lsm']
    truth = Truth(a.truth_stores)
    rows = [('truth', '9 km truth'), ('model', a.label), ('input', '36 km input')]
    def box(lat0, lat1, lon0, lon1):
        return (slice(int(np.searchsorted(glat, lat0)), int(np.searchsorted(glat, lat1))), slice(int(np.searchsorted(glon, lon0)), int(np.searchsorted(glon, lon1))))
    for c in a.case:
        p = c.split(','); name, date, lead, var = p[:4]
        D, ctrl = load(a.pred, truth, date, lead, idx, list(range(a.nmem)))
        if p[4] == 'low':
            LA, LO = np.meshgrid(glat, glon, indexing='ij')
            m = np.where((LA > 15) & (LA < 50) & (LO > -78) & (LO < -30), ctrl['msl'], 1e9); i, j = np.unravel_index(np.argmin(m), m.shape)
            b = (glat[i] - 6, glat[i] + 6, glon[j] - 7, glon[j] + 7)
        else:
            b = tuple(float(x) for x in p[4:8])
        sl = box(*b)
        F = {s: [field(D[s][k], var) for k in range(a.nmem)] for s, _ in rows}
        cm, unit = style(var); vmin, vmax, norm = limits([f[sl] for s in F for f in F[s]], var)
        fig, axs = plt.subplots(3, a.nmem, figsize=(3.4 * a.nmem, 9.6), constrained_layout=True)
        for r, (s, lab) in enumerate(rows):
            for k in range(a.nmem):
                im = panel(axs[r, k], F[s][k], glat, glon, sl, lsm, cm, vmin, vmax, norm)
                axs[r, k].set_title(f'{lab}, member {k + 1}', fontsize=9)
        fig.colorbar(im, ax=axs, shrink=0.6, label=f'{var} ({unit})')
        fig.suptitle(f'{name}: {var}, init {date} +{int(lead)} h, {b[0]:.0f}-{b[1]:.0f} N, {b[2]:.0f} to {b[3]:.0f} E. Members share an index only.', fontsize=10)
        out = os.path.join(a.out, f'members_{name}_{var}_{date}_L{lead}.png'); fig.savefig(out, dpi=130); plt.close(fig); print('WROTE', out, flush=True)
    for c in a.seq:
        p = c.split(','); name, date, var, mem = p[0], p[1], p[2], int(p[3]); b = tuple(float(x) for x in p[4:8]); sl = box(*b)
        F = {s: [] for s, _ in rows}
        for lead in LEADS:
            D, _ = load(a.pred, truth, date, lead, idx, [mem - 1])
            for s, _ in rows:
                F[s].append(field(D[s][0], var))
        cm, unit = style(var); vmin, vmax, norm = limits([f[sl] for s in F for f in F[s]], var)
        fig, axs = plt.subplots(3, len(LEADS), figsize=(3.0 * len(LEADS), 8.4), constrained_layout=True)
        for r, (s, lab) in enumerate(rows):
            for k, lead in enumerate(LEADS):
                im = panel(axs[r, k], F[s][k], glat, glon, sl, lsm, cm, vmin, vmax, norm)
                axs[r, k].set_title(f'{lab}, +{int(lead)} h', fontsize=9)
        fig.colorbar(im, ax=axs, shrink=0.6, label=f'{var} ({unit})')
        fig.suptitle(f'{name}: {var}, init {date}, member {mem} across leads (each lead is an independent draw)', fontsize=10)
        out = os.path.join(a.out, f'leads_{name}_{var}_{date}_m{mem}.png'); fig.savefig(out, dpi=120); plt.close(fig); print('WROTE', out, flush=True)


if __name__ == '__main__':
    main()
