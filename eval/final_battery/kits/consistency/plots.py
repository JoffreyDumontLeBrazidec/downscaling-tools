#!/usr/bin/env python3
"""Figures from the aggregate of realism_consistency.py (scalars.csv, ratios.csv, arrays.npz, stationary_*.npy).
House colours: truth black, input blue dashed, model red. Usage: plots.py AGG_DIR OUT_DIR [LABEL]"""
import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

agg, out = sys.argv[1], sys.argv[2]; LABEL = sys.argv[3] if len(sys.argv) > 3 else 'GL512 450k'
os.makedirs(out, exist_ok=True)
S = {(r['key'], r['lead']): (float(r['mean']), float(r['se']), int(r['n_dates'])) for r in csv.DictReader(open(f'{agg}/scalars.csv'))}
RT = {(r['key'], r['lead']): r for r in csv.DictReader(open(f'{agg}/ratios.csv'))}
A = np.load(f'{agg}/arrays.npz', allow_pickle=True)
leads = sorted({k[1] for k in S if k[1] != 'all'}); LH = [int(l) for l in leads]
C = {'truth': dict(color='k', ls='-'), 'input': dict(color='tab:blue', ls='--'), 'model': dict(color='tab:red', ls='-')}
NAME = {'truth': '9 km truth', 'input': '36 km input', 'model': LABEL}
VN = {'ws': '10 m wind speed', 'tp': 'rain (12 h)', '2t': '2 m temperature', 'msl': 'sea-level pressure'}
glat, glon = A['glat'][::8], A['glon'][::8]

# 1 spread maps (fine scales < 125 km)
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
for r, v in enumerate(('ws', 'tp')):
    t = np.sqrt(A[f'truth|S|{v}|fine|varmap']); m = np.sqrt(A[f'model|S|{v}|fine|varmap'])
    vmax = float(np.quantile(t, 0.99))
    for c, (f, lab) in enumerate(((t, NAME['truth']), (m, NAME['model']))):
        im = axs[r, c].pcolormesh(glon, glat, f, vmin=0, vmax=vmax, cmap='viridis', shading='nearest', rasterized=True)
        axs[r, c].set_title(f'{VN[v]}: member spread below 125 km, {lab}'); fig.colorbar(im, ax=axs[r, c], shrink=0.8)
    ratio = m / np.maximum(t, 1e-6 * vmax); ratio[t < 0.02 * vmax] = np.nan
    im = axs[r, 2].pcolormesh(glon, glat, ratio, norm=LogNorm(0.5, 2.0), cmap='RdBu_r', shading='nearest', rasterized=True)
    axs[r, 2].set_title(f'{VN[v]}: spread ratio model / truth'); fig.colorbar(im, ax=axs[r, 2], shrink=0.8)
fig.suptitle(f'Where the members differ at fine scales (< 125 km): root-mean variance across 10 members, mean over all dates and leads. {LABEL}')
fig.savefig(f'{out}/fig1_spread_maps.png', dpi=130); plt.close(fig)

# 2 spread by lead and scale
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
for ax, v in zip(axs, ('ws', 'tp', '2t', 'msl')):
    for sc, mk in (('fine', 'o'), ('large', 's')):
        for col, kk, lab in (('tab:red', 'ratio_model_truth', LABEL), ('tab:blue', 'ratio_input_truth', '36 km input')):
            y = np.array([float(RT[(f'S|{v}|{sc}|var', l)][kk]) for l in leads])
            ax.plot(LH, np.sqrt(y), color=col, marker=mk, ls='-' if sc == 'fine' else ':', label=f'{lab}, {"< 125 km" if sc == "fine" else "> 400 km"}')
    ax.axhline(1, color='k', lw=1); ax.set_title(VN[v]); ax.set_xlabel('lead (h)'); ax.set_xticks(LH); ax.set_ylim(0, 2)
axs[0].set_ylabel('member spread / truth member spread'); axs[0].legend(fontsize=7)
fig.suptitle('Ensemble spread against the 9 km truth ensemble, by scale and lead (1 = truth; 30 dates, 10 members)')
fig.savefig(f'{out}/fig2_spread_by_lead.png', dpi=130); plt.close(fig)

# 3 physics
fig, axs = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
e = A['tp_edges']; labs = ['< -1', '-1', '-0.1', '±0.01', '0.01', '0.03', '0.1', '0.3', '1', '3', '10', '30', '100', '> 300']
for s in ('truth', 'input', 'model'):
    h = A[f'{s}|P|tp_hist']; axs[0].plot(range(len(h)), np.maximum(h / h.sum(), 1e-9), marker='o', ms=3, label=NAME[s], **C[s])
axs[0].set_yscale('log'); axs[0].set_xticks(range(len(labs))); axs[0].set_xticklabels(labs, rotation=60, fontsize=7)
axs[0].set_title('Rain per 12 h: share of points per class (raw output)'); axs[0].set_xlabel('lower edge of class (mm)'); axs[0].legend(fontsize=8)
for s in ('truth', 'input', 'model'):
    k = f'{s}|P|share_hist_sea'
    if k in A:
        h = A[k].astype(float); axs[1].plot(np.arange(len(h)), h / h.sum(), marker='o', ms=3, **C[s])
axs[1].set_xticks(range(12)); axs[1].set_xticklabels(['< 0', '0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '> 1'], fontsize=7)
axs[1].set_title('Convective share of rain (cp / tp), sea, rain ≥ 1 mm'); axs[1].set_xlabel('share (lower edge)')
ae = A['ang_edges']
for s in ('truth', 'input', 'model'):
    h = A[f'{s}|P|geo|ang_hist'].astype(float); axs[2].plot(0.5 * (ae[1:] + ae[:-1]), h / h.sum(), **C[s])
axs[2].set_xlim(-90, 120); axs[2].set_title('Angle of 10 m wind to the pressure-gradient (geostrophic) wind\nsea, 30-65 N'); axs[2].set_xlabel('degrees (positive = toward low pressure)')
if 'truth|P|tcw|tp_sum_hist' in A:
    te = A['tcw_edges']; x = np.arange(len(te) - 1)
    for s in ('truth', 'input', 'model'):
        axs[3].plot(x, A[f'{s}|P|tcw|tp_sum_hist'] / np.maximum(A[f'{s}|P|tcw|count_hist'], 1), marker='o', ms=3, **C[s])
    axs[3].set_xticks(x); axs[3].set_xticklabels(['< 10', '10', '15', '20', '25', '30', '40', '> 50'], fontsize=8)
    axs[3].set_title('Mean rain against the input column water'); axs[3].set_xlabel('input total column water (kg/m², lower edge)'); axs[3].set_ylabel('mm / 12 h')
fig.suptitle(f'Cross-variable checks, all dates, leads and members pooled. {LABEL}')
fig.savefig(f'{out}/fig3_physics.png', dpi=130); plt.close(fig)

# 4 lead to lead
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
bands = (('bF', '34-125 km'), ('bM', '125-400 km'), ('bL', '> 400 km')); w = 0.25
for i, (b, bl) in enumerate(bands):
    for j, v in enumerate(('ws', 'tp', '2t', 'msl')):
        r = RT[(f'T|{v}|{b}|rms_change', 'all')]
        axs[0].bar(j + (i - 1) * w, float(r['ratio_model_truth']), w, yerr=float(r['se']), color=plt.cm.Reds(0.4 + 0.25 * i), label=bl if j == 0 else None)
        axs[0].plot(j + (i - 1) * w, float(r['ratio_input_truth']), 'v', color='tab:blue')
axs[0].axhline(1, color='k', lw=1); axs[0].set_xticks(range(4)); axs[0].set_xticklabels([VN[v] for v in ('ws', 'tp', '2t', 'msl')], fontsize=8)
axs[0].set_ylabel('12 h change of a member / truth member (rms)'); axs[0].legend(title='band (bars: model; triangles: input)', fontsize=8); axs[0].set_title('Lead-to-lead change of one member')
for j, x in enumerate(('msl_min', 'ws_max', 'tp_max')):
    for s, dx in (('truth', -0.25), ('input', 0), ('model', 0.25)):
        m, se, n = S[(f'{s}|T|{x}|abs_change', 'all')]
        ref = S[(f'truth|T|{x}|abs_change', 'all')][0]
        axs[1].bar(j + dx, m / ref, 0.25, yerr=se / ref, color=C[s]['color'], alpha=0.8, label=NAME[s] if j == 0 else None)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(['box minimum pressure', 'box maximum wind', 'box maximum rain']); axs[1].axhline(1, color='k', lw=1)
axs[1].set_ylabel('mean |12 h change| relative to truth'); axs[1].legend(fontsize=8); axs[1].set_title('Lead-to-lead change of box extremes, per member')
fig.savefig(f'{out}/fig4_lead_to_lead.png', dpi=130); plt.close(fig)

# 5 edges
eb = A['edge_bins']; xl = ['0-0.25', '0.25-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3-5', '> 5']
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
for v, mk in (('ws', 'o'), ('tp', 's'), ('2t', '^'), ('msl', 'd')):
    y = [float(RT[(f'E|{v}|fine_var|bin{i}', 'all')]['ratio_model_truth']) for i in range(8)]
    axs[0].plot(range(8), np.sqrt(y), marker=mk, label=VN[v])
    y2 = [float(RT[(f'E|{v}|paired_mse|bin{i}', 'all')]['ratio_model_input']) for i in range(8)]
    axs[1].plot(range(8), np.sqrt(y2), marker=mk, label=VN[v])
for ax, t in zip(axs, ('Fine-scale (< 125 km) amplitude, model / truth', 'Member error against the same-index truth member, model / input (rms)')):
    ax.axhline(1, color='k', lw=1); ax.set_xticks(range(8)); ax.set_xticklabels(xl, fontsize=8); ax.set_xlabel('distance from the box edge (degrees)'); ax.set_title(t, fontsize=10); ax.legend(fontsize=8)
fig.savefig(f'{out}/fig5_edges.png', dpi=130); plt.close(fig)

# 6 stationary pattern (grain band < 34 km, mean over all dates, leads, members)
if os.path.exists(f'{agg}/stationary_model_ws.npy'):
    G = np.load(f'{agg}/grid.npz') if os.path.exists(f'{agg}/grid.npz') else None
    fl, fo = A['glat'], A['glon']
    boxes = (('open Atlantic', 38, 48, -45, -30), ('Alps', 43, 49, 4, 16))
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for r, (bn, a0, a1, o0, o1) in enumerate(boxes):
        sl = (slice(np.searchsorted(fl, a0), np.searchsorted(fl, a1)), slice(np.searchsorted(fo, o0), np.searchsorted(fo, o1)))
        fs = [np.load(f'{agg}/stationary_{s}_ws.npy')[sl] for s in ('truth', 'model')]; fs.append(fs[1] - fs[0])
        vm = float(np.quantile(np.abs(fs[0]), 0.995)) or 1.0; vm = max(vm, float(np.quantile(np.abs(fs[1]), 0.995)))
        for c, (f, lab) in enumerate(zip(fs, (NAME['truth'], NAME['model'], 'model minus truth'))):
            im = axs[r, c].pcolormesh(fo[sl[1]], fl[sl[0]], f, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='nearest', rasterized=True)
            axs[r, c].set_title(f'{bn}: {lab}'); fig.colorbar(im, ax=axs[r, c], shrink=0.8)
    fig.suptitle('Stationary imprint: 10 m wind speed below 34 km, averaged over every date, lead and member (m/s). Over open sea it should vanish.')
    fig.savefig(f'{out}/fig6_stationary.png', dpi=130); plt.close(fig)
print('PLOTS_DONE')
