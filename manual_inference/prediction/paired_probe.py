"""Paired-input probe (ALREADY-TRIED TD-204, docs epic note 20261119_paired_input_probe.md).

Opt-in by env var, off by default:
  MI_PAIRED_INPUT_PROBE=1        replace each shared in_lres channel of input member m by the
                                 TRUTH member m (out_hres) projected down to the lres grid.
  MI_PAIRED_SIGMA_KM=<float>     Gaussian width of the down-projection (default 18 km, about
                                 half the o320 spacing); k nearest hres points, k=MI_PAIRED_K (32).
  MI_PAIRED_GAP_DIR=<dir>        if set, write per-sample scale-decomposed variance / bias of
                                 real lres input vs truncated truth (the owner's spectral-gap
                                 question) as gap_<idx>.npz there.
Forcing / input-only channels keep the real input's values.
"""

from __future__ import annotations

import os

import numpy as np

EARTH_R_KM = 6371.0


def enabled() -> bool:
    return os.environ.get("MI_PAIRED_INPUT_PROBE", "0") == "1"


def _xyz(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def gaussian_knn_operator(src_lat, src_lon, dst_lat, dst_lon, sigma_km: float, k: int):
    """Row-normalised sparse operator dst <- src: Gaussian weights over the k nearest src points."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix

    src, dst = _xyz(src_lat, src_lon), _xyz(dst_lat, dst_lon)
    d, j = cKDTree(src).query(dst, k=k)
    d_km = 2.0 * EARTH_R_KM * np.arcsin(np.clip(d / 2.0, 0, 1))
    w = np.exp(-0.5 * (d_km / sigma_km) ** 2)
    w /= w.sum(1, keepdims=True)
    rows = np.repeat(np.arange(dst.shape[0]), k)
    return csr_matrix((w.ravel(), (rows, j.ravel())), shape=(dst.shape[0], src.shape[0]))


def build_down(lat_hres, lon_hres, lat_lres, lon_lres):
    sigma = float(os.environ.get("MI_PAIRED_SIGMA_KM", "18"))
    k = int(os.environ.get("MI_PAIRED_K", "32"))
    return gaussian_knn_operator(lat_hres, lon_hres, lat_lres, lon_lres, sigma, k)


def replace_inputs(x_in, y, down, name_to_idx_in, name_to_idx_out):
    """x_in [sample, ens, grid_l, vars_in], y [sample, ens, grid_h, vars_out] -> x_in copy, shared names."""
    x = x_in.copy()
    shared = [n for n in name_to_idx_out if n in name_to_idx_in]
    if not shared:
        raise ValueError("paired probe: no channel shared between in_lres and out_hres")
    n_ens = min(x.shape[1], y.shape[1])
    for s in range(x.shape[0]):
        for m in range(n_ens):
            yo = y[s, m][:, [name_to_idx_out[n] for n in shared]]
            x[s, m][:, [name_to_idx_in[n] for n in shared]] = (down @ yo).astype(x.dtype)
    return x, shared


SCALES_KM = (75.0, 150.0, 300.0)  # band edges: <75, 75-150, 150-300, >300 km
LRES_SPACING_KM = 36.0


def scale_gap(x_real, x_trunc, lat_l, lon_l, shared, name_to_idx_in, out_path):
    """Scale-decomposed variance (box/global lres grid) of real input vs truncated truth.

    Band b = smooth_{L_b-1} - smooth_{L_b} with Gaussian smoothers of width L/2.
    Writes var_real, var_trunc [sample, ens, band, var], bias and rms of (trunc - real).
    """
    ops = [
        gaussian_knn_operator(
            lat_l, lon_l, lat_l, lon_l, L / 2.0,
            min(len(lat_l), int(np.ceil(np.pi * (1.25 * L) ** 2 / LRES_SPACING_KM**2))),
        )
        for L in SCALES_KM
    ]
    idx = [name_to_idx_in[n] for n in shared]
    S, E = x_real.shape[:2]
    nb = len(SCALES_KM) + 1
    vr = np.zeros((S, E, nb, len(idx)))
    vt = np.zeros_like(vr)
    for s in range(S):
        for m in range(E):
            for arr, out in ((x_real, vr), (x_trunc, vt)):
                f = arr[s, m][:, idx].astype(np.float64)
                f = f - f.mean(0)
                levels = [f] + [op @ f for op in ops]
                bands = [levels[i] - levels[i + 1] for i in range(len(ops))] + [levels[-1]]
                out[s, m] = np.stack([b.var(0) for b in bands])
    diff = (x_trunc[..., idx] - x_real[..., idx]).astype(np.float64)
    np.savez_compressed(
        out_path,
        var_real=vr,
        var_trunc=vt,
        bias=diff.mean(2),
        rms=np.sqrt((diff**2).mean(2)),
        shared=np.array(shared),
        band_edges_km=np.array(SCALES_KM),
    )
