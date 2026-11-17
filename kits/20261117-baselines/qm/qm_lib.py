"""Classical baseline QM (17 Nov 2026): per-point empirical quantile mapping of the 36 km input, interpolated to the 9 km
O1280 grid, onto the 9 km climatology of the training pairs. Shared pieces: variables and their transforms, probability
levels, the input interpolation (inverse-distance over the 4 nearest O320 points), the vectorised mapping.

Variables mapped: 2t, msl, ws (10 m wind speed; 10u and 10v are scaled by ws_mapped / ws_in), tp (12 h). Stored in float16
in "working units" chosen so float16 keeps 0.03 K, 0.06 hPa, 0.03 m/s, 0.25 mm at the extremes:
    2t  K - 273.15        msl  Pa / 100 - 1000        ws  m/s        tp  mm per 12 h (m x 1000)
Fit window: the calendar month of the start date and its two neighbours (12 windows); samples = training pairs whose
base date falls in the window, leads 12..72 h pooled (the leads the downscalers were trained on).
Mapping y = Q_t(F_in(x)), F_in by linear interpolation between the stored input quantiles. Beyond the fitted range:
additive shift (2t, msl) or ratio (ws, tp) of the outermost quantile pair. tp: an input below DRY_MM (dry) takes the rank
p0/2 where p0 is the input's dry frequency, so a dry input stays dry unless the 9 km climatology is wetter at that rank.
"""
from __future__ import annotations

import numpy as np

VARS = ("2t", "msl", "ws", "tp")
ADDITIVE = {"2t": True, "msl": True, "ws": False, "tp": False}
DRY_MM = 0.05
LEADS = (12, 24, 36, 48, 60, 72)
PROBS = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1] + [round(0.15 + 0.05 * i, 2) for i in range(15)]
                 + [0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 1.0])
NQ = PROBS.size


def window_months(m):
    """Calendar months (1..12) of the fit window centred on month m."""
    return ((m - 2) % 12 + 1, m, m % 12 + 1)


def to_work(name, phys):
    """phys: dict with 10u, 10v, 2t, msl, tp arrays in SI units -> working-unit array of variable `name` (float32)."""
    if name == "2t":
        return (np.asarray(phys["2t"], np.float32) - 273.15)
    if name == "msl":
        return np.asarray(phys["msl"], np.float32) / 100.0 - 1000.0
    if name == "ws":
        return np.hypot(np.asarray(phys["10u"], np.float32), np.asarray(phys["10v"], np.float32))
    if name == "tp":
        return np.clip(np.asarray(phys["tp"], np.float32), 0.0, None) * 1000.0
    raise KeyError(name)


def from_work(name, w):
    if name == "2t":
        return w + 273.15
    if name == "msl":
        return (w + 1000.0) * 100.0
    if name == "tp":
        return np.clip(w, 0.0, None) / 1000.0
    return np.clip(w, 0.0, None)


def unit_xyz(lat, lon):
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def idw_weights(src_lat, src_lon, dst_lat, dst_lon, k=4):
    """(idx int32 (n_dst, k), w float32 (n_dst, k)): inverse squared chord distance over the k nearest source points;
    a destination on top of a source point takes it with weight 1."""
    from scipy.spatial import cKDTree
    tree = cKDTree(unit_xyz(src_lat, src_lon))
    d, idx = tree.query(unit_xyz(dst_lat, dst_lon), k=k, workers=-1)
    d = np.maximum(d, 0.0)
    exact = d[:, 0] < 1e-9
    w = 1.0 / np.maximum(d, 1e-9) ** 2
    w[exact] = 0.0
    w[exact, 0] = 1.0
    w /= w.sum(axis=1, keepdims=True)
    return idx.astype(np.int32), w.astype(np.float32)


def idw_apply(field, idx, w):
    """field (..., n_src) -> (..., n_dst)."""
    f = np.asarray(field, np.float32)
    return (f[..., idx] * w).sum(axis=-1)


def _bracket(q, x):
    """q (n, K) non-decreasing rows, x (n,) -> (k, frac): x lies between q[k] and q[k+1] at fraction frac; k in 0..K-2,
    frac clipped to [0, 1] (the outside part is handled by the caller)."""
    n, K = q.shape
    k = (q <= x[:, None]).sum(axis=1) - 1
    k = np.clip(k, 0, K - 2)
    r = np.arange(n)
    lo, hi = q[r, k], q[r, k + 1]
    span = hi - lo
    frac = np.where(span > 0, (x - lo) / np.where(span > 0, span, 1.0), 0.5)
    return k, np.clip(frac, 0.0, 1.0)


def qm_map(name, x, qin, qt):
    """Quantile mapping of x (n,) with per-point input quantiles qin (n, NQ) and target quantiles qt (n, NQ), working
    units, float32 in and out."""
    x = np.asarray(x, np.float32)
    qin = np.asarray(qin, np.float32)
    qt = np.asarray(qt, np.float32)
    k, frac = _bracket(qin, x)
    p = PROBS[k] + frac * (PROBS[k + 1] - PROBS[k])
    r = np.arange(x.size)
    if name == "tp":
        # dry input: rank p0/2 with p0 the input's dry frequency (largest PROBS with qin <= DRY_MM)
        dry_in = qin <= DRY_MM
        p0 = np.where(dry_in.any(axis=1), PROBS[np.clip(dry_in.sum(axis=1) - 1, 0, NQ - 1)], 0.0)
        p = np.where(x <= DRY_MM, 0.5 * p0, p)
    # target value at rank p, linear in p between the stored probability levels
    j = np.clip(np.searchsorted(PROBS, p, side="right") - 1, 0, NQ - 2)
    g = (p - PROBS[j]) / (PROBS[j + 1] - PROBS[j])
    y = qt[r, j] + g * (qt[r, j + 1] - qt[r, j])
    # outside the fitted input range
    lo_in, hi_in, lo_t, hi_t = qin[:, 0], qin[:, -1], qt[:, 0], qt[:, -1]
    above, below = x > hi_in, x < lo_in
    if ADDITIVE[name]:
        y = np.where(above, hi_t + (x - hi_in), y)
        y = np.where(below, lo_t + (x - lo_in), y)
    else:
        ratio_hi = np.where(hi_in > 0, hi_t / np.where(hi_in > 0, hi_in, 1.0), 1.0)
        y = np.where(above, x * ratio_hi, y)
        y = np.where(below & (x > DRY_MM), np.maximum(x - lo_in + lo_t, 0.0), y)
        y = np.maximum(y, 0.0)
    return y.astype(np.float32)


def check_qm_map():
    """Self-test on synthetic data: identity climatologies map x to x; a shifted target shifts; dry stays dry."""
    rng = np.random.default_rng(0)
    n = 1000
    s = np.sort(rng.normal(10, 5, (n, 4000)), axis=1)
    q = np.quantile(s, PROBS, axis=1).T.astype(np.float32)
    x = rng.normal(10, 5, n).astype(np.float32)
    assert np.abs(qm_map("2t", x, q, q) - x).max() < 1e-3
    assert np.abs(qm_map("2t", x, q, q + 2.0) - (x + 2.0)).max() < 1e-3
    tp_in = np.where(rng.random((n, 4000)) < 0.6, 0.0, rng.gamma(0.7, 4.0, (n, 4000)))
    qi = np.quantile(tp_in, PROBS, axis=1).T.astype(np.float32)
    y = qm_map("tp", np.zeros(n, np.float32), qi, qi)
    assert (y == 0).all()
    y = qm_map("tp", np.full(n, 1e6, np.float32), qi, qi * 2)
    assert np.allclose(y, 2e6, rtol=1e-4)
    return True


if __name__ == "__main__":
    print("PROBS", PROBS.size, PROBS.tolist())
    print("check_qm_map", check_qm_map())
