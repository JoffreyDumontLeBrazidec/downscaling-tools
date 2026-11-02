"""o2560 Europe-box read kit (2 Nov 2026): helpers shared by the predictor, the scoring scripts and the tests.

No torch and no anemoi import here, so the tests run on a login node.

Rows of the read (one checkpoint each, all on the o1280 -> o2560 Europe box E2, 10-70 N, 80 W-40 E):
  A100  db_o2560_crop_100k (AG crop chain) at step 100,000, EMA-10000, code 4ad32838c
  B24   db_o2560_e2_50k_v2 (Jupiter full-box chain) at step 24,000, EMA-10000, code 20ba15241 (equal GPU-hours with A100)
  B50   db_o2560_e2_50k_v2 at step 50,000, EMA-10000, code 20ba15241 (its last step)
Input reference: the o1280 operational IFS input interpolated to o2560 (x_interp in every file), scored as a one-member row.
"""
from __future__ import annotations

import math

import numpy as np

KIT_ROOT = "/e/data1/jureap-data/ecmwf/users/jdumont/agent-work/20261102-o2560-box-read"
OUT_ROOT = "/e/data1/jureap-data/ecmwf/users/jdumont/outputs/o2560_box_read_20261102"

# Validation split of both chains: dataloader.valid_date / test_date 2024-11 (training ends 2024-10); the store ends 12 Nov 2024.
DATES12 = [f"202411{d:02d}" for d in range(1, 13)]
LEADS_ALL = list(range(6, 73, 6))
LEADS_DEFAULT = [12, 24, 36, 48, 60, 72]
MEMBERS = 10

# Rule 31 (seed locking): one seed per (date, lead, member), identical across arms. Same function as
# downscaling-tools eval/predict/seeding.py and the E2 seed-fix kit (20261030-seedfix-wide/kit/e2_rain_predict.py).
SEED_BASE = 1234
SEED_FORMULA = "numpy SeedSequence([base, yyyymmdd, lead_h, member]).generate_state(1, uint32)"


def derive_draw_seed(base, date, lead, member) -> int:
    """Sampler seed for one draw. date as yyyymmdd int, lead in hours, member 0-based."""
    return int(np.random.SeedSequence([int(base), int(date), int(lead), int(member)]).generate_state(1, dtype=np.uint32)[0])


def case_seeds(hdate: str, lead: int, n: int) -> list[int]:
    return [derive_draw_seed(SEED_BASE, int(hdate), int(lead), m) for m in range(n)]


# The 11 outputs of both chains, in the order of the training config's out_hres `reorder` (no sp, rule 29 as built on the box).
OUTPUT_VARS = ("10u", "10v", "2d", "2t", "msl", "skt", "tcw", "cp", "tp", "z_500", "t_850")
PRECIP_VARS = ("tp", "cp")

# 30-step churn-off piecewise sampler of the run-judging protocol (59 model calls), as the E2 seed-fix kit.
PW_SCHED = {
    "schedule_type": "experimental_piecewise", "num_steps": 30, "sigma_max": 100000.0,
    "sigma_transition": 10.0, "sigma_min": 0.03, "high_schedule_type": "exponential",
    "low_schedule_type": "karras", "num_steps_high": 10, "num_steps_low": 20, "rho": 7.0,
}
PW_SAMPLER = {"sampler": "heun", "S_churn": 0.0, "S_min": 0.75, "S_max": 100000.0, "S_noise": 1.05}


def clip_precip(a: np.ndarray, states, axis_state: int = -1):
    """Clip tp and cp at 0 along the state axis. Returns (clipped copy, {state: (n negative, min value)})."""
    out = np.array(a, copy=True)
    info = {}
    for st in PRECIP_VARS:
        if st not in states:
            continue
        k = list(states).index(st)
        sl = [slice(None)] * out.ndim
        sl[axis_state] = k
        v = out[tuple(sl)]
        neg = v < 0
        info[st] = (int(neg.sum()), float(v.min()) if v.size else 0.0)
        out[tuple(sl)] = np.where(neg, 0.0, v)
    return out, info


def deaccumulate(acc_by_lead: dict[int, np.ndarray], window_h: int = 6) -> dict[int, np.ndarray]:
    """Turn accumulations from the forecast start (one array per lead, in m) into window_h-hour windows ending at each lead.

    NOT used on these stores: both the o2560 truth (i4ql) and the o1280 input (od) store tp and cp as the 6 h window that ends
    at the lead (anemoi-datasets `accumulations` source, variables_metadata period [0, 6] at step 6, and every lead a 6 h
    window; the predictor checks the per-lead box means in its dry run). The function is kept and tested so the kit states
    what de-accumulation would be, and so a store holding totals since the start can be handled the same way."""
    leads = sorted(acc_by_lead)
    out = {}
    for L in leads:
        prev = L - window_h
        if prev <= 0:
            out[L] = np.asarray(acc_by_lead[L], dtype=np.float64)
        else:
            if prev not in acc_by_lead:
                raise KeyError(f"lead {prev} h needed to de-accumulate lead {L} h")
            out[L] = np.asarray(acc_by_lead[L], dtype=np.float64) - np.asarray(acc_by_lead[prev], dtype=np.float64)
    return out


def edge_strip(lat: np.ndarray, lon180: np.ndarray, width_deg: float = 1.0) -> np.ndarray:
    """True on the 1-degree strip at the edge of the E2 box (10-70 N, 80 W-40 E): excluded from every score."""
    return (lat > 70.0 - width_deg) | (lat < 10.0 + width_deg) | (lon180 > 40.0 - width_deg) | (lon180 < -80.0 + width_deg)


def od_area_mask(lat: np.ndarray, lon360: np.ndarray, area_nwse=(71.5, -81.5, 8.5, 41.5)) -> np.ndarray:
    """anemoi `area` cut [N, W, S, E] on a global grid with longitudes in 0..360 (the in_lres box store was cut this way)."""
    n, w, s, e = area_nwse
    lon180 = ((lon360 + 180.0) % 360.0) - 180.0
    return (lat <= n) & (lat >= s) & (lon180 >= w) & (lon180 <= e)


def fcrps_terms(f: np.ndarray, y: np.ndarray):
    """f (m, p), y (p,): pointwise fair CRPS, ensemble VARIANCE (ddof 1) and squared ensemble-mean error.
    Spread over an area is sqrt(weighted mean of the variance), never the mean of the standard deviation.
    With one member (the input row IN): absolute error, zero variance, squared error."""
    m = f.shape[0]
    term1 = np.mean(np.abs(f - y[None]), axis=0)
    if m == 1:
        return term1, np.zeros_like(term1), np.square(f[0] - y)
    sf = np.sort(f, axis=0)
    coeff = (2.0 * np.arange(1, m + 1) - m - 1.0)[:, None]
    pair = 2.0 * np.sum(coeff * sf, axis=0)
    return term1 - pair / (2.0 * m * (m - 1)), np.var(f, axis=0, ddof=1), np.square(f.mean(0) - y)


# ---- area weights on the octahedral reduced Gaussian grid (review fix 3, 2 Nov 2026) ----
O_N = 2560   # o2560: 2 x 2560 latitude rings; ring i (1-based from the pole, per hemisphere) holds 4 i + 16 points


def octahedral_ring_counts(lat: np.ndarray, lon: np.ndarray, n: int = O_N, check: bool = True) -> np.ndarray:
    """Number of points on the FULL global ring of each point (4 i + 16), i from the latitude
    (ring i lies near 90 - (i - 0.5) * 90 / n degrees). With check=True the longitude spacing of the box points on every
    ring is verified to be 360 / (4 i + 16), so a wrong ring index fails instead of weighting silently."""
    i = np.rint((90.0 - np.abs(lat)) * n / 90.0 + 0.5).astype(np.int64)
    if (i < 1).any() or (i > n).any():
        raise ValueError("ring index outside 1..n")
    cnt = 4 * i + 16
    if check:
        ul, inv = np.unique(lat, return_inverse=True)
        for k in range(ul.size):
            sel = inv == k
            if sel.sum() < 3:
                continue
            d = np.diff(np.sort(np.mod(lon[sel], 360.0)))
            step = float(np.median(d))
            want = 360.0 / int(cnt[sel][0])
            if abs(step - want) > 1e-6 * want + 1e-9:
                raise ValueError(f"ring at lat {ul[k]:.5f}: lon spacing {step:.8f} != 360/{int(cnt[sel][0])} = {want:.8f}")
    return cnt


def area_weights(lat: np.ndarray, lon: np.ndarray, n: int = O_N, check: bool = True) -> np.ndarray:
    """Point area weight on the octahedral grid: cos(lat) / (points on the full ring). Summed over a full ring it is
    cos(lat); summing cos(lat) per point instead would over-weight the high-latitude rings, which hold fewer points."""
    return np.cos(np.deg2rad(lat)) / octahedral_ring_counts(lat, lon, n, check)


# ---- spread/RMSE ratio pooled over dates (review fix 2; rule 32 convention of eval/.../probabilistic/core/calibration.py) ----
RATIO_BOOTSTRAP_SEED = 32
RATIO_BOOTSTRAP_RESAMPLES = 2000


def spread_skill_corrected(mean_var: np.ndarray, mean_sq: np.ndarray, m: int) -> float:
    """sqrt((m+1)/m) * sqrt(mean over dates of the area-mean ensemble variance) / sqrt(mean over dates of the area-mean
    squared error of the ensemble mean). A calibrated ensemble reads 1."""
    return float(math.sqrt((m + 1.0) / m) * math.sqrt(max(float(np.mean(mean_var)), 0.0) / float(np.mean(mean_sq))))


def ratio_bootstrap(mean_var, mean_sq, m: int, mean_var_b=None, mean_sq_b=None, n_res: int = RATIO_BOOTSTRAP_RESAMPLES,
                    seed: int = RATIO_BOOTSTRAP_SEED, level: float = 0.95) -> dict:
    """95 % date-bootstrap interval of the corrected ratio; with a second row (same dates, same order) also the paired
    bootstrap of the difference first minus second (the same resampled dates for both rows)."""
    mv, ms = np.asarray(mean_var, float), np.asarray(mean_sq, float)
    n = mv.size
    out = {"ratio": spread_skill_corrected(mv, ms, m), "n_dates": n}
    if n < 2:
        return {**out, "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_res, n))
    c = math.sqrt((m + 1.0) / m)
    r = c * np.sqrt(np.maximum(mv[idx].mean(1), 0) / ms[idx].mean(1))
    a = (1 - level) / 2
    out.update(lo=float(np.quantile(r, a)), hi=float(np.quantile(r, 1 - a)))
    if mean_var_b is not None:
        vb, sb = np.asarray(mean_var_b, float), np.asarray(mean_sq_b, float)
        rb = c * np.sqrt(np.maximum(vb[idx].mean(1), 0) / sb[idx].mean(1))
        d = r - rb
        out.update(ratio_b=spread_skill_corrected(vb, sb, m), diff=out["ratio"] - spread_skill_corrected(vb, sb, m),
                   diff_lo=float(np.quantile(d, a)), diff_hi=float(np.quantile(d, 1 - a)))
    return out
