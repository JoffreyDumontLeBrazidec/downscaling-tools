"""o2560 Europe-box read kit (2 Nov 2026): helpers shared by the predictor, the scoring scripts and the tests.

No torch and no anemoi import here, so the tests run on a login node.

Rows of the read (one checkpoint each, all on the o1280 -> o2560 Europe box E2, 10-70 N, 80 W-40 E):
  A100  db_o2560_crop_100k (AG crop chain) at step 100,000, EMA-10000, code 4ad32838c
  B24   db_o2560_e2_50k_v2 (Jupiter full-box chain) at step 24,000, EMA-10000, code 20ba15241 (equal GPU-hours with A100)
  B50   db_o2560_e2_50k_v2 at step 50,000, EMA-10000, code 20ba15241 (its last step)
Input reference: the o1280 operational IFS input interpolated to o2560 (x_interp in every file), scored as a one-member row.
"""
from __future__ import annotations

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
    """f (m, p), y (p,): pointwise fair CRPS, spread (ddof 1), squared ensemble-mean error.
    With one member (the input row IN): mean absolute error, zero spread, squared error."""
    m = f.shape[0]
    term1 = np.mean(np.abs(f - y[None]), axis=0)
    if m == 1:
        return term1, np.zeros_like(term1), np.square(f[0] - y)
    sf = np.sort(f, axis=0)
    coeff = (2.0 * np.arange(1, m + 1) - m - 1.0)[:, None]
    pair = 2.0 * np.sum(coeff * sf, axis=0)
    return term1 - pair / (2.0 * m * (m - 1)), np.std(f, axis=0, ddof=1), np.square(f.mean(0) - y)
