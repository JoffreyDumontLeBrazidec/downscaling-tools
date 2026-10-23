"""Score primitives for precip_read, free of I/O. All values in mm per accumulation window."""
from __future__ import annotations

import numpy as np

EDGES = np.concatenate([[0.0], np.logspace(-1, 3, 81)])  # bin 0 = [0, 0.1) mm, then 0.05 dex to 1000 mm


def fair_crps(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS per point of an ensemble (members, points) against obs (points).

    mean|x_i - y| - sum_{i != j} |x_i - x_j| / (2 M (M - 1)), with the pair sum taken from the
    sorted members: sum_{i < j} |x_i - x_j| = sum_k (2k - M + 1) x_(k).
    """
    m = ens.shape[0]
    if m < 2:
        return np.abs(ens[0] - obs)
    s = np.sort(ens, axis=0)
    w = (2 * np.arange(m) - m + 1).reshape(-1, *([1] * (ens.ndim - 1)))
    return np.abs(ens - obs).mean(0) - (w * s).sum(0) / (m * (m - 1))


def score_sums(ens: np.ndarray, inp: np.ndarray, truth: np.ndarray, thresholds) -> dict:
    """Additive sums over one region of one field, so dates can be pooled or resampled later."""
    em = ens.mean(0)
    out = {
        "n": int(truth.size), "members": int(ens.shape[0]),
        "s_t": truth.sum(), "s_tt": (truth * truth).sum(),
        "s_i": inp.sum(), "s_ii": (inp * inp).sum(), "s_it": (inp * truth).sum(),
        "s_e": em.sum(), "s_ee": (em * em).sum(), "s_et": (em * truth).sum(),
        "se_ens_mean": ((em - truth) ** 2).sum(), "se_input": ((inp - truth) ** 2).sum(),
        "se_member": ((ens - truth) ** 2).mean(0).sum(),
        "crps_fair": fair_crps(ens, truth).sum(), "ae_input": np.abs(inp - truth).sum(),
        "spread_var": (ens.var(0, ddof=1).sum() if ens.shape[0] > 1 else 0.0),
    }
    for t in thresholds:
        out[f"n_truth_ge{t:g}"] = int((truth >= t).sum())
        out[f"n_input_ge{t:g}"] = int((inp >= t).sum())
        out[f"n_model_ge{t:g}"] = float((ens >= t).sum(1).mean())  # mean over members
    return out


def pooled(rows: list[dict]) -> dict:
    """bias, normalised RMSE, correlation and CRPS from summed rows (one or more dates)."""
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("lead", "date", "region", "members")}
    n = tot["n"]
    mt = tot["s_t"] / n
    sd_t = np.sqrt(max(tot["s_tt"] / n - mt * mt, 0.0))

    def corr(a, aa, at):
        ma = tot[a] / n
        va = tot[aa] / n - ma * ma
        cov = tot[at] / n - ma * mt
        return cov / np.sqrt(va * sd_t ** 2) if va > 0 and sd_t > 0 else np.nan

    return {
        "n": n, "truth_mean": mt, "truth_sd": sd_t,
        "bias_model": tot["s_e"] / n - mt, "bias_input": tot["s_i"] / n - mt,
        "nrmse_model": np.sqrt(tot["se_ens_mean"] / n) / sd_t if sd_t > 0 else np.nan,
        "nrmse_member": np.sqrt(tot["se_member"] / n) / sd_t if sd_t > 0 else np.nan,
        "nrmse_input": np.sqrt(tot["se_input"] / n) / sd_t if sd_t > 0 else np.nan,
        "corr_model": corr("s_e", "s_ee", "s_et"), "corr_input": corr("s_i", "s_ii", "s_it"),
        "crps_model": tot["crps_fair"] / n, "crps_input": tot["ae_input"] / n,
        "spread": np.sqrt(tot["spread_var"] / n), "rmse_ens_mean": np.sqrt(tot["se_ens_mean"] / n),
    }
