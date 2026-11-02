"""Scores of forecasts against station observations. Ensemble scores first (Joffrey, 2 Nov 2026).

Every function takes station-aligned arrays: ``ens`` (M, n) for an ensemble, ``f`` (n,) for one field, ``obs`` (n,).
"""
from __future__ import annotations

import numpy as np


def fair_crps(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS per station (Ferro 2014): mean |x_i - y| - sum_ij |x_i - x_j| / (2 M (M - 1))."""
    ens = np.asarray(ens, np.float64)
    m = ens.shape[0]
    t1 = np.abs(ens - obs[None]).mean(0)
    if m < 2:
        return t1
    s = np.sort(ens, axis=0)
    w = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)[:, None]
    return t1 - (w * s).sum(0) / (m * (m - 1))


def fair_twcrps(ens: np.ndarray, obs: np.ndarray, t: float) -> np.ndarray:
    """Fair threshold-weighted CRPS above t: the fair CRPS of max(x, t) against max(y, t)
    (checked equal to scores 2.7.0, docs epics/misc/20261101_earthkit_assessment_for_obs_eval.md)."""
    return fair_crps(np.maximum(ens, t), np.maximum(obs, t))


def fair_brier(ens: np.ndarray, obs: np.ndarray, t: float) -> np.ndarray:
    """Fair Brier score of exceeding t: (p - o)^2 - p (1 - p) / (M - 1); plain Brier for M = 1."""
    ens = np.asarray(ens, np.float64)
    m = ens.shape[0]
    p = (ens > t).mean(0)
    o = (obs > t).astype(np.float64)
    return (p - o) ** 2 - (p * (1 - p) / (m - 1) if m > 1 else 0.0)


def rank_counts(ens: np.ndarray, obs: np.ndarray, seed: int = 31) -> np.ndarray:
    """Rank histogram counts (M + 1 bins) of the observation among the members, ties broken at random."""
    ens = np.asarray(ens, np.float64)
    m = ens.shape[0]
    rng = np.random.default_rng(seed)
    below = (ens < obs[None]).sum(0)
    ties = (ens == obs[None]).sum(0)
    rank = below + np.floor(rng.random(obs.size) * (ties + 1)).astype(int)
    return np.bincount(rank, minlength=m + 1)


def ensemble_scores(ens: np.ndarray, obs: np.ndarray, thresholds=()) -> dict:
    """Ensemble scores of one sample of stations. Single fields go in as ens of shape (1, n):
    fair CRPS is then the MAE and the spread is 0, so they stay on the same columns, labelled single field."""
    ens = np.asarray(ens, np.float64)
    obs = np.asarray(obs, np.float64)
    m = ens.shape[0]
    mean = ens.mean(0)
    d = mean - obs
    rmse = float(np.sqrt((d ** 2).mean()))
    spread_rms = float(np.sqrt(ens.var(0, ddof=1).mean())) if m > 1 else 0.0
    out = {
        "n": int(obs.size), "members": int(m),
        "fcrps": float(fair_crps(ens, obs).mean()),
        "spread_rms": spread_rms,
        "ssr_corrected": spread_rms / rmse * np.sqrt((m + 1) / m) if m > 1 and rmse > 0 else np.nan,
        "bias": float(d.mean()),
        "rmse_mean": rmse,
        "mae_mean": float(np.abs(d).mean()),
        "mae_member": float(np.abs(ens - obs[None]).mean()),
        "corr_mean": float(np.corrcoef(mean, obs)[0, 1]) if mean.std() > 0 and obs.std() > 0 else np.nan,
    }
    for t in thresholds:
        out[f"n_obs_gt{t:g}"] = int((obs > t).sum())
        out[f"twcrps{t:g}"] = float(fair_twcrps(ens, obs, t).mean())
        out[f"brier{t:g}"] = float(fair_brier(ens, obs, t).mean())
    return out
