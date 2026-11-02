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


def seeps(fc: np.ndarray, obs: np.ndarray, t1, t2, smat) -> np.ndarray:
    """SEEPS (Rodwell et al. 2010) of each member at each station, as quaver computes it
    (ScoresLibAux.seepsField): forecast and observation rounded to 0.1 mm, categories dry (<= t1), light and
    heavy (>= t2), and the station's 3 x 3 error matrix ``smat`` (n, 9) indexed by forecast category x 3 +
    observed category. Lower is better; quaver stores 1 - SEEPS. Returns (M, n).

    Evidence (2 Nov 2026): quaver 3.8.2, quaver/ScoresLibAux.py, ScoresLibAux.seepsField, lines 83-108:
    fc and ob are ``DataArray.round(1)`` (numpy rounding, half to even, as np.round here); t1 = climatology
    field number 1, t2 = number 3, the matrix = numbers 4..12; category = (x > t1) + (x >= t2); index =
    fc_ind * 3 + ob_ind. Scored with seepsField itself on all 5,116 stations of the 24 h climatology ending
    2023-08-30 00 UTC, with edge cases at t1, t2 and the 0.05 mm rounding: max |seepsField - this| = 0;
    the transposed index (ob * 3 + fc) differs by up to 49. Script:
    /ec/res4/scratch/ecm5702/agent-work/20261102-obs-eval/seeps_vs_quaver.py (Atos, quaver's environment)."""
    fc = np.round(np.atleast_2d(np.asarray(fc, np.float64)), 1)
    ob = np.round(np.asarray(obs, np.float64), 1)
    t1, t2, smat = np.asarray(t1, np.float64), np.asarray(t2, np.float64), np.asarray(smat, np.float64)
    oc = (ob > t1).astype(int) + (ob >= t2).astype(int)
    fcat = (fc > t1).astype(int) + (fc >= t2).astype(int)
    idx = fcat * 3 + oc[None]
    return np.take_along_axis(smat.T, idx, axis=0)


def _wmean(x, w):
    return float(np.mean(x)) if w is None else float(np.sum(w * x))


def ensemble_scores(ens: np.ndarray, obs: np.ndarray, thresholds=(), weights=None) -> dict:
    """Ensemble scores of one sample of stations. Single fields go in as ens of shape (1, n):
    fair CRPS is then the MAE and the spread is 0, so they stay on the same columns, labelled single field.

    Headline columns: fcrps, spread (mean member SD, ddof 1), spread_rms (root of the mean member variance),
    ssr_corrected = spread_rms / rmse_mean * sqrt((M + 1) / M), and per threshold t the fair twCRPS, the fair
    Brier score and the observed exceedance count n_obs_gt<t>. Secondary (deterministic) columns: bias,
    rmse_mean, mae_mean, corr_mean of the ensemble mean, mae_member. ``weights`` (summing to one, e.g.
    quaver's station-density weights) turn every station mean into a weighted mean; counts stay counts.
    """
    ens = np.asarray(ens, np.float64)
    obs = np.asarray(obs, np.float64)
    w = None if weights is None else np.asarray(weights, np.float64) / np.sum(weights)
    m = ens.shape[0]
    mean = ens.mean(0)
    d = mean - obs
    rmse = float(np.sqrt(_wmean(d ** 2, w)))
    spread_rms = float(np.sqrt(_wmean(ens.var(0, ddof=1), w))) if m > 1 else 0.0
    if w is None:
        corr = float(np.corrcoef(mean, obs)[0, 1]) if mean.std() > 0 and obs.std() > 0 else np.nan
    else:
        a, b = mean - np.sum(w * mean), obs - np.sum(w * obs)
        den = np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))
        corr = float(np.sum(w * a * b) / den) if den > 0 else np.nan
    out = {
        "n": int(obs.size), "members": int(m),
        "fcrps": _wmean(fair_crps(ens, obs), w),
        "spread": _wmean(ens.std(0, ddof=1), w) if m > 1 else 0.0,
        "spread_rms": spread_rms,
        "ssr_corrected": spread_rms / rmse * np.sqrt((m + 1) / m) if m > 1 and rmse > 0 else np.nan,
    }
    for t in thresholds:
        out[f"n_obs_gt{t:g}"] = int((obs > t).sum())
        out[f"twcrps{t:g}"] = _wmean(fair_twcrps(ens, obs, t), w)
        out[f"brier{t:g}"] = _wmean(fair_brier(ens, obs, t), w)
    out.update({
        "bias": _wmean(d, w),
        "rmse_mean": rmse,
        "mae_mean": _wmean(np.abs(d), w),
        "mae_member": _wmean(np.abs(ens - obs[None]).mean(0), w),
        "corr_mean": corr,
    })
    return out
