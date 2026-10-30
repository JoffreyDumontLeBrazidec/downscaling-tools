"""Local spread/CRPS scoring for eval.cli prediction NetCDFs.

What is computed, per prediction file (one date and lead), variable and region:

* the four historical metrics, unchanged in name and meaning (``scores_by_lead.csv``,
  ``summary_by_lead.csv``): ``crps``, ``fcrps`` (Ferro's fair CRPS), ``spread`` (area mean of
  the pointwise member standard deviation, ddof 1) and ``rmse_ens_mean`` (root of the area
  mean squared error of the ensemble mean);
* the calibration read of the owner's rule (``calibration_by_lead.csv``, see
  ``calibration.py``): spread as the root of the pooled mean ensemble variance, RMSE as the
  root of the pooled mean squared error, their ratio and the ratio times ``sqrt((m+1)/m)``,
  pooled over points (area-weighted) and dates, with ``m``, ``n_dates`` and ``n_points``, and
  the 95 % bootstrap interval of the corrected ratio over dates (``_lo95``, ``_hi95``, ``_se``;
  fixed seed, 2000 resamples by default);
* rank histograms with ``m+1`` bins, ties broken at random with a fixed seed
  (``rank_histograms.csv``), the two outer-bin fractions reported separately; precipitation
  gets a dry floor first (members and truth below ``rank_dry_threshold_mm``, default 0.1 mm,
  set to exactly 0, so trace amounts over a dry truth are ties), and its false-drizzle rate
  (truth dry, every member wet) and missed-rain rate (truth wet, every member dry) are
  written with the histogram;
* when ``scale_split`` is on, the same scores for the large part ``L = P_L(field)`` and the
  fine part ``F = field - L`` of members and truth (``band_scores_by_lead.csv``, band column
  ``full`` / ``full_inner`` / ``large`` / ``fine``; ``scale_split.py`` describes ``P_L``), and
  the inter-member spread of the low-resolution input in each band when the files carry the
  input members (``x_interp`` with two or more members), so the read can say how much of the
  large-scale spread comes from the input. ``full`` is the full field on every point of the
  region (minus the box-edge strip); ``full_inner`` is the full field on the SAME points the
  ``large`` and ``fine`` bands use (inside the edge margin of ``P_L``), so the three are
  comparable. The bands are NOT additive: ``P_L`` is a taper, so a score of ``full_inner`` is
  not the sum of ``large`` and ``fine``, and the band labels are never added up.

* when ``scale_split`` is on and the files carry the input on the output grid (``x_interp``, one
  member or one per prediction member), the residuals against the input split the same way
  (``residual_bands_by_lead.csv``, bands ``full_inner`` / ``large`` / ``fine``): how far the
  truth is from the input in each band (``truth_minus_input_rms``: in the large band the
  synoptic drift between the two forecasts, ablation note section 15), how far the model moves
  from it (``model_minus_input_rms`` per member, ``ens_mean_minus_input_rms`` for the ensemble
  mean), the ratio of the two, the uncentred correlation of the ensemble-mean residual with the
  truth's (``residual_corr``: near 1 the model's change goes towards the truth, a correction;
  near 0 it is drift of its own), the band's share of the truth's residual, and the
  ensemble-mean error over the input's error (``error_over_input``, below 1 the model beats
  the input in that band). It separates the two jobs the model does at once, large-scale
  correction and fine-scale detail; it is a read only, the outputs are never filtered
  (owner's ruling, 30 Oct 2026, ablation note section 15.10).

House conventions. Precipitation (``tp``, ``cp``) is clipped at zero in members and truth
before any score: a sampler can write small negative rain, and the scores are defined on
the physical field (``PRECIP_FIELDS``). The files hold tp and cp in METRES of accumulated
water (the native unit, ``eval/plotting/variables.py``), so the dry threshold given in mm is
divided by ``DRY_THRESHOLD_MM_TO_NATIVE`` (1000) before use; the floor applies to the rank
histograms and the drizzle rates only, never to the CRPS, spread or RMSE. The low-pass of rain is a band split of the SCORES
only (the L part is the synoptic-scale rain mass, the F part its placement and intensity);
no filtered field is written back, so the negative values a Gaussian of rain makes in the F
part are harmless and nothing is clipped after the split. ``10ff`` is derived from ``10u``
and ``10v`` when the file has no wind speed. The truth is member 0 of ``y``.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from eval.evaluators.surface.core.compute import (
    _area_weights,
    _to_member_point_weather,
    _weather_state_index,
)
from eval.discovery.predictions import find_predictions

from .calibration import (
    DEFAULT_RANK_DRY_THRESHOLD_MM,
    RANK_TIE_SEED,
    RATIO_BOOTSTRAP_RESAMPLES,
    RATIO_BOOTSTRAP_SEED,
    PooledMoments,
    dry_floor,
    dry_wet_disagreement,
    histogram_shape,
    rank_counts,
    summarize_histogram,
    tie_rng,
    truth_ranks,
)
from .scale_split import DEFAULT_CUT_KM, DEFAULT_GRID_KM, LowPassOperator, box_edge_mask, inner_mask

# Strip dropped at a regional box edge in every band, as in the wide box kits (degrees).
DEFAULT_BOX_EDGE_DEG = 1.0

LOG = logging.getLogger(__name__)

# tp is in every run (owner, 30 Oct 2026); a field absent from a file is skipped with a warning.
DEFAULT_WEATHER_STATES = ["2t", "10ff", "tp", "2d", "msl", "t_850", "z_500"]
DEFAULT_DOMAINS = ["n.hem", "tropics", "s.hem", "europe"]
METRICS = ("crps", "fcrps", "spread", "rmse_ens_mean")
# clipped at zero before scoring (house convention, see module docstring)
PRECIP_FIELDS = ("tp", "cp")
# tp/cp are stored in metres; the dry threshold is given in mm (see module docstring)
DRY_THRESHOLD_MM_TO_NATIVE = 1000.0
# full: every point of the region; full_inner: the points of the large/fine bands (not additive, see docstring)
BANDS = ("full", "full_inner", "large", "fine")
INNER_BANDS = ("full_inner", "large", "fine")
BAND_METRICS = ("fcrps", "crps", "spread_rms", "rmse_ens_mean", "ms_spread", "ms_err")
INPUT_VARIABLE = "x_interp"
RESIDUAL_BANDS = INNER_BANDS
RESIDUAL_COLUMNS = ("step", "weather_state", "domain", "band", "n_members", "n_input_members", "n_dates",
                    "n_points_total", "truth_minus_input_rms", "model_minus_input_rms", "ens_mean_minus_input_rms",
                    "amplitude_ratio", "ens_mean_amplitude_ratio", "residual_corr", "share_of_truth_residual",
                    "rmse_ens_mean", "error_over_input")


def _as_list(value: Any, *, cast=str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value:
            return []
        return [cast(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [cast(v) for v in value]
    return [cast(value)]


def _grid_point_values(da: xr.DataArray, n_points: int) -> np.ndarray | None:
    if da.size == n_points:
        return np.asarray(da.values, dtype=np.float64).reshape(-1)
    if "grid_point_hres" in da.dims and int(da.sizes["grid_point_hres"]) == n_points:
        indexers = {dim: 0 for dim in da.dims if dim != "grid_point_hres"}
        return np.asarray(da.isel(**indexers).values, dtype=np.float64).reshape(-1)
    return None


def _lat_lon(ds: xr.Dataset, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    if "lat_hres" not in ds or "lon_hres" not in ds:
        raise ValueError("Dataset is missing lat_hres/lon_hres needed for domain masks")
    lat = _grid_point_values(ds["lat_hres"], n_points)
    lon = _grid_point_values(ds["lon_hres"], n_points)
    if lat is None or lon is None:
        raise ValueError("Could not align lat_hres/lon_hres with grid_point_hres")
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lat, lon


def _domain_mask(name: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    key = name.lower()
    if key in {"global", "all"}:
        return np.ones(lat.shape, dtype=bool)
    if key in {"n.hem", "nhem", "north_hemisphere"}:
        return lat >= 20.0
    if key == "tropics":
        return (lat > -20.0) & (lat < 20.0)
    if key in {"s.hem", "shem", "south_hemisphere"}:
        return lat <= -20.0
    if key == "europe":
        return (lat >= 35.0) & (lat <= 75.0) & (lon >= -25.0) & (lon <= 45.0)
    raise ValueError(f"Unknown probabilistic domain: {name!r}")


def _select_field(da: xr.DataArray, ws_index: dict[str, int], field: str) -> np.ndarray | None:
    if field in ws_index:
        return np.asarray(da.isel(weather_state=ws_index[field]).values, dtype=np.float64)
    if field == "10ff":
        if "10u" not in ws_index or "10v" not in ws_index:
            return None
        u = np.asarray(da.isel(weather_state=ws_index["10u"]).values, dtype=np.float64)
        v = np.asarray(da.isel(weather_state=ws_index["10v"]).values, dtype=np.float64)
        return np.hypot(u, v)
    return None


def _clip_precip(field: str, values: np.ndarray | None) -> np.ndarray | None:
    """Negative precipitation is set to zero before scoring (house convention)."""
    if values is None or field not in PRECIP_FIELDS:
        return values
    return np.where(np.isfinite(values), np.maximum(values, 0.0), values)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights)
    if not np.any(finite):
        return math.nan
    w = weights[finite]
    total = float(w.sum())
    if total <= 0.0:
        return math.nan
    return float(np.sum(values[finite] * w) / total)


def crps_ensemble_components(
    forecasts: np.ndarray,
    truth: np.ndarray,
    *,
    spread_ddof: int = 1,
) -> dict[str, np.ndarray]:
    """Return pointwise CRPS, fair CRPS, spread, and ensemble-mean RMSE terms.

    Parameters use shape ``(member, point)`` for forecasts and ``(point,)`` for
    truth. Points with non-finite truth or any non-finite ensemble member are
    marked NaN in all returned arrays.
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if forecasts.ndim != 2:
        raise ValueError(f"forecasts must have shape (member, point), got {forecasts.shape}")
    if forecasts.shape[1] != truth.size:
        raise ValueError(
            f"forecast/truth point mismatch: {forecasts.shape[1]} vs {truth.size}"
        )

    member_count = int(forecasts.shape[0])
    valid = np.isfinite(truth) & np.all(np.isfinite(forecasts), axis=0)
    out = {name: np.full(truth.shape, np.nan, dtype=np.float64) for name in METRICS}
    if member_count <= 0 or not np.any(valid):
        return out

    f = forecasts[:, valid]
    y = truth[valid]
    term1 = np.mean(np.abs(f - y[np.newaxis, :]), axis=0)

    sorted_f = np.sort(f, axis=0)
    ranks = np.arange(1, member_count + 1, dtype=np.float64)
    coeff = (2.0 * ranks - member_count - 1.0)[:, np.newaxis]
    pair_sum = 2.0 * np.sum(coeff * sorted_f, axis=0)

    out["crps"][valid] = term1 - pair_sum / (2.0 * member_count * member_count)
    if member_count > 1:
        out["fcrps"][valid] = term1 - pair_sum / (2.0 * member_count * (member_count - 1))
        ddof = min(max(int(spread_ddof), 0), member_count - 1)
    else:
        out["fcrps"][valid] = np.nan
        ddof = 0
    out["spread"][valid] = np.std(f, axis=0, ddof=ddof)
    out["rmse_ens_mean"][valid] = np.square(np.mean(f, axis=0) - y)
    return out


def _pointwise_variance(members: np.ndarray) -> np.ndarray:
    """Ensemble variance per point, ddof 1 (the quantity the calibration pools), NaN where a member is."""
    members = np.asarray(members, dtype=np.float64)
    if members.shape[0] < 2:
        return np.full(members.shape[1], np.nan)
    out = np.var(members, axis=0, ddof=1)
    out[~np.all(np.isfinite(members), axis=0)] = np.nan
    return out


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    n_points: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = (row["step"], row["weather_state"], row["domain"], row["metric"])
        value = float(row["value"])
        if math.isfinite(value):
            grouped[key].append(value)
            n_points[key] += int(row.get("n_points", 0))

    summaries: list[dict[str, Any]] = []
    for (step, weather_state, domain, metric), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stderr = float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        summaries.append({
            "step": int(step),
            "weather_state": weather_state,
            "domain": domain,
            "metric": metric,
            "mean": mean,
            "std": std,
            "stderr": stderr,
            "n_dates": int(arr.size),
            "n_points_total": int(n_points[(step, weather_state, domain, metric)]),
        })
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _headline_metrics(summary_rows: list[dict[str, Any]],
                      calibration_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        key = (str(row["weather_state"]), str(row["domain"]), str(row["metric"]))
        value = float(row["mean"])
        if math.isfinite(value):
            grouped[key].append(value)
    for (weather_state, domain, metric), values in sorted(grouped.items()):
        metrics[f"probabilistic_{weather_state}_{domain}_{metric}_mean"] = float(np.mean(values))
    # the corrected spread/RMSE of the full field, mean over leads (a calibrated ensemble reads 1)
    ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in calibration_rows or []:
        value = float(row.get("spread_skill_ratio_corrected", math.nan))
        if row.get("band") == "full" and math.isfinite(value):
            ratios[(str(row["weather_state"]), str(row["domain"]))].append(value)
    for (weather_state, domain), values in sorted(ratios.items()):
        metrics[f"probabilistic_{weather_state}_{domain}_spread_skill_ratio_corrected_mean"] = float(np.mean(values))
    return metrics


# --------------------------------------------------------------------------- bands

def _band_fields(
    pred_values: np.ndarray,
    truth_point: np.ndarray,
    input_values: np.ndarray | None,
    operator: LowPassOperator | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """``{band: (members, truth, input members or None)}``; only ``full`` without an operator.

    ``full_inner`` holds the same arrays as ``full``: it differs only in the points scored
    (the inner mask of the large and fine bands), so its CRPS components are reused.
    """
    bands = {"full": (pred_values, truth_point, input_values)}
    if operator is None:
        return bands
    bands["full_inner"] = bands["full"]
    large_pred = np.stack([operator.apply(member) for member in pred_values])
    large_truth = operator.apply(truth_point)
    large_input = None
    if input_values is not None:
        large_input = np.stack([operator.apply(member) for member in input_values])
    bands["large"] = (large_pred, large_truth, large_input)
    bands["fine"] = (pred_values - large_pred, truth_point - large_truth,
                     None if input_values is None else input_values - large_input)
    return bands


def _band_components(
    members: np.ndarray,
    truth: np.ndarray,
    input_members: np.ndarray | None,
    *,
    spread_ddof: int,
) -> dict[str, Any]:
    """The pointwise pieces of one band (computed once, scored on any set of masks)."""
    components = crps_ensemble_components(members, truth, spread_ddof=spread_ddof)
    variance = np.square(components["spread"]) if int(spread_ddof) == 1 else _pointwise_variance(members)
    input_variance = _pointwise_variance(input_members) if input_members is not None else None
    valid = np.isfinite(truth) & np.all(np.isfinite(members), axis=0)
    return {"components": components, "variance": variance, "input_variance": input_variance, "valid": valid}


def _score_band(
    pieces: dict[str, Any],
    *,
    weights: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Per domain: the band metrics and the pooled moments of one date (``_weighted_mean`` of each)."""
    components, variance = pieces["components"], pieces["variance"]
    input_variance, valid = pieces["input_variance"], pieces["valid"]
    out: dict[str, dict[str, Any]] = {}
    for domain_name, domain_mask in masks.items():
        mask = valid & domain_mask
        if not np.any(mask):
            continue
        w = np.where(mask, weights, 0.0)
        ms_spread = _weighted_mean(variance, w)
        ms_err = _weighted_mean(components["rmse_ens_mean"], w)
        values = {
            "fcrps": _weighted_mean(components["fcrps"], w),
            "crps": _weighted_mean(components["crps"], w),
            "ms_spread": ms_spread,
            "ms_err": ms_err,
            "spread_rms": math.sqrt(ms_spread) if math.isfinite(ms_spread) else math.nan,
            "rmse_ens_mean": math.sqrt(ms_err) if math.isfinite(ms_err) else math.nan,
            "n_points": int(mask.sum()),
            "components": components,
            "mask": mask,
        }
        if input_variance is not None:
            ms_input = _weighted_mean(input_variance, w)
            values["ms_spread_input"] = ms_input
            values["spread_input_rms"] = math.sqrt(ms_input) if math.isfinite(ms_input) else math.nan
        out[domain_name] = values
    return out


def _input_members(ds: xr.Dataset, include_input: bool) -> tuple[xr.DataArray | None, dict[str, Any]]:
    """The low-resolution input members on the output grid, and a note for the summary."""
    if not include_input:
        return None, {"variable": INPUT_VARIABLE, "n_members": 0, "note": "input spread not requested"}
    if INPUT_VARIABLE not in ds:
        return None, {"variable": INPUT_VARIABLE, "n_members": 0,
                      "note": f"files carry no {INPUT_VARIABLE}: input spread not reported"}
    da = _to_member_point_weather(ds[INPUT_VARIABLE], ds, label=INPUT_VARIABLE)
    n = int(da.sizes["member"])
    if n < 2:
        return None, {"variable": INPUT_VARIABLE, "n_members": n,
                      "note": "files carry one input member only: input spread not reported"}
    return da, {"variable": INPUT_VARIABLE, "n_members": n,
                "note": "inter-member spread of the input members per band (spread_input_rms)"}


def _input_reference(ds: xr.Dataset, n_pred: int) -> tuple[xr.DataArray | None, str]:
    """The input on the output grid for the residuals: one member, or one per prediction member."""
    if INPUT_VARIABLE not in ds:
        return None, f"files carry no {INPUT_VARIABLE}: no residual bands"
    da = _to_member_point_weather(ds[INPUT_VARIABLE], ds, label=INPUT_VARIABLE)
    n = int(da.sizes["member"])
    if n not in (1, n_pred):
        return None, f"{INPUT_VARIABLE} has {n} members for {n_pred} prediction members: no residual bands"
    return da, ("one input for every member" if n == 1 else "input member k paired with prediction member k")


class PooledResiduals:
    """Area-weighted mean squares of the residuals against the input, pooled over dates (each date once)."""

    _KEYS = ("truth", "model", "ens_mean", "truth_mean", "cross", "error")

    def __init__(self) -> None:
        self.sums = dict.fromkeys(self._KEYS, 0.0)
        self.n_dates = 0
        self.n_points_total = 0
        self.n_members = 0
        self.n_input_members = 0

    def add_date(self, values: dict[str, float], *, n_points: int, n_members: int, n_input_members: int) -> None:
        if not all(math.isfinite(values[k]) for k in self._KEYS):
            return
        for k in self._KEYS:
            self.sums[k] += float(values[k])
        self.n_dates += 1
        self.n_points_total += int(n_points)
        self.n_members, self.n_input_members = int(n_members), int(n_input_members)

    def summary(self, full_truth_ms: float | None) -> dict[str, Any]:
        m = {k: v / self.n_dates for k, v in self.sums.items()} if self.n_dates else dict.fromkeys(self._KEYS, math.nan)

        def ratio(a: float, b: float) -> float:
            return a / b if b > 0 and math.isfinite(a) else math.nan

        den = math.sqrt(m["ens_mean"] * m["truth_mean"]) if m["ens_mean"] >= 0 and m["truth_mean"] >= 0 else math.nan
        return {
            "n_members": self.n_members, "n_input_members": self.n_input_members,
            "n_dates": self.n_dates, "n_points_total": self.n_points_total,
            "truth_minus_input_rms": math.sqrt(m["truth"]),
            "model_minus_input_rms": math.sqrt(m["model"]),
            "ens_mean_minus_input_rms": math.sqrt(m["ens_mean"]),
            "amplitude_ratio": ratio(math.sqrt(m["model"]), math.sqrt(m["truth"])),
            "ens_mean_amplitude_ratio": ratio(math.sqrt(m["ens_mean"]), math.sqrt(m["truth_mean"])),
            "residual_corr": ratio(m["cross"], den),
            "share_of_truth_residual": ratio(m["truth"], full_truth_ms) if full_truth_ms is not None else math.nan,
            "rmse_ens_mean": math.sqrt(m["error"]),
            "error_over_input": ratio(math.sqrt(m["error"]), math.sqrt(m["truth_mean"])),
        }

    @property
    def mean_truth(self) -> float:
        return self.sums["truth"] / self.n_dates if self.n_dates else math.nan


def _residual_band_values(
    pred_values: np.ndarray,
    truth_point: np.ndarray,
    ref_values: np.ndarray,
    operator: LowPassOperator,
    large_pred: np.ndarray,
    large_truth: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{band: (model residuals (m, n), truth residuals (k, n))}`` against the input, per inner band.

    ``ref_values`` holds one input (k = 1, used for every member) or one per member (k = m).
    The large part of a residual is ``P_L(pred) - P_L(input)``: ``P_L`` is linear, so the
    members' large parts already computed for the band scores are reused.
    """
    large_ref = np.stack([operator.apply(r) for r in ref_values])
    full_m, full_t = pred_values - ref_values, truth_point[None] - ref_values
    large_m, large_t = large_pred - large_ref, large_truth[None] - large_ref
    return {"full_inner": (full_m, full_t), "large": (large_m, large_t),
            "fine": (full_m - large_m, full_t - large_t)}


def _residual_moments(model_res: np.ndarray, truth_res: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Area-weighted mean squares of one date (``w`` zero outside the mask)."""
    ens = model_res.mean(axis=0)
    tru = truth_res.mean(axis=0)
    return {
        "truth": float(np.mean([_weighted_mean(np.square(t), w) for t in truth_res])),
        "model": float(np.mean([_weighted_mean(np.square(r), w) for r in model_res])),
        "ens_mean": _weighted_mean(np.square(ens), w),
        "truth_mean": _weighted_mean(np.square(tru), w),
        "cross": _weighted_mean(ens * tru, w),
        "error": _weighted_mean(np.square(ens - tru), w),
    }


# ---------------------------------------------------------------------- main entry

def compute_probabilistic_scores(
    predictions_dir: str | Path,
    output_dir: str | Path,
    *,
    weather_states: Iterable[str] | str | None = None,
    domains: Iterable[str] | str | None = None,
    steps: Iterable[int] | str | None = None,
    dates: Iterable[str] | str | None = None,
    spread_ddof: int = 1,
    scale_split: bool = False,
    scale_cut_km: float = DEFAULT_CUT_KM,
    scale_grid_km: float = DEFAULT_GRID_KM,
    scale_edge_margin_km: float | None = None,
    box_edge_deg: float = DEFAULT_BOX_EDGE_DEG,
    rank_histogram: bool = True,
    rank_tie_seed: int = RANK_TIE_SEED,
    rank_dry_threshold_mm: float = DEFAULT_RANK_DRY_THRESHOLD_MM,
    ratio_bootstrap_resamples: int = RATIO_BOOTSTRAP_RESAMPLES,
    ratio_bootstrap_seed: int = RATIO_BOOTSTRAP_SEED,
    include_input: bool = True,
) -> dict[str, Any]:
    """Compute local probabilistic scores and write CSV/JSON artifacts.

    ``scale_split`` adds the large/fine band scores (``P_L`` with half-power wavelength
    ``scale_cut_km``, built on a ``scale_grid_km`` grid; points within
    ``scale_edge_margin_km`` of a regional box edge, default the cut, are left out of the
    band scores) and the ``full_inner`` band, the full field on those same inner points. On
    a regional box every band, the full field included, also leaves out a ``box_edge_deg``
    strip at the box edge (default 1 degree, as the wide box kits do; the edge is where the
    models differ most, so keeping it inflates gaps). It costs about three times the
    full-field scoring (eleven low-pass applications and two more CRPS passes per variable;
    ``full_inner`` reuses the full field's), so it is off unless asked for.

    ``rank_dry_threshold_mm`` (0.1) is the dry floor of the precipitation rank histograms
    and of the false-drizzle / missed-rain rates, in mm (the files hold metres; converted
    here). ``ratio_bootstrap_resamples`` / ``ratio_bootstrap_seed`` fix the bootstrap of the
    corrected spread/RMSE over dates.
    """
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = _as_list(weather_states, cast=str) or list(DEFAULT_WEATHER_STATES)
    domain_names = _as_list(domains, cast=str) or list(DEFAULT_DOMAINS)
    step_filter = set(_as_list(steps, cast=int)) if steps else None
    date_filter = set(_as_list(dates, cast=str)) if dates else None
    edge_margin_km = float(scale_cut_km if scale_edge_margin_km is None else scale_edge_margin_km)

    pred_files = find_predictions(predictions_dir)
    if step_filter is not None:
        pred_files = [p for p in pred_files if int(p.step) in step_filter]
    if date_filter is not None:
        pred_files = [p for p in pred_files if str(p.date) in date_filter]
    if not pred_files:
        raise ValueError(f"No prediction files matched filters in {predictions_dir}")

    rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pooled: dict[tuple[int, str, str, str], PooledMoments] = defaultdict(PooledMoments)
    hist_counts: dict[tuple[int, str, str], np.ndarray] = {}
    hist_dates: dict[tuple[int, str, str], int] = defaultdict(int)
    # precipitation only: valid points, false-drizzle points, missed-rain points
    dry_counts: dict[tuple[int, str, str], np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    dry_threshold = float(rank_dry_threshold_mm) / DRY_THRESHOLD_MM_TO_NATIVE
    missing_fields: set[str] = set()
    clipped_fields: set[str] = set()
    input_note: dict[str, Any] = {}
    operator: LowPassOperator | None = None
    operator_key: tuple | None = None
    n_members_seen: set[int] = set()
    residuals: dict[tuple[int, str, str, str], PooledResiduals] = defaultdict(PooledResiduals)
    residual_note = "scale_split off: no residual bands"

    for pred in pred_files:
        with xr.open_dataset(pred.path, cache=False, decode_timedelta=False) as ds:
            for required in ("y_pred", "y", "weather_state"):
                if required not in ds:
                    raise ValueError(f"{pred.path}: missing required variable {required!r}")
            y_pred = _to_member_point_weather(ds["y_pred"], ds, label="y_pred")
            y_true = _to_member_point_weather(ds["y"], ds, label="y")
            if y_true.sizes["member"] > 1:
                y_true = y_true.isel(member=0, drop=True).expand_dims(member=[0])
            x_in, note = _input_members(ds, include_input)
            input_note = input_note or note
            x_ref = None
            if scale_split:
                x_ref, residual_note = _input_reference(ds, int(y_pred.sizes["member"]))
            ws_index = _weather_state_index(ds)
            n_points = int(y_pred.sizes["grid_point_hres"])
            weights = _area_weights(ds, n_points)
            lat, lon = _lat_lon(ds, n_points)
            edge = box_edge_mask(lat, lon, float(box_edge_deg))
            domain_masks = {name: _domain_mask(name, lat, lon) & edge for name in domain_names}

            inner = np.ones(n_points, dtype=bool)
            if scale_split:
                # one operator per grid: the files of a run share their point set
                key = (n_points, float(lat[0]), float(lon[0]), float(lat[-1]), float(lon[-1]))
                if operator is None or operator_key != key:
                    operator = LowPassOperator(lat, lon, weights, cut_km=float(scale_cut_km),
                                               grid_km=float(scale_grid_km))
                    operator_key = key
                    LOG.info("probabilistic: P_L built for %d points (%s)", n_points, operator.describe()["method"])
                inner = inner_mask(lat, lon, edge_margin_km)
            band_masks = {name: mask & inner for name, mask in domain_masks.items()}

            for field in fields:
                pred_values = _clip_precip(field, _select_field(y_pred, ws_index, field))
                truth_values = _clip_precip(field, _select_field(y_true, ws_index, field))
                if pred_values is None or truth_values is None:
                    missing_fields.add(field)
                    skipped.append({
                        "path": str(pred.path),
                        "date": pred.date,
                        "step": pred.step,
                        "weather_state": field,
                        "reason": "missing field or derived-field inputs",
                    })
                    continue
                if field in PRECIP_FIELDS:
                    clipped_fields.add(field)
                if truth_values.ndim == 2:
                    truth_point = truth_values[0]
                else:
                    truth_point = np.asarray(truth_values).reshape(-1)
                input_values = _clip_precip(field, _select_field(x_in, ws_index, field)) if x_in is not None else None
                n_members = int(pred_values.shape[0])
                n_members_seen.add(n_members)

                bands = _band_fields(pred_values, truth_point, input_values, operator)
                ref_values = _clip_precip(field, _select_field(x_ref, ws_index, field)) if x_ref is not None else None
                if ref_values is not None and ref_values.shape[0] > 1 and np.array_equal(ref_values, np.broadcast_to(ref_values[:1], ref_values.shape), equal_nan=True):
                    ref_values = ref_values[:1]      # one input copied per member (conv/ files): low-pass it once
                if ref_values is not None and operator is not None:
                    res_bands = _residual_band_values(pred_values, truth_point, ref_values, operator,
                                                      bands["large"][0], bands["large"][1])
                    for band, (model_res, truth_res) in res_bands.items():
                        for domain_name, mask in band_masks.items():
                            valid = mask & np.all(np.isfinite(model_res), axis=0) & np.all(np.isfinite(truth_res), axis=0)
                            if not np.any(valid):
                                continue
                            residuals[(int(pred.step), field, domain_name, band)].add_date(
                                _residual_moments(model_res, truth_res, np.where(valid, weights, 0.0)),
                                n_points=int(valid.sum()), n_members=int(pred_values.shape[0]),
                                n_input_members=int(ref_values.shape[0]))
                    del res_bands
                pieces: dict[str, Any] = {}
                for band, (members, truth, input_members) in bands.items():
                    masks = domain_masks if band == "full" else band_masks
                    if band == "full_inner":
                        pieces = pieces or _band_components(members, truth, input_members, spread_ddof=spread_ddof)
                    else:
                        pieces = _band_components(members, truth, input_members, spread_ddof=spread_ddof)
                    scored = _score_band(pieces, weights=weights, masks=masks)
                    for domain_name in masks:
                        if domain_name not in scored:
                            skipped.append({
                                "path": str(pred.path),
                                "date": pred.date,
                                "step": pred.step,
                                "weather_state": field,
                                "domain": domain_name,
                                "band": band,
                                "reason": "no valid points" if band == "full" else "no valid points inside the edge margin",
                            })
                            continue
                        s = scored[domain_name]
                        common = {"date": pred.date, "step": int(pred.step), "weather_state": field,
                                  "domain": domain_name, "n_points": s["n_points"], "n_members": n_members,
                                  "source_path": str(pred.path)}
                        if band == "full":
                            # the historical rows, unchanged
                            domain_weights = np.where(s["mask"], weights, 0.0)
                            for metric in METRICS:
                                value = _weighted_mean(s["components"][metric], domain_weights)
                                if metric == "rmse_ens_mean" and math.isfinite(value):
                                    value = math.sqrt(value)
                                rows.append({**common, "metric": metric, "value": value})
                        for metric in BAND_METRICS + (("spread_input_rms",) if "spread_input_rms" in s else ()):
                            band_rows.append({**common, "band": band, "metric": metric, "value": s[metric]})
                        pooled[(int(pred.step), field, domain_name, band)].add_date(
                            mean_variance=s["ms_spread"], mean_sq_error=s["ms_err"], n_points=s["n_points"],
                            n_members=n_members, fcrps=s["fcrps"], crps=s["crps"],
                            input_variance=s.get("ms_spread_input"),
                            input_members=0 if input_members is None else int(input_members.shape[0]))

                if rank_histogram and n_members > 1:
                    rank_pred, rank_truth = pred_values, truth_point
                    if field in PRECIP_FIELDS:
                        # the dry floor: trace amounts become exact zeros, so they tie with a dry truth
                        rank_pred, rank_truth = dry_floor(pred_values, dry_threshold), dry_floor(truth_point, dry_threshold)
                        disagree = dry_wet_disagreement(pred_values, truth_point, dry_threshold)
                    ranks = truth_ranks(rank_pred, rank_truth, tie_rng(pred.date, pred.step, field, seed=rank_tie_seed))
                    for domain_name, domain_mask in domain_masks.items():
                        counts = rank_counts(ranks, n_members, domain_mask)
                        if counts.sum() == 0:
                            continue
                        key = (int(pred.step), field, domain_name)
                        hist_counts[key] = hist_counts.get(key, 0) + counts
                        hist_dates[key] += 1
                        if field in PRECIP_FIELDS:
                            dry_counts[key] += np.array([int((disagree["valid"] & domain_mask).sum()),
                                                         int((disagree["false_drizzle"] & domain_mask).sum()),
                                                         int((disagree["missed_rain"] & domain_mask).sum())])
                del bands

    if not rows:
        raise ValueError(
            f"Probabilistic scoring produced no rows for {predictions_dir}; skipped={skipped[:5]}"
        )
    wanted_missing = sorted(missing_fields & set(fields))
    if "tp" in wanted_missing:
        LOG.warning("probabilistic: the files carry no tp (precipitation is in every run by the owner's rule); "
                    "tp rows are missing from %s", output_dir)

    summary_rows = _summarize(rows)
    calibration_rows = [
        {"step": step, "weather_state": ws, "domain": domain, "band": band,
         **moments.summary(bootstrap_resamples=int(ratio_bootstrap_resamples), bootstrap_seed=int(ratio_bootstrap_seed))}
        for (step, ws, domain, band), moments in sorted(pooled.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], BANDS.index(kv[0][3])))
        if moments.n_dates
    ]
    residual_rows = []
    for (step, ws, domain, band), pooled_res in sorted(residuals.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], RESIDUAL_BANDS.index(kv[0][3]))):
        whole = residuals.get((step, ws, domain, "full_inner"))
        residual_rows.append({"step": step, "weather_state": ws, "domain": domain, "band": band,
                              **pooled_res.summary(whole.mean_truth if whole is not None else None)})
    n_bins_max = max((c.size for c in hist_counts.values()), default=0)
    hist_rows = []
    for (step, ws, domain), counts in sorted(hist_counts.items()):
        summary = summarize_histogram(counts)
        dry: dict[str, Any] = {"dry_threshold_mm": "", "false_drizzle_rate": "", "missed_rain_rate": ""}
        if ws in PRECIP_FIELDS:
            n_valid, n_drizzle, n_missed = (int(v) for v in dry_counts[(step, ws, domain)])
            dry = {"dry_threshold_mm": float(rank_dry_threshold_mm),
                   "false_drizzle_rate": n_drizzle / n_valid if n_valid else math.nan,
                   "missed_rain_rate": n_missed / n_valid if n_valid else math.nan}
        hist_rows.append({
            "step": step, "weather_state": ws, "domain": domain, "n_members": int(counts.size - 1),
            "n_bins": int(counts.size), "n_dates": hist_dates[(step, ws, domain)],
            "n_points_total": int(counts.sum()), **summary, "shape": histogram_shape(summary), **dry,
            **{f"bin_{i}": int(counts[i]) if i < counts.size else "" for i in range(n_bins_max)},
        })

    score_csv = output_dir / "scores_by_lead.csv"
    summary_csv = output_dir / "summary_by_lead.csv"
    band_csv = output_dir / "band_scores_by_lead.csv"
    calibration_csv = output_dir / "calibration_by_lead.csv"
    hist_csv = output_dir / "rank_histograms.csv"
    residual_csv = output_dir / "residual_bands_by_lead.csv"
    skipped_json = output_dir / "skipped.json"
    summary_json = output_dir / "probabilistic_summary.json"

    _write_csv(
        score_csv,
        rows,
        ["date", "step", "weather_state", "domain", "metric", "value", "n_points", "n_members", "source_path"],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        ["step", "weather_state", "domain", "metric", "mean", "std", "stderr", "n_dates", "n_points_total"],
    )
    _write_csv(
        band_csv,
        band_rows,
        ["date", "step", "weather_state", "domain", "band", "metric", "value", "n_points", "n_members", "source_path"],
    )
    _write_csv(
        calibration_csv,
        calibration_rows,
        ["step", "weather_state", "domain", "band", "n_members", "n_dates", "n_points_total", "fcrps_mean",
         "fcrps_stderr", "crps_mean", "spread_rms", "rmse_ens_mean", "spread_skill_ratio",
         "spread_skill_ratio_corrected", "spread_skill_ratio_corrected_lo95", "spread_skill_ratio_corrected_hi95",
         "spread_skill_ratio_corrected_se", "correction_factor", "spread_input_rms", "input_n_members",
         "spread_input_share"],
    )
    _write_csv(
        hist_csv,
        hist_rows,
        ["step", "weather_state", "domain", "n_members", "n_bins", "n_dates", "n_points_total",
         "bin_0_fraction", "bin_m_fraction", "bin_expected", "outer_two_fraction", "outer_two_expected",
         "flatness_rms", "rank_mean_norm", "chi2_per_bin", "shape",
         "dry_threshold_mm", "false_drizzle_rate", "missed_rain_rate",
         *[f"bin_{i}" for i in range(n_bins_max)]],
    )
    if residual_rows:
        _write_csv(residual_csv, residual_rows, list(RESIDUAL_COLUMNS))
    skipped_json.write_text(json.dumps(skipped, indent=2) + "\n")
    n_members = max(n_members_seen) if n_members_seen else 0
    payload = {
        "schema_version": "1.3",
        "predictions_dir": str(predictions_dir),
        "score_csv": str(score_csv),
        "summary_csv": str(summary_csv),
        "band_scores_csv": str(band_csv),
        "calibration_csv": str(calibration_csv),
        "rank_histograms_csv": str(hist_csv),
        "residual_bands_csv": str(residual_csv) if residual_rows else None,
        "residual_bands_note": residual_note,
        "n_files": len(pred_files),
        "n_rows": len(rows),
        "n_summary_rows": len(summary_rows),
        "n_band_rows": len(band_rows),
        "weather_states": fields,
        "domains": domain_names,
        "steps": sorted({int(p.step) for p in pred_files}),
        "dates": sorted({str(p.date) for p in pred_files}),
        "skipped_count": len(skipped),
        "missing_fields": wanted_missing,
        "precip_clipped_at_zero": sorted(clipped_fields),
        "n_members": n_members,
        "spread_ddof": int(spread_ddof),
        "correction_factor": math.sqrt((n_members + 1.0) / n_members) if n_members else None,
        "box_edge_deg": float(box_edge_deg),
        "bands": list(BANDS) if scale_split else ["full"],
        "bands_note": "full: every point of the region minus the box-edge strip; full_inner, large, fine: the "
                      "points inside the P_L edge margin. NOT additive (P_L is a taper): full_inner is not "
                      "large + fine.",
        "ratio_bootstrap": {"resamples": int(ratio_bootstrap_resamples), "seed": int(ratio_bootstrap_seed),
                            "level": 0.95, "unit": "dates resampled with replacement; pooled as the point estimate"},
        "scale_split": {
            "enabled": bool(scale_split),
            "edge_margin_km": edge_margin_km if scale_split else None,
            **(operator.describe() if operator is not None else {"cut_km": float(scale_cut_km)}),
        },
        "rank_histogram": {"enabled": bool(rank_histogram), "tie_seed": int(rank_tie_seed),
                           "n_bins": n_members + 1 if n_members else None,
                           "ties": "all ties broken uniformly at random (covers zero precipitation)",
                           "dry_threshold_mm": float(rank_dry_threshold_mm),
                           "dry_threshold_native": dry_threshold,
                           "dry_threshold_note": "tp/cp below the threshold set to exactly 0 before ranking "
                                                 "(files in metres: 0.1 mm = 1e-4 m); also defines the "
                                                 "false-drizzle and missed-rain rates"},
        "input_members": input_note,
        "headline_metrics": _headline_metrics(summary_rows, calibration_rows),
    }
    summary_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload
